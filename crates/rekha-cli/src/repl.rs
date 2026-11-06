//! Interactive Vector Command REPL for RekhaDB.
//!
//! Provides a `redis-cli` / `mongosh`-style terminal shell for interacting
//! with RekhaDB collections, vectors, nearest-neighbor queries, and cluster status.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use serde_json::Value;

/// Context maintained across REPL commands.
struct ReplContext {
    base_url: String,
    api_key: Option<String>,
    cluster_secret: Option<String>,
    client: reqwest::Client,
    current_collection: Option<CollectionInfo>,
}

#[derive(Debug, Clone)]
struct CollectionInfo {
    id: String,
    name: String,
    dimension: Option<usize>,
}

impl ReplContext {
    fn new(base_url: String, api_key: Option<String>, cluster_secret: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            cluster_secret,
            client: reqwest::Client::new(),
            current_collection: None,
        }
    }

    fn auth_request(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref key) = self.api_key {
            builder = builder
                .header("x-chroma-token", key)
                .header("x-api-key", key);
        }
        builder
    }

    fn cluster_request(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder = self.auth_request(builder);
        if let Some(ref sec) = self.cluster_secret {
            builder = builder.header("x-rekha-cluster-secret", sec);
        }
        builder
    }

    async fn fetch_collections(&self) -> Result<Vec<CollectionInfo>, String> {
        let url = format!("{}/api/v2/collections", self.base_url);
        let resp = self
            .auth_request(self.client.get(url))
            .send()
            .await
            .map_err(|e| format!("Network error: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Failed to list collections ({status}): {body}"));
        }

        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("Invalid JSON response: {e}"))?;

        let arr = body
            .as_array()
            .ok_or_else(|| "Expected JSON array of collections".to_string())?;

        let mut collections = Vec::new();
        for item in arr {
            if let (Some(id), Some(name)) = (item["id"].as_str(), item["name"].as_str()) {
                let dimension = item["dimension"].as_u64().map(|d| d as usize);
                collections.push(CollectionInfo {
                    id: id.to_string(),
                    name: name.to_string(),
                    dimension,
                });
            }
        }
        Ok(collections)
    }

    async fn resolve_collection(&self, name_or_id: &str) -> Result<CollectionInfo, String> {
        let collections = self.fetch_collections().await?;
        if let Some(c) = collections
            .iter()
            .find(|c| c.name == name_or_id || c.id == name_or_id)
        {
            return Ok(c.clone());
        }
        Err(format!("Collection '{name_or_id}' not found"))
    }
}

/// Tokenize an input command line into whitespace-separated arguments,
/// preserving quotes (`"..."`, `'...'`) and bracketed blocks (`[...]`, `{...}`).
pub fn tokenize_line(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];

        if c.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            i += 1;
            continue;
        }

        if c == '"' || c == '\'' {
            let quote = c;
            i += 1;
            while i < len && chars[i] != quote {
                if chars[i] == '\\' && i + 1 < len {
                    i += 1;
                    current.push(chars[i]);
                } else {
                    current.push(chars[i]);
                }
                i += 1;
            }
            if i < len && chars[i] == quote {
                i += 1;
            }
            continue;
        }

        if c == '[' || c == '{' {
            let (open, close) = if c == '[' { ('[', ']') } else { ('{', '}') };
            let mut depth = 1;
            current.push(c);
            i += 1;
            while i < len && depth > 0 {
                let ch = chars[i];
                if ch == open {
                    depth += 1;
                } else if ch == close {
                    depth -= 1;
                }
                current.push(ch);
                i += 1;
            }
            continue;
        }

        current.push(c);
        i += 1;
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn parse_f32_vec(text: &str) -> Result<Vec<f32>, String> {
    let trimmed = text.trim();
    let inner = trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(trimmed);

    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }

    let mut vec = Vec::new();
    for part in inner.split(',') {
        let val = part
            .trim()
            .parse::<f32>()
            .map_err(|e| format!("Invalid vector number '{part}': {e}"))?;
        vec.push(val);
    }
    Ok(vec)
}

/// Run the interactive vector REPL.
pub async fn run_repl(
    url: String,
    initial_collection: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("REKHA_API_KEY").ok();
    let cluster_secret = std::env::var("REKHA_CLUSTER_SECRET").ok();

    let mut ctx = ReplContext::new(url.clone(), api_key, cluster_secret);

    println!("===========================================================");
    println!("RekhaDB Interactive Vector REPL");
    println!("Connected to: {url}");
    println!("Type 'help' for command reference, 'exit' or Ctrl-D to quit.");
    println!("===========================================================");

    // Verify connectivity via /health or /api/v2/heartbeat
    match ctx
        .client
        .get(format!("{}/api/v2/heartbeat", ctx.base_url))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            println!("Status: Connected (Server is live)");
        }
        Ok(resp) => {
            println!("Warning: Server returned HTTP {}", resp.status());
        }
        Err(e) => {
            println!("Warning: Could not reach server at {url}: {e}");
        }
    }

    // Set initial collection if specified
    if let Some(col_name) = initial_collection {
        match ctx.resolve_collection(&col_name).await {
            Ok(info) => {
                println!("Active collection set to '{}' ({})", info.name, info.id);
                ctx.current_collection = Some(info);
            }
            Err(_) => {
                println!("Initial collection '{col_name}' does not exist yet.");
            }
        }
    }

    println!();

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        let prompt = match ctx.current_collection {
            Some(ref col) => format!("rekha({})> ", col.name),
            None => "rekha> ".to_string(),
        };

        print!("{prompt}");
        io::stdout().flush()?;

        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            // EOF reached (Ctrl-D)
            println!("\nBye!");
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let tokens = tokenize_line(trimmed);
        if tokens.is_empty() {
            continue;
        }

        let cmd = tokens[0].to_lowercase();
        let args = &tokens[1..];

        let start = Instant::now();

        match cmd.as_str() {
            "exit" | "quit" | "\\q" => {
                println!("Bye!");
                break;
            }
            "help" | "\\h" | "?" => {
                print_help();
            }
            "clear" | "\\clear" | "cls" => {
                print!("\x1b[2J\x1b[1;1H");
                let _ = io::stdout().flush();
            }
            "use" => {
                if args.is_empty() {
                    eprintln!("Usage: use <collection_name_or_id>");
                    continue;
                }
                match ctx.resolve_collection(&args[0]).await {
                    Ok(info) => {
                        println!("Using collection '{}' (id: {})", info.name, info.id);
                        ctx.current_collection = Some(info);
                    }
                    Err(e) => eprintln!("Error: {e}"),
                }
            }
            "collections" | "list" | "\\d" | "\\dt" => {
                handle_list_collections(&ctx).await;
            }
            "describe" => {
                let target = args
                    .first()
                    .cloned()
                    .or_else(|| ctx.current_collection.as_ref().map(|c| c.name.clone()));
                if let Some(target) = target {
                    handle_describe_collection(&ctx, &target).await;
                } else {
                    eprintln!(
                        "Usage: describe <collection_name> (or select one with 'use <name>')"
                    );
                }
            }
            "create" => {
                if args.len() < 2 {
                    eprintln!("Usage: create <name> <dimension> [l2|cosine|ip]");
                    continue;
                }
                let name = &args[0];
                let dim = match args[1].parse::<usize>() {
                    Ok(d) => d,
                    Err(_) => {
                        eprintln!("Error: dimension must be a positive integer");
                        continue;
                    }
                };
                let distance = args.get(2).map(|s| s.as_str()).unwrap_or("l2");
                handle_create_collection(&mut ctx, name, dim, distance).await;
            }
            "drop" => {
                let target = args
                    .first()
                    .cloned()
                    .or_else(|| ctx.current_collection.as_ref().map(|c| c.id.clone()));
                if let Some(target) = target {
                    handle_drop_collection(&mut ctx, &target).await;
                } else {
                    eprintln!("Usage: drop <collection_name_or_id>");
                }
            }
            "count" => {
                let target = args
                    .first()
                    .cloned()
                    .or_else(|| ctx.current_collection.as_ref().map(|c| c.id.clone()));
                if let Some(target) = target {
                    handle_count(&ctx, &target).await;
                } else {
                    eprintln!("Usage: count [collection_name_or_id] (or 'use <name>')");
                }
            }
            "add" | "insert" => {
                handle_add_or_upsert(&ctx, args, false).await;
            }
            "upsert" => {
                handle_add_or_upsert(&ctx, args, true).await;
            }
            "get" => {
                handle_get(&ctx, args).await;
            }
            "delete" | "del" => {
                handle_delete(&ctx, args).await;
            }
            "query" | "search" => {
                handle_query(&ctx, args).await;
            }
            "cluster" | "members" => {
                handle_cluster(&ctx, args).await;
            }
            "status" | "ready" | "health" => {
                handle_status(&ctx).await;
            }
            unknown => {
                eprintln!("Unknown command: '{unknown}'. Type 'help' for command list.");
            }
        }

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 10 {
            println!("(took {:.2?})", elapsed);
        }
    }

    Ok(())
}

fn print_help() {
    println!("\nRekhaDB REPL Command Reference:");
    println!("--------------------------------------------------------------------------------");
    println!("COLLECTION OPERATIONS:");
    println!("  collections, list         List all collections (name, id, dimension)");
    println!("  use <name_or_id>          Set the active collection context");
    println!("  create <name> <dim> [dst] Create a collection (distance: l2, cosine, ip)");
    println!("  describe [name_or_id]     Describe collection metadata & count");
    println!("  drop [name_or_id]         Drop a collection");
    println!("  count [name_or_id]        Return record count for a collection");
    println!();
    println!("VECTOR DATA OPERATIONS (requires active collection or specified context):");
    println!("  add <id> <[f32,...]> [\"doc\"] [{{\"k\":\"v\"}}]");
    println!("                            Insert a vector record");
    println!("  upsert <id> <[f32,...]> [\"doc\"] [{{\"k\":\"v\"}}]");
    println!("                            Upsert a vector record");
    println!("  get <id1> [id2 ...]       Fetch records by ID");
    println!("  delete <id1> [id2 ...]    Delete records by ID");
    println!("  query <[f32,...]> [k=5] [where_json]");
    println!("                            Run nearest-neighbor vector search");
    println!();
    println!("CLUSTER OPERATIONS:");
    println!("  cluster, members          Display cluster nodes, voters, and leader");
    println!("  cluster add <id> <addr>   Admit a learner node to the cluster");
    println!("  cluster remove <id>       Remove a node from the cluster");
    println!("  status, ready, health     Check server readiness and heartbeat");
    println!();
    println!("UTILITIES:");
    println!("  clear                     Clear screen");
    println!("  help, ?                   Show this help menu");
    println!("  exit, quit                Exit the REPL");
    println!("--------------------------------------------------------------------------------\n");
}

async fn handle_list_collections(ctx: &ReplContext) {
    match ctx.fetch_collections().await {
        Ok(cols) => {
            if cols.is_empty() {
                println!("No collections found.");
                return;
            }
            println!("{:<28} {:<10} {:<36}", "NAME", "DIMENSION", "ID");
            println!("{}", "-".repeat(76));
            for c in cols {
                let dim = c.dimension.map_or("-".to_string(), |d| d.to_string());
                println!("{:<28} {:<10} {:<36}", c.name, dim, c.id);
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}

async fn handle_describe_collection(ctx: &ReplContext, name_or_id: &str) {
    match ctx.resolve_collection(name_or_id).await {
        Ok(col) => {
            println!("Collection: {}", col.name);
            println!("ID:         {}", col.id);
            println!(
                "Dimension:  {}",
                col.dimension.map_or("None".to_string(), |d| d.to_string())
            );

            // Fetch count
            let url = format!("{}/api/v2/collections/{}/count", ctx.base_url, col.id);
            if let Ok(resp) = ctx.auth_request(ctx.client.post(url)).send().await
                && let Ok(body) = resp.json::<Value>().await
            {
                println!("Count:      {}", body);
            }
        }
        Err(e) => eprintln!("Error: {e}"),
    }
}

async fn handle_create_collection(
    ctx: &mut ReplContext,
    name: &str,
    dimension: usize,
    distance: &str,
) {
    let url = format!("{}/api/v2/collections", ctx.base_url);
    let body = serde_json::json!({
        "name": name,
        "dimension": dimension,
        "distance": distance,
    });

    let resp = match ctx
        .auth_request(ctx.client.post(url).json(&body))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let created: Value = resp.json().await.unwrap_or_default();
        let id = created["id"].as_str().unwrap_or("").to_string();
        println!("Collection '{name}' created successfully (id: {id})");
        ctx.current_collection = Some(CollectionInfo {
            id,
            name: name.to_string(),
            dimension: Some(dimension),
        });
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error creating collection ({status}): {err}");
    }
}

async fn handle_drop_collection(ctx: &mut ReplContext, name_or_id: &str) {
    let col = match ctx.resolve_collection(name_or_id).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    let url = format!("{}/api/v2/collections/{}", ctx.base_url, col.id);
    let resp = match ctx.auth_request(ctx.client.delete(url)).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        println!("Collection '{}' ({}) dropped.", col.name, col.id);
        if let Some(ref curr) = ctx.current_collection
            && curr.id == col.id
        {
            ctx.current_collection = None;
        }
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error dropping collection ({status}): {err}");
    }
}

async fn handle_count(ctx: &ReplContext, name_or_id: &str) {
    let col = match ctx.resolve_collection(name_or_id).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {e}");
            return;
        }
    };

    let url = format!("{}/api/v2/collections/{}/count", ctx.base_url, col.id);
    let resp = match ctx.auth_request(ctx.client.post(url)).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let count: Value = resp.json().await.unwrap_or_default();
        println!("{count} records in collection '{}'", col.name);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error ({status}): {err}");
    }
}

async fn handle_add_or_upsert(ctx: &ReplContext, args: &[String], is_upsert: bool) {
    let col = match ctx.current_collection.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Error: No active collection. Run 'use <collection>' first.");
            return;
        }
    };

    if args.len() < 2 {
        eprintln!("Usage: add <id> <[f32, ...]> [document] [metadata_json]");
        return;
    }

    let id = &args[0];
    let embedding = match parse_f32_vec(&args[1]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing embedding: {e}");
            return;
        }
    };

    let document = args.get(2).cloned();
    let metadata: Option<Value> = if let Some(m_str) = args.get(3) {
        match serde_json::from_str(m_str) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Error parsing metadata JSON: {e}");
                return;
            }
        }
    } else {
        None
    };

    let mut body = serde_json::json!({
        "ids": vec![id],
        "embeddings": vec![embedding],
    });

    if let Some(doc) = document {
        body["documents"] = serde_json::json!(vec![doc]);
    }
    if let Some(meta) = metadata {
        body["metadatas"] = serde_json::json!(vec![meta]);
    }

    let action = if is_upsert { "upsert" } else { "add" };
    let url = format!("{}/api/v2/collections/{}/{}", ctx.base_url, col.id, action);

    let resp = match ctx
        .auth_request(ctx.client.post(url).json(&body))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let op_name = if is_upsert { "Upserted" } else { "Added" };
        println!("{op_name} record '{id}' into '{}'", col.name);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error ({status}): {err}");
    }
}

async fn handle_get(ctx: &ReplContext, args: &[String]) {
    let col = match ctx.current_collection.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Error: No active collection. Run 'use <collection>' first.");
            return;
        }
    };

    let body = if args.is_empty() {
        serde_json::json!({
            "include": ["documents", "metadatas", "embeddings"]
        })
    } else {
        serde_json::json!({
            "ids": args,
            "include": ["documents", "metadatas", "embeddings"]
        })
    };

    let url = format!("{}/api/v2/collections/{}/get", ctx.base_url, col.id);
    let resp = match ctx
        .auth_request(ctx.client.post(url).json(&body))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let result: Value = resp.json().await.unwrap_or_default();
        print_formatted_records(&result);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error ({status}): {err}");
    }
}

async fn handle_delete(ctx: &ReplContext, args: &[String]) {
    let col = match ctx.current_collection.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Error: No active collection. Run 'use <collection>' first.");
            return;
        }
    };

    if args.is_empty() {
        eprintln!("Usage: delete <id1> [id2 ...]");
        return;
    }

    let body = serde_json::json!({ "ids": args });
    let url = format!("{}/api/v2/collections/{}/delete", ctx.base_url, col.id);

    let resp = match ctx
        .auth_request(ctx.client.post(url).json(&body))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        println!("Deleted {} record(s) from '{}'", args.len(), col.name);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error ({status}): {err}");
    }
}

async fn handle_query(ctx: &ReplContext, args: &[String]) {
    let col = match ctx.current_collection.as_ref() {
        Some(c) => c,
        None => {
            eprintln!("Error: No active collection. Run 'use <collection>' first.");
            return;
        }
    };

    if args.is_empty() {
        eprintln!("Usage: query <[f32, ...]> [n_results=5] [where_json]");
        return;
    }

    let embedding = match parse_f32_vec(&args[0]) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing query embedding: {e}");
            return;
        }
    };

    let n_results = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5);

    let where_filter = if let Some(w) = args.get(2) {
        match serde_json::from_str::<Value>(w) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("Error parsing where JSON filter: {e}");
                return;
            }
        }
    } else {
        None
    };

    let mut body = serde_json::json!({
        "query_embeddings": vec![embedding],
        "n_results": n_results,
        "include": ["distances", "documents", "metadatas"],
    });

    if let Some(w) = where_filter {
        body["where"] = w;
    }

    let url = format!("{}/api/v2/collections/{}/query", ctx.base_url, col.id);
    let resp = match ctx
        .auth_request(ctx.client.post(url).json(&body))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let result: Value = resp.json().await.unwrap_or_default();
        print_query_results(&result);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!("Error ({status}): {err}");
    }
}

async fn handle_cluster(ctx: &ReplContext, args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        None | Some("list" | "status" | "info") => {
            handle_cluster_membership(ctx).await;
        }
        Some("add") => {
            if args.len() < 3 {
                eprintln!("Usage: cluster add <node_id> <addr>");
                return;
            }
            let node_id = match args[1].parse::<u64>() {
                Ok(id) => id,
                Err(_) => {
                    eprintln!("Error: node_id must be a positive integer");
                    return;
                }
            };
            let addr = &args[2];
            let url = format!("{}/internal/cluster/add_learner", ctx.base_url);
            let body = serde_json::json!({ "node_id": node_id, "addr": addr });
            match ctx
                .cluster_request(ctx.client.post(url).json(&body))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let json: Value = resp.json().await.unwrap_or_default();
                    println!("{:#}", json);
                }
                Ok(resp) => {
                    let status = resp.status();
                    let err = resp.text().await.unwrap_or_default();
                    eprintln!("Error ({status}): {err}");
                }
                Err(e) => eprintln!("Network error: {e}"),
            }
        }
        Some("remove") => {
            if args.len() < 2 {
                eprintln!("Usage: cluster remove <node_id>");
                return;
            }
            let node_id = match args[1].parse::<u64>() {
                Ok(id) => id,
                Err(_) => {
                    eprintln!("Error: node_id must be a positive integer");
                    return;
                }
            };
            let url = format!("{}/internal/cluster/remove_member", ctx.base_url);
            let body = serde_json::json!({ "node_id": node_id });
            match ctx
                .cluster_request(ctx.client.post(url).json(&body))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let json: Value = resp.json().await.unwrap_or_default();
                    println!("{:#}", json);
                }
                Ok(resp) => {
                    let status = resp.status();
                    let err = resp.text().await.unwrap_or_default();
                    eprintln!("Error ({status}): {err}");
                }
                Err(e) => eprintln!("Network error: {e}"),
            }
        }
        Some(sub) => {
            eprintln!(
                "Unknown cluster subcommand '{sub}'. Usage: cluster [list | add <id> <addr> | remove <id>]"
            );
        }
    }
}

async fn handle_cluster_membership(ctx: &ReplContext) {
    // Always fetch /ready for consensus role, term, and applied index
    let ready_url = format!("{}/ready", ctx.base_url);
    if let Ok(ready_resp) = ctx.auth_request(ctx.client.get(ready_url)).send().await
        && let Ok(ready_json) = ready_resp.json::<Value>().await
    {
        let status = ready_json["status"].as_str().unwrap_or("-");
        let state = ready_json["state"].as_str().unwrap_or("-");
        let term = ready_json["term"]
            .as_u64()
            .map(|t| t.to_string())
            .unwrap_or_else(|| "-".into());
        let leader_id = ready_json["leader"]["node_id"]
            .as_u64()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".into());
        let applied = ready_json["applied_index"]
            .as_u64()
            .map(|i| i.to_string())
            .unwrap_or_else(|| "-".into());
        println!(
            "Node Status : {status} (role: {state}, term: {term}, leader: {leader_id}, applied_index: {applied})"
        );
    }

    let url = format!("{}/internal/cluster/membership", ctx.base_url);
    let resp = match ctx.cluster_request(ctx.client.get(url)).send().await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Network error: {e}");
            return;
        }
    };

    if resp.status().is_success() {
        let json: Value = resp.json().await.unwrap_or_default();
        println!("{:#}", json);
    } else {
        let status = resp.status();
        let err = resp.text().await.unwrap_or_default();
        eprintln!(
            "Membership details unavailable ({status}): {err}\n\
             (Set REKHA_CLUSTER_SECRET or --cluster-secret to view full membership details)"
        );
    }
}

async fn handle_status(ctx: &ReplContext) {
    let ready_url = format!("{}/ready", ctx.base_url);
    let ready_resp = ctx.auth_request(ctx.client.get(ready_url)).send().await;

    match ready_resp {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            println!("/ready -> HTTP {status}: {body}");
        }
        Err(e) => {
            eprintln!("Failed to check /ready: {e}");
        }
    }
}

fn print_formatted_records(data: &Value) {
    let ids = match data["ids"].as_array() {
        Some(a) => a,
        None => {
            println!("No records returned.");
            return;
        }
    };

    if ids.is_empty() {
        println!("Empty set.");
        return;
    }

    let docs = data["documents"].as_array();
    let metas = data["metadatas"].as_array();

    println!("{:<20} {:<30} {:<30}", "ID", "DOCUMENT", "METADATA");
    println!("{}", "-".repeat(82));

    for (i, id_val) in ids.iter().enumerate() {
        let id_str = id_val.as_str().unwrap_or("-");
        let doc_str = docs
            .and_then(|d| d.get(i))
            .and_then(|v| v.as_str())
            .unwrap_or("-");
        let meta_str = metas
            .and_then(|m| m.get(i))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());

        let doc_trunc = if doc_str.len() > 28 {
            format!("{}...", &doc_str[..25])
        } else {
            doc_str.to_string()
        };
        let meta_trunc = if meta_str.len() > 28 {
            format!("{}...", &meta_str[..25])
        } else {
            meta_str
        };

        println!("{:<20} {:<30} {:<30}", id_str, doc_trunc, meta_trunc);
    }
    println!("({} records)", ids.len());
}

fn print_query_results(data: &Value) {
    // query results format: ids: [[id1, id2, ...]], distances: [[d1, d2, ...]]
    let empty_vec = Vec::new();
    let ids_outer = data["ids"].as_array();
    let ids = ids_outer
        .and_then(|o| o.first())
        .and_then(|a| a.as_array())
        .unwrap_or(&empty_vec);

    if ids.is_empty() {
        println!("No nearest neighbors found.");
        return;
    }

    let dists = data["distances"]
        .as_array()
        .and_then(|o| o.first())
        .and_then(|a| a.as_array());
    let docs = data["documents"]
        .as_array()
        .and_then(|o| o.first())
        .and_then(|a| a.as_array());
    let metas = data["metadatas"]
        .as_array()
        .and_then(|o| o.first())
        .and_then(|a| a.as_array());

    println!(
        "{:<4} {:<18} {:<12} {:<24} {:<24}",
        "#", "ID", "DISTANCE", "DOCUMENT", "METADATA"
    );
    println!("{}", "-".repeat(84));

    for (i, id_val) in ids.iter().enumerate() {
        let id_str = id_val.as_str().unwrap_or("-");
        let dist_str = dists
            .and_then(|d| d.get(i))
            .and_then(|v| v.as_f64())
            .map(|d| format!("{d:.4}"))
            .unwrap_or_else(|| "-".to_string());
        let doc_str = docs
            .and_then(|d| d.get(i))
            .and_then(|v| v.as_str())
            .unwrap_or("-");
        let meta_str = metas
            .and_then(|m| m.get(i))
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());

        let doc_trunc = if doc_str.len() > 22 {
            format!("{}...", &doc_str[..19])
        } else {
            doc_str.to_string()
        };
        let meta_trunc = if meta_str.len() > 22 {
            format!("{}...", &meta_str[..19])
        } else {
            meta_str
        };

        println!(
            "{:<4} {:<18} {:<12} {:<24} {:<24}",
            i + 1,
            id_str,
            dist_str,
            doc_trunc,
            meta_trunc
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_line() {
        let tokens = tokenize_line("use my_collection");
        assert_eq!(tokens, vec!["use", "my_collection"]);

        let tokens = tokenize_line("add doc1 [0.1, 0.2, 0.3] \"some document\" {\"k\": \"v\"}");
        assert_eq!(
            tokens,
            vec![
                "add",
                "doc1",
                "[0.1, 0.2, 0.3]",
                "some document",
                "{\"k\": \"v\"}"
            ]
        );
    }

    #[test]
    fn test_parse_f32_vec() {
        let v = parse_f32_vec("[0.1, 0.2, 0.3]").unwrap();
        assert_eq!(v, vec![0.1f32, 0.2f32, 0.3f32]);

        let v2 = parse_f32_vec("1.0, 2.5, -3.2").unwrap();
        assert_eq!(v2, vec![1.0f32, 2.5f32, -3.2f32]);

        assert!(parse_f32_vec("[]").unwrap().is_empty());
        assert!(parse_f32_vec("not_a_number").is_err());
    }

    #[test]
    fn test_tokenize_quotes_and_brackets() {
        let tokens = tokenize_line("query [0.5, 0.9] 10 {'nested': {'a': 1}}");
        assert_eq!(
            tokens,
            vec!["query", "[0.5, 0.9]", "10", "{'nested': {'a': 1}}"]
        );

        let tokens_empty = tokenize_line("   ");
        assert!(tokens_empty.is_empty());
    }
}
