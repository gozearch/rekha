//! REPL orchestrator — manages input, output, and state.

pub mod prompt;

use reedline::{FileBackedHistory, Reedline, Signal};

use crate::client::{GetRequest, QueryRequest, RekhaClient};
use crate::commands::{self, Command};
use crate::display::{self, OutputFormat};
use crate::query;

pub struct Repl {
    client: RekhaClient,
    collection: Option<String>,
    format: OutputFormat,
    vertical: bool,
    timing: bool,
    pager: bool,
}

impl Repl {
    pub fn new(client: RekhaClient, initial_collection: Option<&str>) -> Self {
        Self {
            client,
            collection: initial_collection.map(|s| s.to_string()),
            format: OutputFormat::Table,
            vertical: false,
            timing: true,
            pager: true,
        }
    }

    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let history =
            FileBackedHistory::with_file(1000, std::path::PathBuf::from(".rekha_history"))
                .unwrap_or_else(|_| FileBackedHistory::new(1000).expect("in-memory history"));
        let mut editor = Reedline::create().with_history(Box::new(history));

        println!("RekhaDB REPL v0.1.0 (Chroma-compatible)");
        println!("Type \\help for help, \\quit to exit.\n");

        loop {
            let prompt = prompt::build_prompt(&self.collection);
            let input = match editor.read_line(&prompt) {
                Ok(Signal::Success(line)) => line,
                Ok(Signal::CtrlC) => {
                    println!("^C");
                    continue;
                }
                Ok(Signal::CtrlD) | Err(_) => {
                    println!("Bye!");
                    break;
                }
                Ok(_) => continue,
            };

            let input = input.trim();
            if input.is_empty() {
                continue;
            }

            if input.starts_with('\\') || input == "exit" || input == "quit" {
                match commands::parse_command(input) {
                    Command::Quit => {
                        println!("Bye!");
                        break;
                    }
                    Command::ListCollections => self.handle_list_collections().await,
                    Command::DescribeCollection(name) => self.handle_describe(&name).await,
                    Command::ToggleVertical => {
                        self.vertical = !self.vertical;
                        println!(
                            "Vertical display {}",
                            if self.vertical { "enabled" } else { "disabled" }
                        );
                    }
                    Command::SetFormat(fmt) => {
                        self.format = match fmt.as_str() {
                            "table" => OutputFormat::Table,
                            "json" => OutputFormat::Json,
                            "csv" => OutputFormat::Csv,
                            "vertical" => OutputFormat::Vertical,
                            _ => {
                                eprintln!(
                                    "Unknown format: {fmt}. Use table, json, csv, or vertical."
                                );
                                continue;
                            }
                        };
                        println!("Format set to {}", self.format);
                    }
                    Command::TogglePager => {
                        self.pager = !self.pager;
                        println!("Pager {}", if self.pager { "enabled" } else { "disabled" });
                    }
                    Command::ToggleTiming => {
                        self.timing = !self.timing;
                        println!(
                            "Timing {}",
                            if self.timing { "enabled" } else { "disabled" }
                        );
                    }
                    Command::CreateCollection(name) => {
                        self.handle_create_collection(&name).await;
                    }
                    Command::UseCollection(name) => {
                        self.collection = Some(name.clone());
                        println!("Now using collection: {name}");
                    }
                    Command::DropCollection(name) => {
                        self.handle_drop_collection(&name).await;
                    }
                    Command::Help => self.print_help(),
                    Command::Clear => print!("\x1b[2J\x1b[1;1H"),
                    _ => eprintln!("Unknown command: {input}"),
                }
                continue;
            }

            let start = std::time::Instant::now();
            match query::parse_query(input) {
                Ok(parsed) => {
                    let result = self.execute_query(parsed).await;
                    let elapsed = if self.timing {
                        Some(start.elapsed())
                    } else {
                        None
                    };

                    match result {
                        Ok(output) => {
                            let timed = match elapsed {
                                Some(d) => {
                                    format!("{output}\n({:.3}s)\n", d.as_secs_f64())
                                }
                                None => output,
                            };
                            print!("{timed}");
                        }
                        Err(e) => eprintln!("Error: {e}"),
                    }
                }
                Err(e) => eprintln!("Parse error: {e}"),
            }
        }

        Ok(())
    }

    async fn execute_query(&self, query: query::sql::ParsedQuery) -> Result<String, String> {
        let tenant = "default_tenant";
        let database = "default_database";

        match query {
            query::sql::ParsedQuery::Select {
                collection,
                nearest_to,
                where_filter,
                limit,
                vertical,
                ..
            } => {
                let collection = if collection == "*" {
                    self.collection.as_deref().unwrap_or("default")
                } else {
                    &collection
                };

                let limit = limit.unwrap_or(10);

                if let Some(embedding) = nearest_to {
                    let req = QueryRequest {
                        query_embeddings: vec![embedding],
                        n_results: Some(limit),
                        where_filter,
                        include: Some(vec![
                            "metadatas".into(),
                            "documents".into(),
                            "distances".into(),
                        ]),
                    };
                    let result = self.client.query(tenant, database, collection, req).await?;
                    let fmt = if vertical {
                        OutputFormat::Vertical
                    } else {
                        self.format.clone()
                    };
                    Ok(display::format_result(&result, &fmt))
                } else {
                    let req = GetRequest {
                        ids: None,
                        where_filter,
                        include: Some(vec!["metadatas".into(), "documents".into()]),
                    };
                    let result = self.client.get(tenant, database, collection, req).await?;

                    let row_count = result.ids.len();
                    if row_count == 0 {
                        return Ok("Empty set".to_string());
                    }

                    let query_result = crate::client::QueryResult {
                        ids: vec![result.ids],
                        distances: None,
                        metadatas: result.metadatas.map(|m| vec![m]),
                        documents: result.documents.map(|d| vec![d]),
                    };
                    let fmt = if vertical {
                        OutputFormat::Vertical
                    } else {
                        self.format.clone()
                    };
                    Ok(display::format_result(&query_result, &fmt))
                }
            }
            query::sql::ParsedQuery::Count { collection } => {
                let collection = if collection == "*" {
                    self.collection.as_deref().unwrap_or("default")
                } else {
                    &collection
                };
                let count = self.client.count(tenant, database, collection).await?;
                Ok(display::format_count(count, None))
            }
            query::sql::ParsedQuery::CreateCollection { name } => {
                self.handle_create_collection(&name).await;
                Ok("Collection created.".to_string())
            }
        }
    }

    async fn handle_list_collections(&self) {
        let tenant = "default_tenant";
        let database = "default_database";

        match self.client.list_collections(tenant, database).await {
            Ok(collections) => {
                if collections.is_empty() {
                    println!("No collections found.");
                    return;
                }
                println!("{:<30} {:<10} {:<10}", "Name", "Dimension", "ID");
                println!("{}", "-".repeat(55));
                for c in &collections {
                    let dim = c.dimension.map_or("-".to_string(), |d| d.to_string());
                    println!(
                        "{:<30} {:<10} {:<10}",
                        c.name,
                        dim,
                        &c.id[..8.min(c.id.len())]
                    );
                }
                println!("\n{} collections", collections.len());
            }
            Err(e) => eprintln!("Error: {e}"),
        }
    }

    async fn handle_describe(&self, name: &str) {
        let tenant = "default_tenant";
        let database = "default_database";

        match self.client.list_collections(tenant, database).await {
            Ok(collections) => {
                if let Some(c) = collections.iter().find(|c| c.name == name) {
                    println!("Collection: {}", c.name);
                    println!("ID: {}", c.id);
                    println!(
                        "Dimension: {}",
                        c.dimension.map_or("Not set".to_string(), |d| d.to_string())
                    );

                    if let Ok(count) = self.client.count(tenant, database, &c.name).await {
                        println!("Records: {count}");
                    }
                } else {
                    eprintln!("Collection not found: {name}");
                }
            }
            Err(e) => eprintln!("Error: {e}"),
        }
    }

    async fn handle_create_collection(&self, name: &str) {
        let tenant = "default_tenant";
        let database = "default_database";

        match self.client.create_collection(tenant, database, name).await {
            Ok(collection) => {
                println!(
                    "Collection '{}' created (id: {})",
                    collection.name,
                    &collection.id[..8.min(collection.id.len())]
                );
            }
            Err(e) => eprintln!("Error creating collection: {e}"),
        }
    }

    async fn handle_drop_collection(&self, name: &str) {
        let tenant = "default_tenant";
        let database = "default_database";

        match self.client.delete_collection(tenant, database, name).await {
            Ok(()) => {
                println!("Collection '{name}' dropped.");
            }
            Err(e) => eprintln!("Error dropping collection: {e}"),
        }
    }

    fn print_help(&self) {
        println!("RekhaDB REPL Commands:");
        println!("  \\d, \\dt          List all collections");
        println!("  \\d <name>        Describe a collection");
        println!("  \\create <name>   Create a new collection");
        println!("  \\use <name>      Switch to a collection");
        println!("  \\drop <name>     Drop a collection");
        println!("  \\x               Toggle vertical output");
        println!("  \\format <fmt>    Set output format (table/json/csv/vertical)");
        println!("  \\pager           Toggle pager");
        println!("  \\timing          Toggle query timing");
        println!("  \\clear           Clear screen");
        println!("  \\help, \\h, ?     Show this help");
        println!("  \\quit, \\q        Exit");
        println!();
        println!("SQL Queries:");
        println!("  CREATE COLLECTION <name>;");
        println!("  SELECT * FROM <collection> LIMIT 10;");
        println!("  SELECT * FROM <collection> NEAREST TO [0.1, 0.2, ...] LIMIT 5;");
        println!("  SELECT * FROM <collection> WHERE category = 'tech' LIMIT 10;");
        println!("  SELECT COUNT(*) FROM <collection>;");
        println!("  SELECT * FROM <collection> NEAREST TO [0.1, ...] VERTICAL;");
    }
}
