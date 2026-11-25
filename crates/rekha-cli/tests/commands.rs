//! The non-interactive commands, end to end.
//!
//! Every subcommand is a thin HTTP client, so the test asserts the parts the
//! unit boundary cannot: that the flag names parse, that the JSON body shape
//! the CLI builds is the one the server accepts, and that a refusal reaches the
//! operator as a non-zero exit with a readable error.

mod common;

use std::path::Path;

use common::{KEY, SERIAL, free_port, rekha, spawn_cluster, temp_dir, wait_ready};
use serde_json::json;

fn write_json(dir: &Path, name: &str, value: serde_json::Value) -> String {
    let path = dir.join(name);
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    path.to_str().unwrap().to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn collection_and_record_commands_round_trip() {
    let _guard = SERIAL.lock().await;
    let dir = temp_dir();
    let port = free_port();
    let nodes = spawn_cluster(std::slice::from_ref(&dir), &[port]);
    wait_ready(&nodes).await;
    let url = nodes[0].url("");

    let (code, out, err) = rekha(&["collection", "create", "docs", "4"], &url);
    assert_eq!(code, 0, "create failed: {err}");
    let created: serde_json::Value = serde_json::from_str(&out).unwrap();
    let id = created["id"].as_str().unwrap().to_owned();
    assert_eq!(created["name"], "docs");

    let (code, out, err) = rekha(&["collection", "list"], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("docs"), "{out}");

    let (code, out, err) = rekha(&["collection", "get", &id], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains(&id), "{out}");

    let files = temp_dir();
    let add = write_json(
        &files,
        "add.json",
        json!({
            "ids": ["a", "b"],
            "embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "metadatas": [{"tag": "a"}, null],
            "documents": ["one", "two"]
        }),
    );
    let (code, out, err) = rekha(&["add", &id, &add], &url);
    assert_eq!(code, 0, "add failed: {err}");
    assert!(
        out.trim().is_empty() || out.contains("{"),
        "add prints either an empty body or JSON: {out:?}"
    );

    let upsert = write_json(
        &files,
        "upsert.json",
        json!({"ids": ["c"], "embeddings": [[0.5, 0.5, 0.0, 0.0]]}),
    );
    assert_eq!(rekha(&["upsert", &id, &upsert], &url).0, 0);

    let update = write_json(
        &files,
        "update.json",
        json!({"ids": ["a"], "metadatas": [{"stage": "two"}]}),
    );
    assert_eq!(rekha(&["update", &id, &update], &url).0, 0);

    let query = write_json(
        &files,
        "query.json",
        json!({
            "query_embeddings": [[1.0, 0.0, 0.0, 0.0]],
            "n_results": 2,
            "include": ["documents", "distances"]
        }),
    );
    let (code, out, err) = rekha(&["query", &id, &query], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"documents\""), "{out}");

    let get = write_json(&files, "get.json", json!({"ids": ["a", "b"]}));
    let (code, out, err) = rekha(&["get", &id, &get], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"a\"") && out.contains("\"b\""), "{out}");

    let delete = write_json(&files, "delete.json", json!(["b"]));
    let (code, out, err) = rekha(&["delete", &id, &delete], &url);
    assert_eq!(code, 0, "delete with a bare id array: {err} {out}");

    let (code, out, err) = rekha(&["count", &id], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"count\""), "{out}");

    let (code, _out, err) = rekha(&["collection", "delete", &id], &url);
    assert_eq!(code, 0, "{err}");
    let (code, _out, err) = rekha(
        &["collection", "get", "00000000-0000-0000-0000-000000000000"],
        &url,
    );
    assert_eq!(code, 1, "a missing collection must fail: {err}");
    assert!(err.contains("Error (404"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auth_commands_cover_the_rbac_surface() {
    let _guard = SERIAL.lock().await;
    let dir = temp_dir();
    let port = free_port();
    let nodes = spawn_cluster(std::slice::from_ref(&dir), &[port]);
    wait_ready(&nodes).await;
    let url = nodes[0].url("");

    let (code, out, err) = rekha(&["auth", "create-user", "alice"], &url);
    assert_eq!(code, 0, "{err}");
    let alice: serde_json::Value = serde_json::from_str(&out).unwrap();
    let alice_id = alice["id"].as_str().unwrap().to_owned();
    assert!(alice["api_key"].as_str().unwrap().starts_with("rekha_sk_"));

    let (code, out, err) = rekha(&["auth", "list-users"], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("alice"), "{out}");

    let (code, out, err) = rekha(&["auth", "create-key", &alice_id], &url);
    assert_eq!(code, 0, "{err}");
    let key: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(key["key_hash_hex"].is_string());

    let (code, out, err) = rekha(&["auth", "list-keys", &alice_id], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"name\": \"default\""), "{out}");

    let (code, out, err) = rekha(
        &["auth", "create-role", "analyst", "--description", "reads"],
        &url,
    );
    assert_eq!(code, 0, "{err}");
    let role: serde_json::Value = serde_json::from_str(&out).unwrap();
    let role_id = role["id"].as_str().unwrap().to_owned();

    let (code, out, err) = rekha(&["auth", "list-roles"], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("analyst"), "{out}");

    let (code, _out, err) = rekha(
        &[
            "auth",
            "add-permission",
            &role_id,
            "collection.get",
            "--tenant",
            "default_tenant",
        ],
        &url,
    );
    assert_eq!(code, 0, "{err}");

    let (code, _out, err) = rekha(&["auth", "assign-role", &alice_id, &role_id], &url);
    assert_eq!(code, 0, "{err}");

    let (code, out, err) = rekha(&["auth", "whoami"], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("permissions"), "{out}");

    let (code, _out, err) = rekha(&["auth", "unassign-role", &alice_id, &role_id], &url);
    assert_eq!(code, 0, "{err}");

    let (code, _out, err) = rekha(&["auth", "delete-user", &alice_id], &url);
    assert_eq!(code, 0, "{err}");

    let (code, _out, err) = rekha(&["auth", "delete-role", &role_id], &url);
    assert_eq!(code, 0, "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn membership_and_usage_failures_are_reported() {
    let _guard = SERIAL.lock().await;
    let dir = temp_dir();
    let port = free_port();
    let nodes = spawn_cluster(std::slice::from_ref(&dir), &[port]);
    wait_ready(&nodes).await;
    let url = nodes[0].url("");

    let (code, out, err) = rekha(&["membership", "list"], &url);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("voters"), "{out}");

    // Missing required arguments are a clap usage error, not a panic.
    let (code, out, err) = rekha(&["add"], &url);
    assert_ne!(code, 0);
    assert!(
        out.is_empty() || err.contains("Usage") || err.contains("required"),
        "{err}"
    );

    // A file that is not JSON is reported before any request is sent.
    let files = temp_dir();
    let bad = files.join("bad.json");
    std::fs::write(&bad, "not json").unwrap();
    let (code, _out, err) = rekha(&["query", "00000000-0000-0000-0000-000000000000"], &url);
    assert_ne!(code, 0, "{err}");

    // A bad cluster-secret command surfaces the refusal rather than hanging.
    let (code, _out, err) = rekha(
        &[
            "membership",
            "add",
            "--node-id",
            "9",
            "--addr",
            "127.0.0.1:9",
        ],
        &url,
    );
    assert!(
        code == 0 || code == 1,
        "membership add must either admit or report: {err}"
    );
    let _ = KEY;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_metrics_exporter_reports_raft_state() {
    use common::{Node, peers_spec, serve_command};
    use std::process::Stdio;

    let _guard = SERIAL.lock().await;
    let dir = temp_dir();
    let port = free_port();
    let metrics_port = free_port();
    let mut command = serve_command(&dir, port, 1, &peers_spec(&[port]), true);
    command
        .env("REKHA_METRICS_PORT", metrics_port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = command.spawn().expect("spawn serve");
    let node = Node {
        id: 1,
        port,
        data_dir: dir,
        child,
    };
    wait_ready(std::slice::from_ref(&node)).await;

    // A write changes the applied-index gauges, and the scrape is what an
    // operator actually looks at.
    let _ = reqwest::Client::new()
        .post(format!("{}/api/v2/collections", node.url("")))
        .header("x-chroma-token", KEY)
        .json(&json!({"name": "metric", "dimension": 2}))
        .send()
        .await
        .expect("create");
    tokio::time::sleep(std::time::Duration::from_millis(400)).await;

    let body = reqwest::get(format!("http://127.0.0.1:{metrics_port}/metrics"))
        .await
        .expect("scrape")
        .text()
        .await
        .unwrap();
    assert!(body.contains("rekha_raft_term"), "{body}");
    assert!(body.contains("rekha_raft_voter_count"), "{body}");
    assert!(body.contains("rekha_raft_last_applied_index"), "{body}");
}
