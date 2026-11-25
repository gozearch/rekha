//! Offline maintenance commands: `verify`, `backup`, `restore`.
//!
//! These read and write a data directory without a running server, which makes
//! them exactly the commands an operator reaches for when something is already
//! wrong. A round trip proves the three agree about what a collection's on-disk
//! state is.

mod common;

use std::process::Command;

use common::{KEY, SECRET, SERIAL, free_port, post_json, spawn_cluster, temp_dir, wait_ready};
use serde_json::json;

fn rekha(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .args(args)
        .env("REKHA_API_KEY", KEY)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("RUST_LOG", "error")
        .output()
        .expect("run rekha");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_data_dir_verifies_backs_up_and_restores() {
    let _guard = SERIAL.lock().await;
    let dir = temp_dir();
    let port = free_port();
    let mut nodes = spawn_cluster(std::slice::from_ref(&dir), &[port]);
    wait_ready(&nodes).await;

    let (status, body) = post_json(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "docs", "dimension": 4}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let id = serde_json::from_str::<serde_json::Value>(&body).unwrap()["id"]
        .as_str()
        .unwrap()
        .to_owned();
    let (status, body) = post_json(
        &nodes[0],
        &format!("/api/v2/collections/{id}/add"),
        json!({"ids": ["a"], "embeddings": [[1.0, 0.0, 0.0, 0.0]]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");

    // Stop the writer: these commands read the directory, and a live server is
    // the one process that may still be appending to it.
    let _ = nodes[0].child.kill();
    let _ = nodes[0].child.wait();

    let dir_str = dir.to_str().unwrap();
    let (code, out, err) = rekha(&["verify", "--data-dir", dir_str]);
    assert_eq!(code, 0, "verify failed: {err}");
    assert!(out.contains("clean"), "{out}");

    let (code, out, err) = rekha(&["verify", "--data-dir", dir_str, "--json"]);
    assert_eq!(code, 0, "verify --json failed: {err}");
    let report: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(report["clean"], true);
    assert_eq!(report["collections"], 1);
    assert!(report["findings"].as_array().unwrap().is_empty());

    let backup = temp_dir().join("backup");
    let backup_str = backup.to_str().unwrap();
    let (code, out, err) = rekha(&["backup", "--data-dir", dir_str, "--out", backup_str]);
    assert_eq!(code, 0, "backup failed: {err}");
    assert!(out.contains("backup:"), "{out}");

    let restored = temp_dir().join("restored");
    let restored_str = restored.to_str().unwrap();
    let (code, out, err) = rekha(&["restore", "--data-dir", restored_str, "--from", backup_str]);
    assert_eq!(code, 0, "restore failed: {err}");
    assert!(out.contains("restore:"), "{out}");

    let (code, out, err) = rekha(&["verify", "--data-dir", restored_str]);
    assert_eq!(code, 0, "restored directory failed verification: {err}");
    assert!(out.contains("clean"), "{out}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn maintenance_commands_refuse_bad_paths() {
    let _guard = SERIAL.lock().await;

    let missing = temp_dir().join("absent");
    let (code, _out, err) = rekha(&["verify", "--data-dir", missing.to_str().unwrap()]);
    assert_eq!(code, 1, "a missing directory is not clean");
    assert!(!err.is_empty());

    let (code, _out, err) = rekha(&[
        "backup",
        "--data-dir",
        missing.to_str().unwrap(),
        "--out",
        temp_dir().join("never").to_str().unwrap(),
    ]);
    assert_eq!(code, 1);
    assert!(!err.is_empty());

    let (code, _out, err) = rekha(&[
        "restore",
        "--data-dir",
        temp_dir().join("target").to_str().unwrap(),
        "--from",
        missing.to_str().unwrap(),
    ]);
    assert_eq!(code, 1);
    assert!(!err.is_empty());
}
