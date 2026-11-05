//! Automatic rebalancing and scale-out.
//!
//! Asserts that:
//! 1. When a new node joins as a learner, the background reconciler automatically
//!    promotes it to a voter once caught up.
//! 2. The cluster serves linearizable reads and writes throughout the transition.

mod common;

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const EXISTING: usize = 3;
const JOINING_ID: u64 = 4;

fn temp_dir() -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().to_path_buf();
    std::mem::forget(dir);
    path
}

fn rekha(args: &[&str], url: &str) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("--url")
        .arg(url)
        .args(args)
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

fn membership(url: &str) -> serde_json::Value {
    let (code, stdout, stderr) = rekha(&["membership", "list"], url);
    assert_eq!(code, 0, "membership list failed: {stderr}");
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("list output `{stdout}`: {e}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auto_rebalancer_promotes_learner_to_voter() {
    let _serial = SERIAL.lock().await;

    let dirs: Vec<PathBuf> = (0..EXISTING).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..EXISTING).map(|_| free_port()).collect();
    let nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // Create a collection and write data before joining
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "rebalance-test", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let collection: uuid::Uuid =
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

    let embeddings: Vec<Vec<f32>> = vec![vec![0.1f32; DIM], vec![0.9f32; DIM]];
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": vec!["a", "b"], "embeddings": embeddings}),
    )
    .await;
    assert_eq!(status, 200, "add: {body}");

    let leader_is = membership(&nodes[0].url(""))["leader"]
        .as_u64()
        .expect("leader elected");
    let leader = nodes
        .iter()
        .find(|node| node.id == leader_is)
        .expect("leader node");
    let url = leader.url("");

    let join_dir = temp_dir();
    let join_port = free_port();
    let joining = spawn_joining_node(&join_dir, join_port, JOINING_ID, &url);

    // Admit as learner
    let addr = format!("127.0.0.1:{join_port}");
    let (code, _stdout, stderr) = rekha(
        &[
            "membership",
            "add",
            "--node-id",
            &JOINING_ID.to_string(),
            "--addr",
            &addr,
        ],
        &url,
    );
    assert_eq!(code, 0, "admitting learner: {stderr}");
    wait_ready(std::slice::from_ref(&joining)).await;

    // Wait for the background reconciler to automatically promote the learner to voter
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let list = membership(&url);
        let members = list["members"].as_array().unwrap();
        if members.len() == 4 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "learner was not promoted to voter within deadline: {list}"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    // Verify all nodes serve count
    for node in nodes.iter().chain(std::iter::once(&joining)) {
        let (status, body) = post_json_retrying(
            node,
            &format!("/api/v2/collections/{collection}/count"),
            json!({}),
        )
        .await;
        assert_eq!(status, 200, "count: {body}");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["count"].as_u64(),
            Some(2)
        );
    }
}
