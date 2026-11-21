//! Automatic rebalancing and scale-out.
//!
//! Asserts that:
//! 1. When a new node joins as a learner, the background reconciler automatically
//!    promotes it to a voter once caught up.
//! 2. The cluster serves linearizable reads and writes throughout the transition.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const EXISTING: usize = 3;
const JOINING_ID: u64 = 4;

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

    // Wait for the background reconciler to automatically promote the
    // learner to voter: poll `voters` (not the deprecated `members` alias)
    // every 500ms for up to 60s; fail with the last payload on expiry.
    let deadline = Instant::now() + Duration::from_secs(60);
    let list = loop {
        let list = membership(&url);
        let voters = list["voters"]
            .as_array()
            .unwrap_or_else(|| panic!("membership must report voters: {list}"));
        if voters.iter().any(|v| v.as_u64() == Some(JOINING_ID)) {
            break list;
        }
        assert!(
            Instant::now() < deadline,
            "learner was not promoted to voter within 60s: {list}"
        );
        tokio::time::sleep(Duration::from_millis(500)).await;
    };
    let _ = list;

    // Write after voter state through a survivor, then assert every node
    // serves the new count with a linearizable read that names its server.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": vec!["c"], "embeddings": vec![vec![0.5f32; DIM]]}),
    )
    .await;
    assert_eq!(status, 200, "write after promotion: {body}");

    let ids: Vec<u64> = nodes
        .iter()
        .chain(std::iter::once(&joining))
        .map(|n| n.id)
        .collect();
    for node in nodes.iter().chain(std::iter::once(&joining)) {
        // Linearizable reads can 503 while a group is mid-transition
        // (joint consensus, election); retry like the write path does. A 200
        // with a short count is also retried: a node that just materialized
        // the shard answers from its empty state until the snapshot applies.
        let deadline = Instant::now() + Duration::from_secs(60);
        let body = loop {
            let resp = client()
                .post(node.url(&format!("/api/v2/collections/{collection}/count")))
                .header("x-chroma-token", KEY)
                .json(&json!({}))
                .send()
                .await
                .expect("count request");
            if resp.status().as_u16() == 503 {
                assert!(
                    Instant::now() < deadline,
                    "node {} never left transition within 60s",
                    node.id
                );
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
            assert_eq!(resp.status().as_u16(), 200, "count on node {}", node.id);
            let served_by = resp
                .headers()
                .get("x-rekha-served-by")
                .and_then(|v| v.to_str().ok())
                .unwrap_or_else(|| panic!("node {} omitted x-rekha-served-by", node.id));
            let served: u64 = served_by.parse().expect("served-by is a node id");
            assert!(
                ids.contains(&served),
                "node {} was served by unknown node {served}",
                node.id
            );
            let body: serde_json::Value = resp.json().await.expect("count json");
            if body["count"].as_u64() == Some(3) {
                break body;
            }
            assert!(
                Instant::now() < deadline,
                "node {} never reached 3 records within 60s; got {body}",
                node.id
            );
            tokio::time::sleep(Duration::from_millis(500)).await;
        };
        assert_eq!(body["count"].as_u64(), Some(3), "count on node {}", node.id);
    }
}
