//! Reads are linearizable, and the same read is answered by the same node.
//!
//! Before this, `POST .../get`, `/query` and `/count` were allowed on any
//! replica and served from whatever that replica had applied. The comment in the
//! middleware said reads "may be stale", which is not a consistency model: a
//! client that writes a record and immediately reads it from another node can see
//! it missing, and there is no lag number it could check to know.
//!
//! Two observable properties are asserted here:
//!
//! 1. Every node answers a read with the write that just committed.
//! 2. The answer comes from the shard's leader even when the request arrives at
//!    a replica (`x-rekha-served-by`), which is what makes (1) true across a
//!    failover rather than by luck.
//!
//! `?consistency=local` is the explicit opt-out, and it is served locally.

mod common;

use std::path::PathBuf;
use std::process::Stdio;

use common::*;
use reqwest::header::HeaderMap;
use serde_json::json;

const NODES: usize = 3;

fn dirs(count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect()
}

fn spawn(dir: &std::path::Path, port: u16, id: u64, peers: &str, bootstrap: bool) -> Node {
    let child = serve_command(dir, port, id, peers, bootstrap)
        .env("REKHA_SHARDS", "4")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    Node {
        id,
        port,
        data_dir: dir.to_path_buf(),
        child,
    }
}

/// A GET request that returns the headers too.
async fn post_with_headers(
    node: &Node,
    path: &str,
    body: serde_json::Value,
) -> (u16, HeaderMap, serde_json::Value) {
    let resp = client()
        .post(node.url(path))
        .header("x-chroma-token", KEY)
        .json(&body)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let body = resp.json().await.unwrap_or(serde_json::Value::Null);
    (status, headers, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_read_after_a_write_sees_it_from_every_node() {
    let _serial = SERIAL.lock().await;
    let dirs = dirs(NODES);
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();
    let peers = peers_spec(&ports);

    let nodes: Vec<Node> = (0..NODES)
        .map(|i| spawn(&dirs[i], ports[i], (i + 1) as u64, &peers, i == 0))
        .collect();
    wait_ready(&nodes).await;

    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "linear", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let collection: uuid::Uuid =
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

    // Write through node 1, then read through every node. The write is acked
    // after commit, so a replica that answers from applied state is exactly the
    // case that used to lose this read.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ["r1"], "embeddings": [vec![0.5f32; DIM]]}),
    )
    .await;
    assert_eq!(status, 200, "write: {body}");

    let mut served_by: Vec<String> = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let (status, headers, body) = post_with_headers(
            node,
            &format!("/api/v2/collections/{collection}/get"),
            json!({"ids": ["r1"]}),
        )
        .await;
        assert_eq!(status, 200, "read from node {}: {body}", node.id);
        assert!(
            body["ids"]
                .as_array()
                .is_some_and(|ids| ids.iter().any(|id| id == "r1")),
            "node {} answered the read with the wrong record: {body}",
            node.id
        );

        let served = headers
            .get("x-rekha-served-by")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_else(|| panic!("node {} did not say who served the read", node.id))
            .to_string();
        assert_ne!(
            served, "0",
            "node {} reported an impossible serving node",
            node.id
        );
        served_by.push(served);
        assert!(
            index == 0 || served_by[index] == served_by[0],
            "reads were answered by different nodes ({served_by:?}); placement routing is not \
             sending them to one place"
        );
    }

    // With three replicas and one leader, at least one request must have been
    // answered somewhere other than where it arrived. Otherwise this test would
    // pass even with reads served locally.
    assert!(
        nodes.iter().any(|node| node.id.to_string() != served_by[0]),
        "every read was answered by the node it was sent to, so nothing was forwarded"
    );

    // The explicit opt-out is served locally: the header names the node asked.
    let (status, headers, body) = post_with_headers(
        &nodes[2],
        &format!("/api/v2/collections/{collection}/get?consistency=local"),
        json!({"ids": ["r1"]}),
    )
    .await;
    assert_eq!(status, 200, "local read: {body}");
    assert_eq!(
        headers
            .get("x-rekha-served-by")
            .and_then(|v| v.to_str().ok()),
        Some(nodes[2].id.to_string().as_str()),
        "a local read must be served by the node it was sent to"
    );
}
