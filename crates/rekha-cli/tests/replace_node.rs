//! A member that loses its disk comes back, or says why it cannot.
//!
//! This is the question every replacement design depends on, and it is not
//! answerable by reading openraft's source: a *voter* whose storage is empty is
//! indistinguishable, to the protocol, from a voter whose log was truncated.
//! Either the group backtracks it to the leader's log and ships it a snapshot —
//! in which case replacing a dead node is "wipe, restart, wait" and no data has
//! to move — or it must be removed from the configuration first and re-added as
//! a learner, which is a vnode dance under quorum pressure.
//!
//! Measured, on three real processes with R=3 so every shard is on every node:
//! a node is killed, its data directory deleted, and it is started again with
//! the same id and the same `--peers`.
//!
//! Both outcomes are acceptable; silently continuing with a node that believes
//! it is a replica while holding nothing is not. So the assertion is on what a
//! client sees: the collection is fully readable afterwards, and the restarted
//! node holds the data again.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const NODES: usize = 3;

fn temp_dir() -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    // Leak the guard: the directory must outlive the child processes.
    let path = dir.path().to_path_buf();
    std::mem::forget(dir);
    path
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_voter_whose_disk_was_wiped_catches_up_again() {
    let _serial = SERIAL.lock().await;

    let dirs: Vec<PathBuf> = (0..NODES).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();
    let mut nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "wiped", "dimension": DIM, "distance": "l2"}),
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

    // R=3 over three nodes clamps to every node, so the assertion is about this
    // node holding the data, not about which shard it owns.
    eventually("the data to reach every node", || {
        nodes.iter().all(|node| has_data(node, collection))
    })
    .await;

    // Kill the last node and delete everything it had.
    let mut victim = nodes.pop().expect("a node to kill");
    let victim_id = victim.id;
    let victim_port = victim.port;
    let victim_dir = victim.data_dir.clone();
    let _ = victim.child.kill();
    let _ = victim.child.wait();
    drop(victim);
    std::fs::remove_dir_all(&victim_dir).expect("wipe");
    assert!(!victim_dir.exists(), "the directory must be gone");

    // It comes back with the identity and topology it had, and nothing else.
    let restarted = spawn_node(
        &victim_dir,
        victim_port,
        victim_id,
        &peers_spec(&ports),
        false,
    );
    wait_ready(std::slice::from_ref(&restarted)).await;

    // The property an operator cares about: the data is back on that node
    // without anyone copying a file.
    let deadline = Instant::now() + Duration::from_secs(30);
    while !has_data(&restarted, collection) {
        assert!(
            Instant::now() < deadline,
            "node {victim_id} never re-materialized collection {collection} after its disk \
             was wiped; a same-id replacement does not converge, so a replacement has to be \
             removed and re-added as a learner"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // And the collection reads back, from the node that has just relearned it.
    let (status, body) = post_json_retrying(
        &restarted,
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count on the replaced node: {body}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["count"].as_u64(),
        Some(2),
        "the replaced node must serve the records it relearned: {body}"
    );
}
