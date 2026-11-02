//! A node cannot be restarted into a different place.
//!
//! Placement has three inputs: shard count, replication factor, and the voter
//! set. The first two were persisted in `placement.json` and validated at
//! startup; the third was taken from `--peers` on every start. So the one input
//! an operator could change with a flag was also the only one that moves data:
//! a node restarted with a different topology computed owners its peers did not
//! agree with, and then routed — and stored — shards the rest of the cluster had
//! placed elsewhere. Nothing refused it, because nothing compared the two.
//!
//! What is asserted here is what an operator relies on: a node either comes back
//! exactly as it was, or it refuses to come back at all. Silence is the failure
//! mode this file exists to prevent.

mod common;

use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const NODES: usize = 4;

fn temp_dirs(count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            // Leak the guard: the directory must outlive the child processes.
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect()
}

/// Start a node that must be refused, and return everything it said.
///
/// Bounded on purpose: a node that starts anyway would otherwise hang this test
/// on `wait()`, and a test that hangs does not report what broke.
async fn refused_start(dir: &std::path::Path, port: u16, id: u64, peers: &str) -> String {
    let mut child = serve_command(dir, port, id, peers, false)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rekha");

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => {
                let output = child.wait_with_output().expect("collect output");
                assert!(
                    !status.success(),
                    "a node started with a topology it was not placed across; \
                     it should have refused. stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                return String::from_utf8_lossy(&output.stderr).into_owned();
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "the node was still running after 30s: it accepted a topology it was \
                     not placed across instead of refusing"
                );
            }
            None => tokio::time::sleep(Duration::from_millis(50)).await,
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restart_with_a_different_topology_is_refused() {
    let _serial = SERIAL.lock().await;

    let dirs = temp_dirs(NODES);
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let mut nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // Something to lose. Without data whose home matters, "the shards did not
    // move" would be vacuously true.
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "identity", "dimension": DIM, "distance": "l2"}),
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

    let placement_file = dirs[0].join(rekha_cluster::PlacementConfig::FILE);
    let before = std::fs::read(&placement_file).expect("placement file");
    let persisted: rekha_cluster::PlacementConfig =
        serde_json::from_slice(&before).expect("placement file must parse");
    assert_eq!(
        persisted.voters.len(),
        NODES,
        "the voter set is a placement input and must be in the document, not only in --peers"
    );

    // The refusal happens before consensus starts, so it needs no quorum: a
    // cluster that is entirely down is exactly when an operator edits a flag.
    for node in &mut nodes {
        let _ = node.child.kill();
        let _ = node.child.wait();
    }

    // A fifth node's address added to the topology.
    let mut wider = ports.clone();
    wider.push(free_port());
    let stderr = refused_start(&dirs[0], ports[0], 1, &peers_spec(&wider)).await;
    assert!(
        stderr.contains("--peers lists nodes"),
        "the refusal must name the voter mismatch, not fail for an unrelated reason. \
         stderr: {stderr}"
    );

    let after = std::fs::read(&placement_file).expect("placement file");
    assert_eq!(
        before, after,
        "a refused start must leave the placement document untouched, or the next \
         start would disagree with the cluster for a new reason"
    );

    // And a clustered directory with no topology at all is refused rather than
    // served as a single node over every collection.
    let stderr = refused_start(&dirs[0], ports[0], 1, "").await;
    assert!(
        stderr.contains("holds a cluster placement"),
        "an empty --peers on a clustered directory must be refused. stderr: {stderr}"
    );

    // Nothing above may have cost the cluster its data: every node comes back
    // with the topology it was placed across, and the records are still there.
    let restarted = spawn_cluster(&dirs, &ports);
    wait_ready(&restarted).await;

    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (status, body) = post_json_retrying(
            &restarted[0],
            &format!("/api/v2/collections/{collection}/count"),
            json!({}),
        )
        .await;
        assert_eq!(status, 200, "count after restart: {body}");
        let count = serde_json::from_str::<serde_json::Value>(&body).expect("json")["count"]
            .as_u64()
            .expect("count");
        if count == 2 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the collection did not come back after the refused starts: {body}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
