//! Readiness has to describe the groups this node actually serves.
//!
//! `/ready` consulted only the metadata group's leader. A node whose *shard*
//! group was mid-election therefore advertised ready and then answered `503` to
//! every write for that shard — routing load straight into a window it already
//! knew was bad. Shard leadership is now part of the answer and is reported.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;

const NODES: usize = 3;

fn dirs(count: usize) -> Vec<PathBuf> {
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

async fn ready_body(node: &Node) -> (u16, serde_json::Value) {
    let resp = client()
        .get(node.url("/ready"))
        .send()
        .await
        .expect("ready request");
    let status = resp.status().as_u16();
    let body = resp.json().await.unwrap_or(serde_json::Value::Null);
    (status, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn readiness_reports_shard_leadership_and_turns_false_without_a_quorum() {
    let _serial = SERIAL.lock().await;

    let mut nodes = spawn_cluster(
        &dirs(NODES),
        &(0..NODES).map(|_| free_port()).collect::<Vec<_>>(),
    );
    wait_ready(&nodes).await;

    // Healthy: the node hosts shards, and every one of them has a leader.
    let (status, body) = ready_body(&nodes[0]).await;
    assert_eq!(status, 200, "healthy node not ready: {body}");
    let hosted = body["shards"]["hosted"]
        .as_u64()
        .expect("`shards.hosted` is part of the readiness answer");
    assert!(
        hosted > 0,
        "a clustered node must report the shards it hosts"
    );
    assert_eq!(
        body["shards"]["without_leader"],
        serde_json::json!([]),
        "a healthy node must name no leaderless shard: {body}"
    );

    // Lose the quorum: two of three. The survivor cannot lead any group.
    for node in nodes.iter_mut().take(2) {
        let _ = node.child.kill();
        let _ = node.child.wait();
    }

    let survivor = &nodes[2];
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut last = serde_json::Value::Null;
    let mut last_status = 0;
    while Instant::now() < deadline {
        let (status, body) = ready_body(survivor).await;
        last_status = status;
        last = body;
        if status == 503 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    assert_eq!(
        last_status, 503,
        "a node without a quorum reported ready: {last}"
    );
    assert_eq!(last["ready"], serde_json::json!(false), "{last}");
}
