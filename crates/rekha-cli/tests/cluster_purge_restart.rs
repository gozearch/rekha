//! Restart after a **shard** group's Raft log has been purged.
//!
//! The single-node case is covered by `purge_restart.rs`. This one covers the
//! part that only exists in a cluster: a shard group's membership and placement
//! live in the log prefix a snapshot purges, so a restarted owner has to
//! reconstruct both from durable sources or it cannot elect, cannot serve, and —
//! worse — would derive an empty snapshot scope and archive no collections on
//! the next snapshot.
//!
//! Four nodes with three replicas is the smallest cluster where a shard has an
//! owner that another node does not hold, which is what makes "this node must
//! still serve its own shard after restart" a real assertion.

mod common;

use std::path::PathBuf;
use std::process::Stdio;

use common::*;
use serde_json::json;

const NODES: usize = 4;
/// Low enough that the purge path runs in seconds rather than after thousands
/// of entries. `max_in_snapshot_log_to_keep` is set to the same value, so the
/// log is compacted once it is roughly twice this.
const SNAPSHOT_EVERY: &str = "20";
/// Batches written after the collection exists. Each is one log entry in the
/// owning shard group, so this crosses the snapshot and purge thresholds.
const BATCHES: usize = 60;

fn spawn(dir: &std::path::Path, port: u16, id: u64, peers: &str, bootstrap: bool) -> Node {
    let child = serve_command(dir, port, id, peers, bootstrap)
        .env("REKHA_SNAPSHOT_EVERY_N_LOGS", SNAPSHOT_EVERY)
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

fn dirs(count: usize) -> Vec<PathBuf> {
    (0..count)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            // Leak the guard: the directory has to outlive the child processes.
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_shard_owner_restarts_after_its_log_is_purged() {
    let _serial = SERIAL.lock().await;
    let dirs = dirs(NODES);
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();
    let peers = peers_spec(&ports);

    let mut nodes: Vec<Node> = (0..NODES)
        .map(|i| spawn(&dirs[i], ports[i], (i + 1) as u64, &peers, i == 0))
        .collect();
    wait_ready(&nodes).await;

    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "durable", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let collection: uuid::Uuid =
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

    let shipment = shard_of(&nodes, collection)
        .await
        .expect("the cluster reports the shard");
    let owners = placement_of(&nodes).owners(shipment).to_vec();
    assert_eq!(owners.len(), 3, "R=3 with four nodes");

    // Enough single-record batches to cross the snapshot and purge thresholds
    // in the owning shard group. The collection is created through node 1, so
    // writes are routed to an owner and the log they land in is the one that
    // gets compacted.
    for i in 0..BATCHES {
        let (status, body) = post_json_retrying(
            &nodes[0],
            &format!("/api/v2/collections/{collection}/add"),
            json!({"ids": [format!("r{i}")], "embeddings": [vec![i as f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "write {i}: {body}");
    }

    // Restart an owner that was fully up to date. A node that merely lagged
    // would need a snapshot install, which is a separate piece of work; this
    // test is about a node whose own log was compacted underneath it.
    let victim = owners[0];
    let index = nodes
        .iter()
        .position(|n| n.id == victim)
        .expect("victim is one of the nodes");
    let _ = nodes[index].child.kill();
    let _ = nodes[index].child.wait();
    let dir = dirs[index].clone();
    let port = ports[index];

    // Restart without `--bootstrap`: the cluster already exists, and a node that
    // re-initialised would be inventing a voter set.
    let restarted = spawn(&dir, port, victim, &peers, false);
    nodes[index] = restarted;

    wait_ready(&nodes).await;

    // The node must still serve its own shard's data, which is the observable
    // consequence of having reconstructed membership and placement.
    let (status, body) = post_json_retrying(
        &nodes[index],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count after restart: {body}");
    let count: u64 = serde_json::from_str::<serde_json::Value>(&body).expect("json")["count"]
        .as_u64()
        .expect("count");
    assert_eq!(count, BATCHES as u64, "every record survived the restart");

    // And it must still be able to commit. Readiness alone would not prove the
    // group can elect and append after compaction.
    let (status, body) = post_json_retrying(
        &nodes[index],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ["after-restart"], "embeddings": vec![vec![1.0f32; DIM]]}),
    )
    .await;
    assert_eq!(status, 200, "write after restart: {body}");

    // A second restart, now that another purge cycle has run on top of the
    // first, is where an empty snapshot scope would show up as data loss.
    for i in 0..BATCHES {
        let (status, body) = post_json_retrying(
            &nodes[0],
            &format!("/api/v2/collections/{collection}/add"),
            json!({"ids": [format!("s{i}")], "embeddings": [vec![2.0f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "second round write {i}: {body}");
    }
    let mut restarted_again = nodes.remove(index);
    let _ = restarted_again.child.kill();
    let _ = restarted_again.child.wait();
    let again = spawn(&dir, port, victim, &peers, false);
    nodes.insert(index, again);

    wait_ready(&nodes).await;
    let (status, body) = post_json_retrying(
        &nodes[index],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count after second restart: {body}");
    let count: u64 = serde_json::from_str::<serde_json::Value>(&body).expect("json")["count"]
        .as_u64()
        .expect("count");
    assert_eq!(
        count,
        (BATCHES * 2 + 1) as u64,
        "no records were lost across two purge cycles and two restarts"
    );
}

/// Snapshotting must stay off unless it is asked for, whichever group is in
/// question. This is the invariant the `REKHA_SNAPSHOT_EVERY_N_LOGS` default
/// exists to hold: a deployment that has not enabled compaction cannot purge.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_default_configuration_does_not_compact() {
    let _serial = SERIAL.lock().await;
    let dirs = dirs(3);
    let ports: Vec<u16> = (0..3).map(|_| free_port()).collect();
    let peers = peers_spec(&ports);

    let nodes: Vec<Node> = (0..3)
        .map(|i| {
            let child = serve_command(&dirs[i], ports[i], (i + 1) as u64, &peers, i == 0)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn rekha");
            Node {
                id: (i + 1) as u64,
                port: ports[i],
                data_dir: dirs[i].clone(),
                child,
            }
        })
        .collect();
    wait_ready(&nodes).await;

    for i in 0..40 {
        let (status, body) = post_json_retrying(
            &nodes[0],
            "/api/v2/collections",
            json!({"name": format!("c{i}"), "dimension": DIM}),
        )
        .await;
        assert_eq!(status, 200, "create c{i}: {body}");
    }

    // Every node's metadata log still holds its history: no snapshot means no
    // purge, which is what makes a restart safe while the flag is unset.
    // Inspected after the processes are gone: the redb files are locked while a
    // node holds them.
    let mut nodes = nodes;
    for node in nodes.iter_mut() {
        let _ = node.child.kill();
        let _ = node.child.wait();
    }
    for node in &nodes {
        let mut store = rekha_cluster::RedbLogStore::open(node.data_dir.join("raft_log.redb"))
            .expect("open raft log");
        let (last_purged, _committed) = store.watermarks().await;
        assert_eq!(
            last_purged, None,
            "node {} purged its log with snapshotting disabled",
            node.id
        );
    }
}
