//! A replica that fell behind the purged log prefix rejoins by snapshot.
//!
//! This is the case the other restart tests deliberately avoid. There, a node is
//! *up to date* and merely restarts: its applied state is in durable storage, so
//! resuming from the purge watermark is enough. Here the node is **stopped**
//! while the rest of the cluster keeps writing and compacts past the point it
//! reached, so the entries it needs no longer exist anywhere. Replay cannot
//! repair it; openraft asks the leader for a snapshot, and if that snapshot does
//! not exist the replication stream stops and the replica never returns.
//!
//! The assertion is not only "the node answers". It is that the data is there,
//! that the snapshot file exists on the receiver (proving the install path ran
//! and not a log replay), and that the group can still commit afterwards.

mod common;

use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use common::*;
use serde_json::json;

const NODES: usize = 3;
/// Small enough that one round of writes crosses the purge point several times.
const SNAPSHOT_EVERY: &str = "10";
/// Every write is one entry in the owning shard's log. With three nodes and
/// R=3, every node owns every shard, so the stopped node is a replica of the
/// collection's group.
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
    (0..count).map(|_| temp_dir()).collect()
}

/// True when `dir` holds at least one persisted snapshot.
///
/// Snapshots live under the engine's storage directory, which is `wal/` in a
/// data directory (the WAL, checkpoints and snapshots all belong to one store).
fn has_snapshot(dir: &std::path::Path) -> bool {
    let root = dir.join("wal").join("snapshots");
    let Ok(groups) = std::fs::read_dir(&root) else {
        return false;
    };
    for group in groups.flatten() {
        if !group.path().is_dir() {
            continue;
        }
        if let Ok(files) = std::fs::read_dir(group.path()) {
            for file in files.flatten() {
                if file.path().extension().is_some_and(|ext| ext == "snap") {
                    return true;
                }
            }
        }
    }
    false
}

async fn count(node: &Node, collection: uuid::Uuid) -> Option<u64> {
    // `consistency=local` on purpose: this reads *this replica's* applied state.
    // Reads default to the group leader now, and a count answered by the leader
    // would say nothing about whether the restarted node has caught up, which is
    // the entire point of this test.
    let (status, body) = post_json_retrying(
        node,
        &format!("/api/v2/collections/{collection}/count?consistency=local"),
        json!({}),
    )
    .await;
    if status != 200 {
        return None;
    }
    serde_json::from_str::<serde_json::Value>(&body)
        .ok()?
        .get("count")?
        .as_u64()
}

/// Poll until `node` reports `expected` records.
///
/// A write returns after it commits, but a replica applies asynchronously, so
/// the first read after the last write can legitimately be one behind. Polling
/// is the honest way to wait: a fixed sleep would pass slower machines and hide
/// a real lag.
async fn wait_for_count(node: &Node, collection: uuid::Uuid, expected: u64) -> Option<u64> {
    for _ in 0..150 {
        if let Some(value) = count(node, collection).await
            && value >= expected
        {
            return Some(value);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
    count(node, collection).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_replica_that_fell_behind_the_purge_point_rejoins_by_snapshot() {
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
        json!({"name": "streamed", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let collection: uuid::Uuid =
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

    // A few writes so the stopped node has a real applied position, then stop it.
    for i in 0..5 {
        let (status, body) = post_json_retrying(
            &nodes[0],
            &format!("/api/v2/collections/{collection}/add"),
            json!({"ids": [format!("before{i}")], "embeddings": [vec![i as f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "pre-stop write {i}: {body}");
    }

    // Stop the last node: with R=3 the other two keep quorum, and the stopped
    // node is the one that will be behind the purge point.
    let victim = NODES - 1;
    let _ = nodes[victim].child.kill();
    let _ = nodes[victim].child.wait();

    // Write past the snapshot threshold several times over. The leader snapshots
    // and purges, so the entries the stopped node needs are gone from every log.
    for i in 0..BATCHES {
        let (status, body) = post_json_retrying(
            &nodes[0],
            &format!("/api/v2/collections/{collection}/add"),
            json!({"ids": [format!("during{i}")], "embeddings": [vec![2.0f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "write {i} while one replica is down: {body}");
    }

    let expected = (5 + BATCHES) as u64;

    // The survivor must be healthy and hold everything: this is the baseline the
    // rejoining node has to match.
    let live = wait_for_count(&nodes[0], collection, expected)
        .await
        .expect("count on node 1");
    assert_eq!(live, expected, "the surviving replica lost records");

    // Restart without `--bootstrap`: the cluster already exists.
    let port = ports[victim];
    let dir = dirs[victim].clone();
    let id = nodes[victim].id;
    nodes[victim] = spawn(&dir, port, id, &peers, false);

    wait_ready(&nodes).await;

    // Readiness does not mean caught up, so poll the count on the restarted node
    // itself. A stale or missing replica answers 0 or fails here.
    let seen = wait_for_count(&nodes[victim], collection, expected).await;
    assert_eq!(
        seen,
        Some(expected),
        "the rejoining replica never reached {expected} records; snapshot transfer did not \
         repair it"
    );

    // Direct evidence that the install path ran: the receiver persisted the
    // snapshot it was given. A node that caught up by replaying the log would
    // have no snapshot of its own here, because it has not built one yet.
    assert!(
        has_snapshot(&dir),
        "the restarted node holds no snapshot, so it caught up by replay rather than by \
         install — this test would not be exercising the transfer path"
    );

    // And the group must still commit with the returned replica.
    let (status, body) = post_json_retrying(
        &nodes[victim],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ["after-rejoin"], "embeddings": vec![vec![3.0f32; DIM]]}),
    )
    .await;
    assert_eq!(status, 200, "write on the rejoined replica: {body}");

    // The write is committed, not necessarily applied on this replica yet, so
    // poll rather than reading once: under load the apply lags the ack by a few
    // milliseconds and a fixed read would be a flaky assertion.
    let final_count = wait_for_count(&nodes[victim], collection, expected + 1)
        .await
        .expect("final count");
    assert_eq!(final_count, expected + 1, "the write landed everywhere");
}
