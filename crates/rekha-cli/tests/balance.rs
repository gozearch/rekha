//! Balance, against ten real processes.
//!
//! Measured before this gate existed: at 200 collections on 10 nodes with 40
//! shards, one node held **46%** of the corpus and another **8%** (ideal 30%).
//! The cause was structural, not a bad hash — collections hashed to shards by
//! id, and shard ownership was itself skewed 5..17 shards per node, so
//! spreading collections evenly across shards still piled them onto whichever
//! node owned many shards. Raising the shard count did not help, because groups
//! per node is `shards x R / nodes`.
//!
//! What is asserted here is the property an operator cares about: after the
//! leader places collections by the load it can see, no node holds materially
//! more of the corpus than any other. The bound is two collections on top of the
//! ideal — a figure taken from the placement function's own measurement
//! (`rekha_cluster::Placement::place` gives a spread of 2 across 30 simulated
//! trials), not a number chosen to make this pass.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const NODES: usize = 10;
const COLLECTIONS: usize = 200;

/// Collections per node, counted from the data files.
///
/// Every collection is one `.wal` on each of its three replicas, so this is the
/// number of collections a node actually stores — not an estimate from the
/// placement function, which would be checking the function against itself.
fn collections_per_node(nodes: &[Node]) -> Vec<(u64, usize)> {
    nodes
        .iter()
        .map(|node| {
            let count = std::fs::read_dir(node.data_dir.join("wal"))
                .map(|entries| {
                    entries
                        .filter_map(Result::ok)
                        .filter(|entry| entry.path().extension().is_some_and(|e| e == "wal"))
                        .count()
                })
                .unwrap_or(0);
            (node.id, count)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn collections_are_spread_across_nodes() {
    let _serial = SERIAL.lock().await;

    let dirs: Vec<PathBuf> = (0..NODES).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let nodes = spawn_cluster(&dirs, &ports);
    // Ten processes need more than 60 s to elect on a 2-core CI runner.
    wait_ready_timeout(&nodes, Duration::from_secs(180)).await;

    let started = Instant::now();
    for i in 0..COLLECTIONS {
        let (status, body) = post_json_retrying(
            &nodes[0],
            "/api/v2/collections",
            json!({"name": format!("balance-{i}"), "dimension": DIM, "distance": "l2"}),
        )
        .await;
        assert_eq!(status, 200, "creating collection {i}: {body}");
        let id: uuid::Uuid = serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

        // One record each: a collection's data file appears when its shard
        // group first learns of it, which is on the first write. Creating
        // without writing would leave nothing to count, and the measurement is
        // about stored collections, not registry rows.
        let (status, body) = post_json_retrying(
            &nodes[0],
            &format!("/api/v2/collections/{id}/add"),
            json!({"ids": ["r0"], "embeddings": [vec![0.5f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "writing to collection {i}: {body}");
    }
    let elapsed = started.elapsed();

    // Placing is a metadata write; the data files appear as each replica
    // materializes, which is asynchronous by design.
    let ideal = COLLECTIONS * 3 / NODES;
    let deadline = Instant::now() + Duration::from_secs(60);
    let per_node = loop {
        let counts = collections_per_node(&nodes);
        let total: usize = counts.iter().map(|(_, n)| n).sum();
        if total == COLLECTIONS * 3 {
            break counts;
        }
        assert!(
            Instant::now() < deadline,
            "only {total} of {} data files appeared after 60s: {counts:?}",
            COLLECTIONS * 3
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    };

    let min = per_node.iter().map(|(_, n)| *n).min().unwrap_or(0);
    let max = per_node.iter().map(|(_, n)| *n).max().unwrap_or(0);

    eprintln!(
        "placed {COLLECTIONS} collections in {elapsed:?}: per node {per_node:?} \
         (ideal {ideal}, spread {})",
        max - min
    );

    // Two collections of slack. The old id-derived placement produced a spread
    // of 69 on the same workload (21..90), and 92 vs 17 at this size in a real
    // run, so this is not a bound that a regression slips past.
    assert!(
        max - min <= 2,
        "placement is not balanced: min {min}, max {max} (ideal {ideal}); \
         per node {per_node:?}"
    );

    // And no node may hold the whole corpus: that is what sharding removes.
    assert!(
        max < COLLECTIONS,
        "a node holds all {COLLECTIONS} collections; placement is not distributing: {per_node:?}"
    );
}
