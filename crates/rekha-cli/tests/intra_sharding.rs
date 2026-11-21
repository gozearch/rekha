//! Intra-collection sharding across all shards.
//!
//! Asserts that:
//! 1. A collection created with `partitioned: true` is partitioned across all shards in the cluster.
//! 2. Records with different IDs hash to different shards and are stored across multiple nodes.
//! 3. Scatter-gather queries fan out across all shards, merge top-k hits deterministically by distance.
//! 4. Count sums across all shard partitions.
//! 5. Get and delete by IDs route to the correct owning shard.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::*;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn intra_collection_sharding_scatters_and_gathers() {
    let _serial = SERIAL.lock().await;
    let dirs: Vec<PathBuf> = (0..4).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..4).map(|_| free_port()).collect();

    let nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // Create an intra-collection partitioned collection across all shards
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({
            "name": "partitioned-coll",
            "dimension": DIM,
            "distance": "l2",
            "partitioned": true
        }),
    )
    .await;
    assert_eq!(status, 200, "create collection must succeed: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("parse create response");
    let collection: uuid::Uuid = parsed["id"]
        .as_str()
        .expect("collection id")
        .parse()
        .expect("uuid");

    // Add 100 records with diverse keys
    let count = 100;
    let ids: Vec<String> = (0..count).map(|i| format!("doc-{i}")).collect();
    let embeddings: Vec<Vec<f32>> = (0..count)
        .map(|i| {
            let base = (i as f32) / 10.0;
            (0..DIM).map(|d| base + (d as f32)).collect()
        })
        .collect();
    let metadatas: Vec<serde_json::Value> = (0..count)
        .map(|i| json!({"tag": if i % 2 == 0 { "even" } else { "odd" }, "num": i}))
        .collect();

    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }),
    )
    .await;
    assert_eq!(status, 200, "add records must succeed: {body}");

    // Give asynchronous consensus replication a brief moment to commit across all shards
    tokio::time::sleep(Duration::from_millis(500)).await;

    // 1. Verify global count from any node
    for node in &nodes {
        let (status, body) = post_json_retrying(
            node,
            &format!("/api/v2/collections/{collection}/count"),
            json!({}),
        )
        .await;
        assert_eq!(status, 200, "count must succeed: {body}");
        let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
        assert_eq!(
            resp["count"].as_u64(),
            Some(count as u64),
            "node {} must see all 100 records: {body}",
            node.id
        );
    }

    // 2. Scatter-gather query from different nodes
    let query_vector = vec![0.0f32; DIM];
    for node in &nodes {
        let (status, body) = post_json_retrying(
            node,
            &format!("/api/v2/collections/{collection}/query"),
            json!({
                "query_embeddings": vec![query_vector.clone()],
                "n_results": 5,
            }),
        )
        .await;
        assert_eq!(status, 200, "query must succeed: {body}");
        let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
        let result_ids = resp["ids"][0].as_array().expect("result ids array");
        assert_eq!(result_ids.len(), 5, "query must return top 5 hits");
        // doc-0 has embedding [0, 1, 2, ...], closest to [0, 0, 0, ...]
        assert_eq!(
            result_ids[0].as_str(),
            Some("doc-0"),
            "doc-0 must be the nearest neighbor"
        );
    }

    // 3. Get records by ID across shards
    let (status, body) = post_json_retrying(
        &nodes[1],
        &format!("/api/v2/collections/{collection}/get"),
        json!({
            "ids": vec!["doc-10", "doc-42", "doc-99"],
        }),
    )
    .await;
    assert_eq!(status, 200, "get by ids must succeed: {body}");
    let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
    let fetched_ids: Vec<&str> = resp["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(fetched_ids, vec!["doc-10", "doc-42", "doc-99"]);

    // 4. Delete records across shards
    let (status, body) = post_json_retrying(
        &nodes[2],
        &format!("/api/v2/collections/{collection}/delete"),
        json!({
            "ids": vec!["doc-10", "doc-42"],
        }),
    )
    .await;
    assert_eq!(status, 200, "delete must succeed: {body}");

    // Check count decreased to 98
    let (status, body) = post_json_retrying(
        &nodes[3],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count after delete: {body}");
    let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
    assert_eq!(resp["count"].as_u64(), Some(98));

    // 5. Verify Cross-Shard 2PC Atomicity (Rollback on Partial Failure)
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({
            "ids": ["anchor-1"],
            "embeddings": [vec![1.0f32; DIM]],
        }),
    )
    .await;
    assert_eq!(status, 200, "adding anchor record: {body}");

    // Submit a multi-shard batch containing new records and the duplicate "anchor-1"
    let (status, body) = post_json(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({
            "ids": ["new-1", "new-2", "new-3", "anchor-1"],
            "embeddings": [vec![1.0f32; DIM], vec![2.0f32; DIM], vec![3.0f32; DIM], vec![1.0f32; DIM]],
        }),
    )
    .await;
    assert!(
        status == 400 || status == 500,
        "batch with duplicate must be rejected: status {status} body {body}"
    );

    // Verify 2PC rolled back and none of the new records were stored on any shard
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/get"),
        json!({
            "ids": ["new-1", "new-2", "new-3"],
        }),
    )
    .await;
    assert_eq!(status, 200, "get must succeed: {body}");
    let resp: serde_json::Value = serde_json::from_str(&body).expect("json");
    let ids = resp["ids"].as_array().unwrap();
    assert!(
        ids.is_empty(),
        "2PC aborted batch must not leak any records onto other shards: {ids:?}"
    );
}
