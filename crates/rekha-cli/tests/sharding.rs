//! Sharding, against real processes.
//!
//! The property under test cannot be proved in-process: it is that a node which
//! is not a replica of a collection has **no data for it on disk** and still
//! serves requests for it correctly by forwarding. Both halves are only visible
//! from outside the process — one is a file listing, the other is an HTTP
//! response from a node holding nothing.
//!
//! Four nodes with three replicas is the smallest cluster where placement is
//! meaningful. At three nodes and `R = 3` every node holds every shard, so a
//! suite that only ever runs three nodes would pass while storing the whole
//! corpus everywhere.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn data_lives_on_its_replicas_only_and_any_node_serves_it() {
    let _serial = SERIAL.lock().await;
    let dirs: Vec<PathBuf> = (0..4)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            // Leak the guard: the directory must outlive the child processes for
            // the whole test, and `TempDir`'s drop would remove it too early if
            // it were kept alive by a local.
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect();
    let ports: Vec<u16> = (0..4).map(|_| free_port()).collect();

    let nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // Create one collection through node 1.
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "placed", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create must succeed: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("create response");
    let collection: uuid::Uuid = parsed["id"]
        .as_str()
        .expect("collection id")
        .parse()
        .expect("uuid");

    // Placement is deterministic, so the test can name the owners rather than
    // asking the cluster and checking it against itself.
    let placement = placement_of(&nodes);
    assert_eq!(placement.voters().len(), 4);
    assert_eq!(
        placement.replication_factor(),
        3,
        "four nodes with R=3 is the smallest cluster where placement distributes"
    );
    let shard = shard_of(&nodes, collection)
        .await
        .expect("the cluster reports where it placed the collection");
    let owners = placement.owners(shard);
    let outsider = nodes
        .iter()
        .find(|n| !owners.contains(&n.id))
        .expect("with 3 replicas on 4 nodes exactly one node holds no copy");

    // Write through the node that holds nothing: it must forward, not fail.
    let ids: Vec<String> = (0..RECORDS).map(|i| format!("rec-{i}")).collect();
    let embeddings: Vec<Vec<f32>> = (0..RECORDS)
        .map(|i| (0..DIM).map(|d| (i as f32) + (d as f32) / 100.0).collect())
        .collect();
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ids, "embeddings": embeddings}),
    )
    .await;
    assert_eq!(status, 200, "add must be accepted: {body}");

    // The replicas must have the data, and the non-replica must not.
    let owners_dirs: Vec<PathBuf> = nodes
        .iter()
        .filter(|n| owners.contains(&n.id))
        .map(|n| n.data_dir.clone())
        .collect();
    eventually("every replica to materialize the collection", || {
        nodes
            .iter()
            .filter(|n| owners.contains(&n.id))
            .all(|n| has_data(n, collection))
    })
    .await;

    // Give the cluster a moment in which a wrong write would show up, then
    // assert the outsider still has nothing. Polling only for the positive case
    // would pass even if the data were copied everywhere.
    tokio::time::sleep(Duration::from_millis(1500)).await;
    for node in &nodes {
        let expected = owners.contains(&node.id);
        assert_eq!(
            has_data(node, collection),
            expected,
            "node {} {} hold collection {collection}; owners are {owners:?} \
             (dirs with data: {owners_dirs:?})",
            node.id,
            if expected { "must" } else { "must not" }
        );
    }

    // Every node must answer the same query, including the one with no data.
    let mut answers: Vec<Vec<String>> = Vec::new();
    for node in &nodes {
        let (status, body) = post_json_retrying(
            node,
            &format!("/api/v2/collections/{collection}/query"),
            json!({
                "query_embeddings": vec![vec![0.0f32; DIM]],
                "n_results": 5,
            }),
        )
        .await;
        assert_eq!(
            status, 200,
            "node {} must serve a query for a collection it may not hold: {body}",
            node.id
        );
        let parsed: serde_json::Value = serde_json::from_str(&body).expect("query response");
        let ids: Vec<String> = parsed["ids"][0]
            .as_array()
            .expect("ids")
            .iter()
            .map(|v| v.as_str().unwrap_or_default().to_owned())
            .collect();
        assert_eq!(ids.len(), 5, "node {} returned {ids:?}", node.id);
        answers.push(ids);
    }
    for (i, answer) in answers.iter().enumerate() {
        assert_eq!(
            answer, &answers[0],
            "node {} answered differently; a replica and a forwarder must agree",
            nodes[i].id
        );
    }

    // Count is a data question, so the non-replica must forward it too.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count: {body}");
    assert!(
        body.contains(&RECORDS.to_string()),
        "count must report {RECORDS} regardless of which node serves it: {body}"
    );

    assert!(
        !owners.contains(&outsider.id),
        "the node used to prove forwarding must not be a replica"
    );

    // The drift check this refactor exists for: the bytes on disk must be on
    // exactly the nodes that own the shard the catalog row names. If routing
    // and storage ever answer "where does this live" differently, this is where
    // it surfaces — before it surfaces as a read returning nothing.
    let mut holding: Vec<u64> = nodes
        .iter()
        .filter(|node| has_data(node, collection))
        .map(|node| node.id)
        .collect();
    holding.sort_unstable();
    let mut expected = owners.to_vec();
    expected.sort_unstable();
    assert_eq!(
        holding, expected,
        "the nodes holding the data must be exactly the owners of shard {shard}: \
         catalog says {expected:?}, disk says {holding:?}"
    );
}

/// Kill the replica that placement routing picks first.
///
/// Reproduced before the fix this test accompanies: a non-owner forwarded every
/// request to the first owner, that owner was dead, and the collection became
/// permanently unwritable from every node that did not hold it — while the
/// shard had a healthy leader and never lost quorum. Losing one replica is the
/// failure the cluster exists to survive, so it must not be the failure the
/// router creates.
///
/// The leader is moved by the kill, which also exercises the second half: a
/// request that lands on a replica which is no longer the leader must reach the
/// new one, instead of being refused because it was already forwarded once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_survives_the_loss_of_the_replica_routing_prefers() {
    let _serial = SERIAL.lock().await;
    let dirs: Vec<PathBuf> = (0..4)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect();
    let ports: Vec<u16> = (0..4).map(|_| free_port()).collect();

    let mut nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    let placement = placement_of(&nodes);

    // Find a collection whose shard does not include node 1, so node 1 is a
    // non-owner and the victim (its first owner) is somebody else.
    let mut chosen = None;
    for i in 0..40 {
        let (status, body) = post_json_retrying(
            &nodes[0],
            "/api/v2/collections",
            json!({"name": format!("loss-{i}"), "dimension": DIM, "distance": "l2"}),
        )
        .await;
        assert_eq!(status, 200, "create: {body}");
        let id: uuid::Uuid = serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");
        let owners = shard_of(&nodes, id)
            .await
            .map(|shard| placement.owners(shard))
            .unwrap_or_default();
        if !owners.contains(&nodes[0].id) && owners.len() == 3 {
            chosen = Some((id, owners));
            break;
        }
    }
    let (collection, owners) = chosen.expect("a collection placed away from node 1");
    let victim = owners[0];

    // The collection must exist in its shard group before the kill, otherwise
    // the test would be measuring creation rather than failover.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ["before"], "embeddings": vec![vec![0.5f32; DIM]]}),
    )
    .await;
    assert_eq!(status, 200, "add before the kill: {body}");

    // Kill the first owner. This is the node the old router forwarded to and
    // never fell back from.
    let index = nodes
        .iter()
        .position(|n| n.id == victim)
        .expect("victim is one of the nodes");
    let _ = nodes[index].child.kill();
    let _ = nodes[index].child.wait();

    // Wait for the shard to elect among its two surviving replicas.
    let shard = shard_of(&nodes, collection)
        .await
        .expect("a surviving node still reports the placement");
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut leader = None;
    while Instant::now() < deadline {
        if let Some(found) = shard_leader(&nodes, shard).await
            && found != victim
        {
            leader = Some(found);
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let leader = leader.unwrap_or_else(|| {
        panic!("shard {shard} never elected a leader other than the killed node {victim}")
    });

    // A non-owner must still be able to serve the collection.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ["after"], "embeddings": vec![vec![0.6f32; DIM]]}),
    )
    .await;
    assert_eq!(
        status, 200,
        "a non-owner must reach the shard after replica {victim} died          (leader is now {leader}, owners {owners:?}): {body}"
    );

    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count after the kill: {body}");
    assert!(
        body.contains('2'),
        "both records must be readable after one replica died: {body}"
    );

    // And a surviving replica must answer a query directly, not only via a
    // forward: the data is there, the routing just had to find it.
    let survivor = owners
        .iter()
        .find(|id| **id != victim)
        .copied()
        .expect("two survivors");
    let node = nodes
        .iter()
        .find(|n| n.id == survivor)
        .expect("survivor node");
    let (status, body) = post_json_retrying(
        node,
        &format!("/api/v2/collections/{collection}/query"),
        json!({"query_embeddings": vec![vec![0.5f32; DIM]], "n_results": 2}),
    )
    .await;
    assert_eq!(status, 200, "query on a surviving replica: {body}");
    assert!(
        body.contains("before") && body.contains("after"),
        "the surviving replica must still hold both records: {body}"
    );
}

/// The elected leader of `shard`, as reported by any node that hosts it.
async fn shard_leader(nodes: &[Node], shard: rekha_cluster::ShardId) -> Option<u64> {
    for node in nodes {
        // A killed node simply fails the request; there is no liveness flag to
        // consult, and there should not be one here.
        let Ok(response) = client()
            .get(node.url("/internal/cluster/membership"))
            .header("x-rekha-cluster-secret", SECRET)
            .send()
            .await
        else {
            continue;
        };
        let Ok(body) = response.json::<serde_json::Value>().await else {
            continue;
        };
        if let Some(shards) = body["shards"].as_array() {
            for entry in shards {
                if entry["shard"].as_u64() == Some(u64::from(shard.0))
                    && let Some(leader) = entry["leader"].as_u64()
                {
                    return Some(leader);
                }
            }
        }
    }
    None
}

/// A collection whose shard has this node as an owner must be readable after a
/// restart even though the shard group's log is the source of truth: the engine
/// re-opens the collections placement says it owns.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn placement_survives_a_restart_without_moving_data() {
    let _serial = SERIAL.lock().await;
    let dirs: Vec<PathBuf> = (0..4)
        .map(|_| {
            let dir = tempfile::tempdir().expect("tempdir");
            let path = dir.path().to_path_buf();
            std::mem::forget(dir);
            path
        })
        .collect();
    let ports: Vec<u16> = (0..4).map(|_| free_port()).collect();

    let mut nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "restart", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let collection: uuid::Uuid =
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
            .as_str()
            .expect("id")
            .parse()
            .expect("uuid");

    let placement = placement_of(&nodes);
    let owners = placement.owners(
        shard_of(&nodes, collection)
            .await
            .expect("the cluster reports where it placed the collection"),
    );

    let ids: Vec<String> = vec!["a".into(), "b".into()];
    let embeddings: Vec<Vec<f32>> = vec![vec![0.1f32; DIM], vec![0.9f32; DIM]];
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/add"),
        json!({"ids": ids, "embeddings": embeddings}),
    )
    .await;
    assert_eq!(status, 200, "add: {body}");

    eventually("replicas to write their copies", || {
        nodes
            .iter()
            .filter(|n| owners.contains(&n.id))
            .all(|n| has_data(n, collection))
    })
    .await;

    // Restart every node. Placement must come back from `placement.json`, not
    // be recomputed from a possibly different default.
    for node in &mut nodes {
        let _ = node.child.kill();
        let _ = node.child.wait();
    }
    let restarted = spawn_cluster(&dirs, &ports);
    wait_ready(&restarted).await;

    let after = placement_of(&restarted);
    assert_eq!(
        after.shard_count(),
        placement.shard_count(),
        "shard count must be read back, not recomputed"
    );

    // Data in cluster mode is durable in the shard group's log, not in the
    // engine's WAL, and a restarted node has to replay that log once its group
    // elects. So the assertion is that the data *comes back on its own*, with
    // no operator action — not that it is visible in the first millisecond.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let (status, body) = post_json_retrying(
            &restarted[0],
            &format!("/api/v2/collections/{collection}/count"),
            json!({}),
        )
        .await;
        assert_eq!(status, 200, "count after restart: {body}");
        if body.contains('2') {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "both records must survive the restart without operator action: {body}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    for node in &restarted {
        assert_eq!(
            has_data(node, collection),
            owners.contains(&node.id),
            "restart must not move data onto node {}",
            node.id
        );
    }
}
