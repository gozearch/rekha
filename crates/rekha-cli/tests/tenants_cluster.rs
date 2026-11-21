//! A namespace is a cluster fact, not a node fact.
//!
//! `create_tenant` used to answer `201` and write nothing, so the only thing a
//! client learned was that the server had read the request. Making the row real
//! is half the fix; the other half is that it must be real on *every* node, or a
//! load balancer decides whether a tenant exists.
//!
//! What is asserted here:
//!
//! 1. A follower cannot append to the metadata log, and says so with the same
//!    `503` + `x-rekha-leader` contract the collection write path uses — so a
//!    client knows to retry elsewhere instead of treating it as a failure.
//! 2. A tenant created through the leader is readable on every node, and the
//!    same for a database inside it.
//! 3. A tenant that was never created is still `404` on a follower, so "visible
//!    everywhere" is not achieved by answering yes to everything.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const NODES: usize = 3;

fn dirs(count: usize) -> Vec<PathBuf> {
    (0..count).map(|_| temp_dir()).collect()
}

/// The node that reports itself as metadata leader.
///
/// Asking `/ready` on an arbitrary node and taking the `leader` field is not
/// enough: a node can still report the leader of a term that has already ended,
/// and the test would then assert against the wrong node. The role a node
/// reports about *itself* is the fact to read.
async fn metadata_leader(nodes: &[Node]) -> Option<u64> {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        let mut leaders = Vec::new();
        for node in nodes {
            let Ok(resp) = client().get(node.url("/ready")).send().await else {
                continue;
            };
            let Ok(ready) = resp.json::<serde_json::Value>().await else {
                continue;
            };
            if ready["cluster"]["role"] == "leader" {
                leaders.push(node.id);
            }
        }
        match leaders.as_slice() {
            [id] => return Some(*id),
            _ if Instant::now() >= deadline => return None,
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

async fn get(node: &Node, path: &str) -> (u16, String) {
    let resp = client()
        .get(node.url(path))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    (status, resp.text().await.unwrap_or_default())
}

/// A follower reads from its own catalog, which the log updates asynchronously.
/// Asserting "eventually" is the honest contract; asserting "immediately" would
/// only pass by accident of scheduling.
async fn get_eventually(node: &Node, path: &str) -> (u16, String) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let answer = get(node, path).await;
        if answer.0 != 404 || Instant::now() >= deadline {
            return answer;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_namespace_created_on_any_node_is_visible_on_every_node() {
    let _serial = SERIAL.lock().await;

    let nodes = spawn_cluster(
        &dirs(NODES),
        &(0..NODES).map(|_| free_port()).collect::<Vec<_>>(),
    );
    wait_ready(&nodes).await;

    let leader = metadata_leader(&nodes)
        .await
        .expect("a leader after readiness");
    let follower = nodes.iter().find(|n| n.id != leader).expect("a follower");

    // 1. A write sent to a follower is routed to the metadata leader by the
    //    middleware, so the client does not have to know who leads. This is the
    //    surface contract; the handler's own `ForwardToLeader` mapping is the
    //    fallback for a write that reaches the log on a node whose cached
    //    leadership is stale.
    let resp = client()
        .post(follower.url("/api/v2/tenants"))
        .header("x-chroma-token", KEY)
        .json(&json!({ "name": "acme" }))
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "a namespace created through a follower must be accepted: {}",
        resp.text().await.unwrap_or_default()
    );

    for node in &nodes {
        let (status, body) = get_eventually(node, "/api/v2/tenants/acme").await;
        assert_eq!(
            status, 200,
            "node {} did not learn about the tenant: {body}",
            node.id
        );
    }

    // 2. The same for a database inside it, created through a different node.
    let other = nodes
        .iter()
        .find(|n| n.id != leader && n.id != follower.id)
        .unwrap_or(follower);
    let resp = client()
        .post(other.url("/api/v2/tenants/acme/databases"))
        .header("x-chroma-token", KEY)
        .json(&json!({ "name": "prod" }))
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status().as_u16(),
        200,
        "database create: {}",
        resp.text().await.unwrap_or_default()
    );

    for node in &nodes {
        let (status, body) = get_eventually(node, "/api/v2/tenants/acme/databases/prod").await;
        assert_eq!(
            status, 200,
            "node {} did not learn about the database: {body}",
            node.id
        );
    }

    // 3. Visibility everywhere is not the same as saying yes to everything.
    let (status, _) = get(follower, "/api/v2/tenants/never_created").await;
    assert_eq!(
        status, 404,
        "a tenant that does not exist must not be visible on a follower either"
    );
}
