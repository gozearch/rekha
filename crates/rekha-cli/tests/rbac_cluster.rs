//! RBAC requests that cannot succeed must be refused, not committed.
//!
//! In cluster mode every administrative write is a Raft entry. A request naming
//! a user or role that does not exist used to commit, fail to apply on every
//! node, and — because the apply path treated any unexpected catalog error as
//! divergence — set the node-wide not-ready flag. One authenticated call took
//! the whole cluster out of rotation until someone restarted it.
//!
//! There are two properties here and both matter:
//!
//! 1. The client gets a 4xx instead of a 200 for work that was never done.
//! 2. Every node stays ready, and the cluster can still commit afterwards.

mod common;

use std::path::PathBuf;
use std::time::Duration;

use common::*;
use serde_json::json;

const NODES: usize = 3;

fn dirs(count: usize) -> Vec<PathBuf> {
    (0..count).map(|_| temp_dir()).collect()
}

async fn get_json(node: &Node, path: &str) -> (u16, serde_json::Value) {
    let resp = client()
        .get(node.url(path))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("request");
    let status = resp.status().as_u16();
    let body = resp.json().await.unwrap_or(serde_json::Value::Null);
    (status, body)
}

async fn ready(node: &Node) -> String {
    client()
        .get(node.url("/ready"))
        .send()
        .await
        .expect("request")
        .text()
        .await
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rejected_rbac_calls_leave_the_cluster_ready() {
    let _serial = SERIAL.lock().await;
    let nodes = spawn_cluster(
        &dirs(NODES),
        &(0..NODES).map(|_| free_port()).collect::<Vec<_>>(),
    );
    wait_ready(&nodes).await;

    // The bootstrap superadmin exists on every node; the key in `KEY` is its.
    let (status, users) = get_json(&nodes[0], "/api/v2/auth/users").await;
    assert_eq!(status, 200, "users: {users}");
    let admin_id = users
        .as_array()
        .expect("user list")
        .iter()
        .find(|u| u["name"] == "admin")
        .and_then(|u| u["id"].as_str())
        .expect("the bootstrap user")
        .to_owned();

    // 1. A key for a user that does not exist.
    let (status, body) = post_json(
        &nodes[0],
        "/api/v2/auth/users/does-not-exist/keys",
        json!({"name": "orphan"}),
    )
    .await;
    assert_eq!(status, 404, "key for a missing user: {body}");

    // 2. A duplicate user name. The catalog refuses it, so committing the
    //    entry would return 200 and a key that never authenticates.
    let (status, body) = post_json(&nodes[0], "/api/v2/auth/users", json!({"name": "admin"})).await;
    assert_eq!(status, 409, "duplicate user name: {body}");

    // 3. A role assignment for a role that does not exist.
    let (status, body) = post_json(
        &nodes[0],
        &format!("/api/v2/auth/users/{admin_id}/roles"),
        json!({"role_id": "no-such-role"}),
    )
    .await;
    assert_eq!(status, 404, "assignment to a missing role: {body}");

    // 4. Deleting a built-in role.
    let (status, roles) = get_json(&nodes[0], "/api/v2/auth/roles").await;
    assert_eq!(status, 200, "roles: {roles}");
    let builtin = roles
        .as_array()
        .expect("role list")
        .iter()
        .find(|r| r["built_in"] == true)
        .and_then(|r| r["id"].as_str())
        .expect("a built-in role")
        .to_owned();
    let resp = client()
        .delete(nodes[0].url(&format!("/api/v2/auth/roles/{builtin}")))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status().as_u16(),
        409,
        "a built-in role must not be deletable"
    );

    // Give any wrongly-committed entry time to land before judging readiness.
    tokio::time::sleep(Duration::from_millis(500)).await;
    for node in &nodes {
        let body = ready(node).await;
        assert!(
            body.contains("\"ready\":true"),
            "node {} is not ready after rejected RBAC calls: {body}",
            node.id
        );
    }

    // And the cluster must still commit: a rejected call must not wedge it.
    let (status, body) =
        post_json(&nodes[0], "/api/v2/auth/users", json!({"name": "operator"})).await;
    assert_eq!(status, 200, "a valid create after the rejections: {body}");
}
