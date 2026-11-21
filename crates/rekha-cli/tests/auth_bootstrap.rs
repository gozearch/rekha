//! The superadmin is one user in the cluster, not one per node.
//!
//! `bootstrap_superadmin_if_empty` ran on every node against its own catalog,
//! before Raft existed. With `REKHA_API_KEY` set, every node minted its own
//! `admin` row with the same key, so `GET /auth/users` returned a different set
//! on every node and a node joining later ended up with two admins. Without the
//! key, each node generated a *different* random key and printed it, and that
//! key authenticated only against the node that printed it.
//!
//! What is asserted:
//!
//! 1. A clustered node refuses to start without `REKHA_API_KEY`, because that is
//!    the only configuration that cannot diverge — and it says why.
//! 2. Every node reports the same user list, and the environment key
//!    authenticates against all of them.

mod common;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::*;

const NODES: usize = 3;

fn dirs(count: usize) -> Vec<PathBuf> {
    (0..count).map(|_| temp_dir()).collect()
}

async fn users(node: &Node) -> Option<Vec<serde_json::Value>> {
    let resp = client()
        .get(node.url("/api/v2/auth/users"))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .ok()?;
    if resp.status().as_u16() != 200 {
        return None;
    }
    resp.json::<Vec<serde_json::Value>>().await.ok()
}

/// The user list every node agrees on, once there is one.
async fn agreed_users(nodes: &[Node]) -> Option<Vec<serde_json::Value>> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let mut lists = Vec::new();
        let mut all_served = true;
        for node in nodes {
            match users(node).await {
                Some(list) => lists.push(list),
                None => {
                    all_served = false;
                    lists.push(Vec::new());
                }
            }
        }
        let settled = all_served
            && lists.iter().all(|list| !list.is_empty())
            && lists.iter().all(|list| *list == lists[0]);
        if settled {
            return lists.into_iter().next();
        }
        if Instant::now() >= deadline {
            for (node, list) in nodes.iter().zip(&lists) {
                eprintln!("node {} reports {} users: {list:?}", node.id, list.len());
            }
            return None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[test]
fn a_clustered_node_refuses_to_start_without_an_api_key() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let output = serve_command(dir.path(), ports[0], 1, &peers_spec(&ports), true)
        // The one configuration that must not run: each node would mint its own
        // superadmin, and the key printed by one would work only on that one.
        .env_remove("REKHA_API_KEY")
        .output()
        .expect("run rekha");

    assert!(
        !output.status.success(),
        "a clustered node started without REKHA_API_KEY; its superadmin would exist on one \
         node only"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("REKHA_API_KEY"),
        "the refusal must name the missing variable: {stderr}"
    );
}

/// The bootstrap key is stored as a fast hash, so a short value is a wordlist
/// entry recoverable from a stolen catalog. `openssl rand -hex 16` is the
/// documented way to produce one.
#[test]
fn a_short_bootstrap_key_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let output = serve_command(dir.path(), ports[0], 1, &peers_spec(&ports), true)
        .env("REKHA_API_KEY", "smoke-key")
        .output()
        .expect("run rekha");

    assert!(
        !output.status.success(),
        "a node started with a 9-byte bootstrap key"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("at least 32"),
        "the refusal must say what is required: {stderr}"
    );
}

/// The cluster secret authenticates every internal Raft and forwarding call, so
/// a short value is a guessable cluster-wide credential.
#[test]
fn a_clustered_node_refuses_a_short_cluster_secret() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let output = serve_command(dir.path(), ports[0], 1, &peers_spec(&ports), true)
        .env("REKHA_CLUSTER_SECRET", "too-short")
        .output()
        .expect("run rekha");

    assert!(
        !output.status.success(),
        "a clustered node started with a short REKHA_CLUSTER_SECRET"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("at least 32"),
        "the refusal must say what is required: {stderr}"
    );
}

/// The public and internal listeners cannot share a port: the second bind
/// fails after Raft has started, leaving a half-initialized directory behind.
#[test]
fn a_colliding_internal_port_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let port = free_port();

    let output = serve_command(dir.path(), port, 1, "", false)
        .arg("--internal-port")
        .arg(port.to_string())
        .output()
        .expect("run rekha");

    assert!(
        !output.status.success(),
        "a node started with the internal listener on its public port"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--internal-port"),
        "the refusal must name the flag: {stderr}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_node_reports_the_same_superadmin() {
    let _serial = SERIAL.lock().await;

    let nodes = spawn_cluster(
        &dirs(NODES),
        &(0..NODES).map(|_| free_port()).collect::<Vec<_>>(),
    );
    wait_ready(&nodes).await;

    let users = agreed_users(&nodes)
        .await
        .expect("nodes never agreed on a non-empty user list");

    assert_eq!(
        users.len(),
        1,
        "the cluster has exactly one bootstrap user; more means a node minted its own: {users:?}"
    );
    assert_eq!(users[0]["name"], "admin");

    // The environment key is the one the bootstrap hashed, so it must still be
    // accepted now that a user exists and the legacy path no longer applies.
    for node in &nodes {
        let resp = client()
            .get(node.url("/api/v2/auth/users"))
            .header("x-chroma-token", KEY)
            .send()
            .await
            .expect("request");
        assert_eq!(
            resp.status().as_u16(),
            200,
            "the bootstrap key must authenticate against node {}",
            node.id
        );
    }
}
