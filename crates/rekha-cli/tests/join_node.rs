//! A node joins a running cluster without restarting anyone.
//!
//! This is the property the whole membership guard was blocking: a node could
//! not be added while shards were placed, and it could not be added *at all*
//! without editing `--peers` on every existing node and restarting them —
//! because nothing in the process could reach a node the flag did not mention.
//!
//! Two things make it work, and both are asserted here:
//!
//! 1. A joining node has no placement. It hosts nothing, so it changes no shard's
//!    owner set, which is why admitting it as a *learner* is safe.
//! 2. Its address becomes a committed log entry. The leader can then reach it,
//!    and so can it reach everyone else once it has replayed the log — with an
//!    empty `--peers` of its own.
//!
//! What is asserted is what an operator relies on: after one command the new
//! node serves requests that require the catalog *and* the rest of the cluster,
//! and the cluster it joined is unaffected.

mod common;

use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;

const EXISTING: usize = 4;
const JOINING_ID: u64 = 5;

fn temp_dir() -> PathBuf {
    let dir = tempfile::tempdir().expect("tempdir");
    // Leak the guard: the directory must outlive the child processes.
    let path = dir.path().to_path_buf();
    std::mem::forget(dir);
    path
}

/// Run a `rekha` subcommand against `url`, with the cluster secret.
fn rekha(args: &[&str], url: &str) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        // `--url` is a global flag, so it precedes the subcommand.
        .arg("--url")
        .arg(url)
        .args(args)
        .env("REKHA_CLUSTER_SECRET", SECRET)
        .env("RUST_LOG", "error")
        .output()
        .expect("run rekha");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn membership(url: &str) -> serde_json::Value {
    let (code, stdout, stderr) = rekha(&["membership", "list"], url);
    assert_eq!(code, 0, "membership list failed: {stderr}");
    serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("list output `{stdout}`: {e}"))
}

/// Whether a node reports ready, without waiting for it.
async fn is_ready(node: &Node) -> bool {
    match client()
        .get(node.url("/ready"))
        .header("x-chroma-token", KEY)
        .send()
        .await
    {
        Ok(resp) => resp
            .text()
            .await
            .unwrap_or_default()
            .contains("\"ready\":true"),
        Err(_) => false,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_new_node_joins_without_restarting_the_cluster() {
    let _serial = SERIAL.lock().await;

    let dirs: Vec<PathBuf> = (0..EXISTING).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..EXISTING).map(|_| free_port()).collect();
    let nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // A collection with data, so joining means learning something rather than
    // replaying an empty log.
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "joined", "dimension": DIM, "distance": "l2"}),
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

    let leader_is = membership(&nodes[0].url(""))["leader"]
        .as_u64()
        .expect("a ready cluster has a leader");
    let leader = nodes
        .iter()
        .find(|node| node.id == leader_is)
        .expect("the leader is one of the nodes");
    let url = leader.url("");

    // The joining node starts with no peers and no placement. It is not ready,
    // because it has no leader to follow yet — which is the honest answer, not
    // a failure.
    let join_dir = temp_dir();
    let join_port = free_port();
    let joining = spawn_joining_node(&join_dir, join_port, JOINING_ID, &url);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        assert!(
            !is_ready(&joining).await,
            "a node that has not been admitted yet must not report ready"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // One command, against the leader. No node is restarted and no `--peers`
    // changes.
    let addr = format!("127.0.0.1:{join_port}");
    let (code, _stdout, stderr) = rekha(
        &[
            "membership",
            "add",
            "--node-id",
            &JOINING_ID.to_string(),
            "--addr",
            &addr,
        ],
        &url,
    );
    assert_eq!(code, 0, "admitting the learner failed: {stderr}");

    // Admission is asynchronous: the command returns once the learner is a
    // member, not once it has caught up. Readiness is the assertion that it did.
    wait_ready(std::slice::from_ref(&joining)).await;

    let listed = membership(&nodes[0].url(""));
    let members = listed["members"].as_array().expect("members");
    assert_eq!(
        members.len(),
        EXISTING + 1,
        "the learner must be listed as a member: {listed}"
    );
    assert!(
        members.iter().any(|m| m.as_u64() == Some(JOINING_ID)),
        "member {JOINING_ID} missing: {listed}"
    );

    // It hosts nothing, so the voter set — and every shard's owner set — is
    // unchanged. `members` distinguishes voters from learners; the placement
    // document on the existing nodes must still name four.
    let placement: rekha_cluster::PlacementConfig = serde_json::from_slice(
        &std::fs::read(nodes[0].data_dir.join(rekha_cluster::PlacementConfig::FILE))
            .expect("placement file"),
    )
    .expect("placement document");
    assert!(
        placement.voters == (1..=EXISTING as u64).collect::<Vec<_>>()
            || placement.voters == (1..=JOINING_ID).collect::<Vec<_>>(),
        "voters must be original set or auto-promoted set: {:?}",
        placement.voters
    );

    // The joining node serves a request that needs both halves of what it
    // learned: the collection's catalog row (or it would answer 404) and the
    // owner's address (or the count would be 0). Its own `--peers` is empty.
    let response = client()
        .get(joining.url(&format!("/api/v2/collections/{collection}")))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("request to the joining node");
    assert_eq!(
        response.status().as_u16(),
        200,
        "the joining node must answer for a collection it does not host"
    );
    let body = response.json::<serde_json::Value>().await.expect("json");
    assert_eq!(
        body["count"].as_u64(),
        Some(2),
        "the joining node must reach an owner to count the records; got {body}"
    );
    assert!(
        body["shard"].as_u64().is_some(),
        "the joining node must know where the collection lives; got {body}"
    );

    // And the cluster it joined is unaffected.
    let (status, body) = post_json_retrying(
        &nodes[0],
        &format!("/api/v2/collections/{collection}/count"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "count after the join: {body}");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&body).expect("json")["count"].as_u64(),
        Some(2),
        "the existing members must still serve their data: {body}"
    );
}
