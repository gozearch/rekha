//! Membership commands must reach the server and say the truth about it.
//!
//! Two failures this file exists for, both measured against a running cluster:
//!
//! 1. `rekha membership list|add|remove` posted to `/internal/raft/...` while
//!    the server serves `/internal/cluster/...`, so every one of them answered
//!    `404 Not Found`. The only operator entry point for changing membership
//!    did not exist.
//! 2. The commands sent no cluster secret, so on a correctly configured
//!    cluster the same calls would have answered `401` even with the right
//!    path. A command that only works on an unconfigured deployment is not a
//!    command.
//!
//! What is asserted is what an operator relies on: `list` reports the members,
//! and `add` reports *why* it is refused rather than "not found".

mod common;

use common::*;
use serde_json::json;
use std::path::PathBuf;

const NODES: usize = 4;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn membership_commands_reach_the_server_and_refuse_what_they_must() {
    let _serial = SERIAL.lock().await;

    let dirs: Vec<PathBuf> = (0..NODES).map(|_| temp_dir()).collect();
    let ports: Vec<u16> = (0..NODES).map(|_| free_port()).collect();

    let nodes = spawn_cluster(&dirs, &ports);
    wait_ready(&nodes).await;

    // A collection, so the cluster is in the state the refusal is about. The
    // refusal is a function of "shards are placed", and a cluster with no
    // collections still has shards, but asserting after real traffic is the
    // state an operator is actually in.
    let (status, body) = post_json_retrying(
        &nodes[0],
        "/api/v2/collections",
        json!({"name": "membership", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");

    let url = nodes[0].url("");

    let (code, stdout, stderr) = rekha(&["membership", "list"], &url);
    assert_eq!(code, 0, "membership list failed: {stderr}");
    let listed: serde_json::Value =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("list output `{stdout}`: {e}"));
    let members = listed["members"]
        .as_array()
        .unwrap_or_else(|| panic!("list must report the members: {stdout}"));
    assert_eq!(
        members.len(),
        NODES,
        "every node must be listed as a member: {stdout}"
    );

    // Removing a voting member is refused, and the reason is a sentence about
    // placement — not `404`, which is what an unreachable route looks like.
    //
    // Adding a *learner* is admitted instead: a learner does not vote, so no
    // shard's owner set changes. `join_node.rs` exercises that end to end.
    let (code, _stdout, stderr) = rekha(&["membership", "remove", "--node-id", "2"], &url);
    assert_eq!(code, 1, "a refused membership change must exit non-zero");
    assert!(
        stderr.contains("membership cannot change while shards are placed"),
        "the refusal must explain itself; got: {stderr}"
    );
    assert!(
        !stderr.contains("404"),
        "the route must exist, or the refusal is unreachable: {stderr}"
    );
}
