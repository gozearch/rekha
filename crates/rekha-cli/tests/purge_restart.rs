//! Restart after the Raft log has been purged.
//!
//! openraft purges the log prefix once a snapshot covers it. Restarting a node
//! whose log has been purged used to panic inside `RaftState::new`, because the
//! store persisted the purge watermark but not the commit point it is checked
//! against:
//!
//! ```text
//! expect: self.purge_upto()(Some(LogId { term: 1, node_id: 0, index: 3999 }))
//!     <= self.committed()(None)
//! ```
//!
//! The node then stayed up reporting `ready:false` with `term:0` and answered
//! every write with 503, forever.
//!
//! The snapshot threshold is overridden to a small number so this exercises the
//! purge path in seconds instead of after thousands of collections.

mod common;

use std::process::Stdio;
use std::time::Duration;

use common::*;
use rekha_cluster::RedbLogStore;
use serde_json::json;

/// How many metadata entries to append before restarting. With a threshold of
/// 10 this crosses several snapshot boundaries, so the log is definitely
/// purged rather than merely large.
const COLLECTIONS: usize = 30;
const SNAPSHOT_EVERY: &str = "10";

fn spawn(dir: &std::path::Path, port: u16) -> Node {
    let child = serve_command(dir, port, 1, "", true)
        .env("REKHA_SNAPSHOT_EVERY_N_LOGS", SNAPSHOT_EVERY)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    Node {
        id: 1,
        port,
        data_dir: dir.to_path_buf(),
        child,
    }
}

#[tokio::test]
async fn node_restarts_after_its_raft_log_is_purged() {
    let _guard = SERIAL.lock().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("data");
    std::fs::create_dir_all(&root).expect("data dir");
    let port = free_port();

    let node = spawn(&root, port);
    wait_ready(std::slice::from_ref(&node)).await;

    for i in 0..COLLECTIONS {
        let (status, body) = post_json(
            &node,
            "/api/v2/collections",
            json!({"name": format!("c{i}"), "dimension": DIM}),
        )
        .await;
        assert_eq!(status, 200, "create c{i}: {body}");
    }

    // SIGKILL: no graceful shutdown, which is what a crashed or OOM-killed node
    // looks like. A clean shutdown could flush state this test is meant to
    // prove can be rebuilt from disk.
    let mut node = node;
    let _ = node.child.kill();
    let _ = node.child.wait();
    drop(node);

    // The test is only meaningful if a snapshot (and therefore a purge) ran.
    // Assert it on the store rather than inferring it from the number of
    // collections, which depends on openraft's exact threshold arithmetic.
    // Inspected after the process is gone: the redb file is locked while the
    // node holds it.
    let mut store = RedbLogStore::open(root.join("raft_log.redb")).expect("open raft log");
    let (last_purged, committed) = store.watermarks().await;
    assert!(
        last_purged.is_some(),
        "no purge happened, so this test would not cover the restart bug \
         (committed={committed:?})"
    );
    assert!(
        committed.is_some_and(|c| last_purged.is_some_and(|p| p <= c)),
        "the store persisted a purge watermark without a commit point: \
         last_purged={last_purged:?} committed={committed:?}"
    );
    drop(store);

    let node = spawn(&root, port);
    wait_ready_timeout(std::slice::from_ref(&node), Duration::from_secs(60)).await;

    // Ready is not enough: the node must still be able to commit.
    let (status, body) = post_json(
        &node,
        "/api/v2/collections",
        json!({"name": "after-restart", "dimension": DIM}),
    )
    .await;
    assert_eq!(status, 200, "create after restart: {body}");
}

/// Snapshotting is off unless asked for. Purging is only safe once the state
/// machine is persisted, so the default must not purge: disk growth is
/// observable, a node that cannot restart is not.
#[tokio::test]
async fn snapshots_are_disabled_by_default() {
    let _guard = SERIAL.lock().await;
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("data");
    std::fs::create_dir_all(&root).expect("data dir");
    let port = free_port();

    let child = serve_command(&root, port, 1, "", true)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn rekha");
    let node = Node {
        id: 1,
        port,
        data_dir: root.clone(),
        child,
    };
    wait_ready(std::slice::from_ref(&node)).await;

    for i in 0..COLLECTIONS {
        let (status, body) = post_json(
            &node,
            "/api/v2/collections",
            json!({"name": format!("d{i}"), "dimension": DIM}),
        )
        .await;
        assert_eq!(status, 200, "create d{i}: {body}");
    }

    let mut node = node;
    let _ = node.child.kill();
    let _ = node.child.wait();
    drop(node);

    let mut store = RedbLogStore::open(root.join("raft_log.redb")).expect("open raft log");
    let (last_purged, _committed) = store.watermarks().await;
    assert_eq!(
        last_purged, None,
        "the default configuration purged the log; restore the state machine \
         persistence before enabling snapshots by default"
    );
}
