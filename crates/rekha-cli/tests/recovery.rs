//! `rekha verify` against real damaged directories.
//!
//! Each case starts from a real node's data directory, copies it, damages the
//! copy in one specific way, and asserts that verification names that damage.
//! The point is not that the checker returns `false` somewhere — it is that it
//! distinguishes a recoverable torn tail from a hole, and a missing file from a
//! file whose bytes changed.
//!
//! The directory is also hashed before and after, because a checker that
//! repairs while it inspects is the worst of both.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use serde_json::json;

const RECORDS: usize = 3;

/// A single node's data directory, populated so it has both a checkpoint
/// generation and a WAL tail past that generation's replay boundary.
struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Fixture {
    fn wal_dir(&self) -> PathBuf {
        self.root.join("wal")
    }

    fn collection(&self) -> uuid::Uuid {
        let mut found = None;
        for entry in std::fs::read_dir(self.wal_dir()).expect("wal dir") {
            let path = entry.expect("entry").path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && let Some(id) = name.strip_suffix(".wal")
                && let Ok(id) = id.parse()
            {
                found = Some(id);
            }
        }
        found.expect("a write-ahead log")
    }
}

async fn build_fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().join("data");
    std::fs::create_dir_all(&root).expect("data dir");
    let port = free_port();

    let child = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("serve")
        .arg("--data-dir")
        .arg(&root)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("REKHA_API_KEY", KEY)
        .env("RUST_LOG", "error")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn rekha");

    let mut node = Node {
        id: 1,
        port,
        data_dir: root.clone(),
        child,
    };
    wait_ready(std::slice::from_ref(&node)).await;

    let (status, body) = post_json(
        &node,
        "/api/v2/collections",
        json!({"name": "verify-me", "dimension": DIM, "distance": "l2"}),
    )
    .await;
    assert_eq!(status, 200, "create: {body}");
    let id: uuid::Uuid = serde_json::from_str::<serde_json::Value>(&body).expect("json")["id"]
        .as_str()
        .expect("id")
        .parse()
        .expect("uuid");

    let embeddings: Vec<Vec<f32>> = (0..RECORDS).map(|i| vec![i as f32 + 0.5; DIM]).collect();
    let ids: Vec<String> = (0..RECORDS).map(|i| format!("r{i}")).collect();
    let (status, body) = post_json(
        &node,
        &format!("/api/v2/collections/{id}/add"),
        json!({"ids": ids, "embeddings": embeddings}),
    )
    .await;
    assert_eq!(status, 200, "add: {body}");

    // Force a checkpoint, so the directory has a published generation.
    let (status, body) = post_json(
        &node,
        &format!("/api/v2/collections/{id}/optimize"),
        json!({}),
    )
    .await;
    assert_eq!(status, 200, "optimize: {body}");

    // And a WAL tail past the generation's boundary, which is what a restart
    // replays. Two separate batches, so damage in the first one has verified
    // data after it — which is what distinguishes a hole from a torn tail.
    for i in 0..2 {
        let (status, body) = post_json(
            &node,
            &format!("/api/v2/collections/{id}/add"),
            json!({"ids": [format!("tail{i}")], "embeddings": [vec![9.0f32; DIM]]}),
        )
        .await;
        assert_eq!(status, 200, "add after checkpoint: {body}");
    }

    // SIGTERM, not SIGKILL: this is a fixture, not a crash test.
    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(node.child.id().to_string())
        .status();
    let _ = node.child.wait();
    drop(node);

    Fixture { _dir: dir, root }
}

/// Run `rekha verify` and return `(exit code, parsed JSON)`.
fn verify(dir: &Path) -> (i32, serde_json::Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("verify")
        .arg("--data-dir")
        .arg(dir)
        .arg("--json")
        .output()
        .expect("run verify");
    let body = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(body.trim()).unwrap_or_else(|e| panic!("verify output `{body}`: {e}"));
    (output.status.code().unwrap_or(-1), parsed)
}

/// Classes named anywhere in the report.
fn classes(report: &serde_json::Value) -> Vec<String> {
    report["findings"]
        .as_array()
        .map(|findings| {
            findings
                .iter()
                .filter_map(|f| f["class"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

/// Copy a data directory, so damage is never applied to the fixture.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("dest");
    for entry in std::fs::read_dir(from).expect("read dir") {
        let entry = entry.expect("entry");
        let target = to.join(entry.file_name());
        if entry.file_type().expect("type").is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).expect("copy file");
        }
    }
}

/// Every path and its bytes, so "did it write anything" is answerable.
fn fingerprint(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).expect("read dir") {
            let entry = entry.expect("entry");
            let path = entry.path();
            if entry.file_type().expect("type").is_dir() {
                walk(&path, base, out);
            } else {
                let bytes = std::fs::read(&path).expect("read file");
                out.insert(
                    path.strip_prefix(base).expect("relative").to_path_buf(),
                    bytes,
                );
            }
        }
    }
    walk(dir, dir, &mut out);
    out
}

/// The generation directory a collection's `CURRENT` names.
fn generation(dir: &Path, collection: uuid::Uuid) -> PathBuf {
    let base = dir.join("wal").join(format!("{collection}.checkpoint"));
    let name = std::fs::read_to_string(base.join("CURRENT")).expect("CURRENT");
    base.join("generations").join(name.trim())
}

#[tokio::test]
async fn a_healthy_directory_is_clean_and_untouched() {
    let fixture = build_fixture().await;
    let before = fingerprint(&fixture.root);

    let (code, report) = verify(&fixture.root);
    assert_eq!(code, 0, "a healthy directory must verify: {report}");
    assert_eq!(report["clean"], true, "{report}");
    assert_eq!(report["errors"], 0, "{report}");
    assert!(
        classes(&report).is_empty(),
        "nothing to report, got {:?}",
        classes(&report)
    );
    assert!(
        report["collections"].as_u64().unwrap_or(0) >= 1,
        "the collection must be seen: {report}"
    );

    assert_eq!(
        fingerprint(&fixture.root),
        before,
        "verify must not modify the directory it inspects"
    );
}

#[tokio::test]
async fn every_corruption_class_is_named() {
    let fixture = build_fixture().await;
    let collection = fixture.collection();
    let work = tempfile::tempdir().expect("workdir");

    // (label, expected class, damage)
    type Damage = fn(&Path, uuid::Uuid);
    let cases: Vec<(&str, &str, Damage)> = vec![
        ("stale CURRENT", "stale_current", |dir, id| {
            let current = dir
                .join("wal")
                .join(format!("{id}.checkpoint"))
                .join("CURRENT");
            std::fs::write(&current, "gen-00000000000000000999-deadbeef\n").expect("write CURRENT");
        }),
        ("missing manifest", "bad_manifest", |dir, id| {
            let generation_dir = generation(dir, id);
            std::fs::remove_file(generation_dir.join("checkpoint.json")).expect("remove manifest");
        }),
        ("missing file", "missing_file", |dir, id| {
            let generation_dir = generation(dir, id);
            std::fs::remove_file(generation_dir.join("records.bin")).expect("remove records");
        }),
        ("wrong length", "wrong_length", |dir, id| {
            let generation_dir = generation(dir, id);
            let path = generation_dir.join("records.bin");
            let bytes = std::fs::read(&path).expect("read");
            std::fs::write(&path, &bytes[..bytes.len() - 1]).expect("truncate");
        }),
        ("checksum mismatch", "checksum_mismatch", |dir, id| {
            let generation_dir = generation(dir, id);
            let path = generation_dir.join("records.bin");
            let mut bytes = std::fs::read(&path).expect("read");
            // Flip a byte in the middle: the length still matches, so only the
            // checksum can catch it.
            let middle = bytes.len() / 2;
            bytes[middle] ^= 0xff;
            std::fs::write(&path, &bytes).expect("write");
        }),
        ("wal corrupt", "wal_corrupt", |dir, id| {
            let path = dir.join("wal").join(format!("{id}.wal"));
            let mut bytes = std::fs::read(&path).expect("read wal");
            // Damage the *first* batch's body, leaving its length intact and a
            // second verified batch after it. Recovery can only discard a tail,
            // so this one is unrecoverable.
            const HEADER_LEN: usize = 18;
            const BATCH_HEAD: usize = 8; // length + crc
            bytes[HEADER_LEN + BATCH_HEAD + 4] ^= 0xff;
            std::fs::write(&path, &bytes).expect("write wal");
        }),
        ("wal truncated", "wal_truncated", |dir, id| {
            let path = dir.join("wal").join(format!("{id}.wal"));
            let bytes = std::fs::read(&path).expect("read wal");
            std::fs::write(&path, &bytes[..bytes.len() - 3]).expect("truncate wal");
        }),
    ];

    for (label, expected, damage) in cases {
        let copy = work.path().join(label.replace(' ', "-"));
        copy_dir(&fixture.root, &copy);
        damage(&copy, collection);

        let (code, report) = verify(&copy);
        let found = classes(&report);

        // A torn tail is recoverable: it is reported but does not fail the
        // check. Everything else must.
        let recoverable = expected == "wal_truncated";
        assert!(
            found.iter().any(|c| c == expected),
            "{label}: expected `{expected}`, got {found:?} in {report}"
        );
        assert_eq!(
            report["clean"],
            json!(recoverable),
            "{label}: clean should be {recoverable}: {report}"
        );
        assert_eq!(
            code,
            i32::from(!recoverable),
            "{label}: exit code must follow severity: {report}"
        );
    }
}

/// The WAL must still hold every batch after the checkpoint's boundary.
///
/// Constructed rather than damaged: the log is replaced with one whose first
/// record is far past the checkpoint, which is what pruning too far or losing a
/// segment would look like.
#[tokio::test]
async fn a_wal_that_starts_after_the_checkpoint_is_a_hole() {
    let fixture = build_fixture().await;
    let collection = fixture.collection();

    let base = fixture
        .root
        .join("wal")
        .join(format!("{collection}.checkpoint"));
    let name = std::fs::read_to_string(base.join("CURRENT")).expect("CURRENT");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            base.join("generations")
                .join(name.trim())
                .join("checkpoint.json"),
        )
        .expect("manifest"),
    )
    .expect("json");
    let snapshot_seq = manifest["snapshot_seq"].as_u64().expect("snapshot_seq");

    let wal_dir = fixture.root.join("wal");
    std::fs::remove_file(wal_dir.join(format!("{collection}.wal"))).expect("remove wal");
    let mut options = rekha_wal::WalOptions::new(true);
    // Resume far past the checkpoint's boundary: the records in between exist
    // nowhere, which is a hole rather than a torn tail.
    options.start_seq = Some(snapshot_seq + 10);
    let mut wal = rekha_wal::Wal::open(
        &wal_dir,
        &collection,
        rekha_core::cluster::Epoch(0),
        options,
        rekha_wal::TailRepair::Refuse,
    )
    .expect("open wal");
    wal.append(rekha_core::op::Operation::Delete { id: "tail".into() })
        .expect("append");
    drop(wal);

    let (code, report) = verify(&fixture.root);
    let found = classes(&report);
    assert!(
        found.iter().any(|c| c == "wal_hole"),
        "expected wal_hole, got {found:?} in {report}"
    );
    assert_eq!(code, 1, "a hole loses records: {report}");
}

/// The property restore exists for, end to end: a copy of the data comes back on
/// a fresh directory, verifies clean, and serves the records that were written
/// before the backup.
#[tokio::test]
async fn a_restored_directory_serves_the_data_that_was_backed_up() {
    let fixture = build_fixture().await;
    let work = tempfile::tempdir().expect("workdir");
    let backup = work.path().join("backup");
    let restored = work.path().join("restored");

    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("backup")
        .arg("--data-dir")
        .arg(&fixture.root)
        .arg("--out")
        .arg(&backup)
        .output()
        .expect("run backup");
    assert!(
        output.status.success(),
        "backup failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("restore")
        .arg("--data-dir")
        .arg(&restored)
        .arg("--from")
        .arg(&backup)
        .output()
        .expect("run restore");
    assert!(
        output.status.success(),
        "restore failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // A restored directory must be indistinguishable from a healthy one to the
    // checker before anything tries to open it.
    let (code, report) = verify(&restored);
    assert_eq!(code, 0, "restored directory must verify: {report}");
    assert_eq!(report["clean"], true, "{report}");

    // And it must serve. This is the half that a checksum comparison cannot
    // prove: the node opens the restored files and answers from them.
    let port = free_port();
    let child = Command::new(env!("CARGO_BIN_EXE_rekha"))
        .arg("serve")
        .arg("--data-dir")
        .arg(&restored)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .env("REKHA_API_KEY", KEY)
        .env("RUST_LOG", "error")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn restored node");
    let mut node = Node {
        id: 1,
        port,
        data_dir: restored.clone(),
        child,
    };
    wait_ready(std::slice::from_ref(&node)).await;

    // Every record written before the backup, including the one that was only
    // in the WAL at the time.
    let ids: Vec<String> = (0..RECORDS)
        .map(|i| format!("r{i}"))
        .chain(std::iter::once("tail0".to_owned()))
        .chain(std::iter::once("tail1".to_owned()))
        .collect();
    // Ask for the records by id, through the public API.
    let collection = first_collection(&node).await;
    let (status, body) = post_json(
        &node,
        &format!("/api/v2/collections/{collection}/get"),
        json!({ "ids": ids }),
    )
    .await;
    assert_eq!(status, 200, "get after restore: {body}");
    let parsed: serde_json::Value = serde_json::from_str(&body).expect("json");
    let returned = parsed["ids"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or_else(|| parsed["ids"].as_object().map(|o| o.len()).unwrap_or(0));
    assert_eq!(
        returned,
        RECORDS + 2,
        "every record written before the backup must come back: {body}"
    );

    let _ = Command::new("kill")
        .arg("-TERM")
        .arg(node.child.id().to_string())
        .status();
    let _ = node.child.wait();
}

/// The id of the first collection on the node.
async fn first_collection(node: &Node) -> uuid::Uuid {
    let response = client()
        .get(node.url("/api/v2/collections"))
        .header("x-chroma-token", KEY)
        .send()
        .await
        .expect("list");
    let body: serde_json::Value = response.json().await.expect("json");
    body.as_array()
        .and_then(|a| a.first())
        .and_then(|c| c["id"].as_str())
        .expect("a collection")
        .parse()
        .expect("uuid")
}
