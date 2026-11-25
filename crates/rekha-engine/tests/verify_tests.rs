//! `rekha verify` must fail on a disk that the node would refuse to open.
//!
//! The command exists for one moment: an operator is about to take a backup,
//! wipe a node, or restore over a directory, and wants to know whether what is
//! on disk is what the metadata says it is. A `clean` result that misses a
//! corrupt vector segment is worse than no command at all — it is an explicit
//! all-clear for data that cannot be loaded.
//!
//! This is a real gap that was tested for and fixed: segments carry their own
//! header checksum and were skipped by the manifest pass, because the manifest
//! intentionally records no CRC for them. Nothing validated that checksum outside
//! the collection loader.

use std::sync::Arc;

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::types::{Distance, Embedding};
use rekha_engine::verify::{Severity, verify_data_dir};
use rekha_engine::{Durability, Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog, Storage};

const DIM: usize = 8;

fn open_engine(dir: &tempfile::TempDir) -> Engine {
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).expect("catalog"));
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            durability: Durability::Durable,
            epoch: Epoch(0),
            ..EngineConfig::default()
        },
    )
    .expect("engine")
}

fn embedding(seed: f32) -> Embedding {
    (0..DIM)
        .map(|i| seed + i as f32)
        .collect::<Vec<f32>>()
        .into()
}

/// A collection with a published generation containing one segment.
fn seeded(dir: &tempfile::TempDir) -> (Engine, uuid::Uuid) {
    let engine = open_engine(dir);
    let mut config = CollectionConfig::new("docs".into(), DIM, Distance::L2);
    config.id = uuid::Uuid::new_v4();
    let record = engine.create_collection(&config).expect("create");

    let ids: Vec<String> = (0..4).map(|i| format!("r{i}")).collect();
    let embs: Vec<Embedding> = (0..4).map(|i| embedding(i as f32)).collect();
    engine
        .add(&record.config.id, &ids, &embs, None, None)
        .expect("add");
    engine.checkpoint(&record.config.id).expect("checkpoint");
    (engine, record.config.id)
}

fn segment_path(dir: &std::path::Path, collection: uuid::Uuid) -> std::path::PathBuf {
    let root = dir.join("wal").join(format!("{collection}.checkpoint"));
    let generation = std::fs::read_to_string(root.join("CURRENT")).expect("CURRENT");
    let segments = root
        .join("generations")
        .join(generation.trim())
        .join("segments");
    let mut files: Vec<_> = std::fs::read_dir(&segments)
        .expect("segments dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    files.sort();
    files.into_iter().next().expect("a segment file")
}

#[test]
fn a_healthy_directory_is_clean() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (_engine, collection) = seeded(&dir);

    let report = verify_data_dir(dir.path()).expect("verify");
    assert!(
        report.is_clean(),
        "a freshly checkpointed directory must verify: {:?}",
        report.findings
    );
    assert!(
        segment_path(dir.path(), collection).exists(),
        "the fixture must actually produce a segment"
    );
}

#[test]
fn a_corrupt_segment_is_an_error_not_a_clean_report() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Drop the engine first: the collection holds an mmap of the segment, and
    // editing a mapped file underneath it is not what an operator's disk looks
    // like anyway.
    let (engine, collection) = seeded(&dir);
    drop(engine);

    let segment = segment_path(dir.path(), collection);
    let mut bytes = std::fs::read(&segment).expect("read segment");
    // Flip a byte well past the header: the header checksum covers the vector
    // data, so this is corruption the loader detects and a length check cannot.
    let index = bytes.len() - 1;
    bytes[index] ^= 0xff;
    std::fs::write(&segment, &bytes).expect("write segment");

    let report = verify_data_dir(dir.path()).expect("verify");
    assert!(
        report.count(Severity::Error) > 0,
        "a corrupted segment must be reported as an error: {:?}",
        report.findings
    );
    assert!(
        report.findings.iter().any(|f| f.detail.contains("seg-")),
        "the finding must name the segment: {:?}",
        report.findings
    );
}

#[test]
fn a_missing_segment_is_reported() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (engine, collection) = seeded(&dir);
    drop(engine);

    std::fs::remove_file(segment_path(dir.path(), collection)).expect("remove segment");
    let report = verify_data_dir(dir.path()).expect("verify");
    assert!(
        report.count(Severity::Error) > 0,
        "a deleted segment must be an error: {:?}",
        report.findings
    );
}

#[test]
fn partitioned_collection_verifies_cleanly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("partitioned".into(), DIM, Distance::L2);
    config.id = uuid::Uuid::new_v4();
    let shard = Some(rekha_core::cluster::ShardId(2));
    let record = engine.record_collection(&config, shard).expect("record");
    let ids: Vec<String> = (0..4).map(|i| format!("r{i}")).collect();
    let embs: Vec<Embedding> = (0..4).map(|i| embedding(i as f32)).collect();
    let ops = (0..4)
        .map(|i| rekha_core::op::Operation::Add {
            id: ids[i].clone(),
            embedding: embs[i].clone(),
            metadata: None,
            document: None,
        })
        .collect::<Vec<_>>();

    engine
        .append_ops_partition(&record.config.id, shard, &ops)
        .expect("append ops");
    engine
        .checkpoint_partition(&record.config.id, shard)
        .expect("checkpoint");

    let report = verify_data_dir(dir.path()).expect("verify");
    assert!(
        report.is_clean(),
        "a checkpointed shard partition must verify clean: {:?}",
        report.findings
    );
    assert_eq!(report.collections, 1);
}

#[test]
fn a_missing_data_directory_is_an_error_not_a_clean_report() {
    let dir = tempfile::tempdir().unwrap();
    let absent = dir.path().join("typo");
    let err = rekha_engine::verify::verify_data_dir(&absent).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);

    // A directory that exists but has nothing in it is a legitimate empty
    // install, and reports clean.
    let empty = tempfile::tempdir().unwrap();
    let report = rekha_engine::verify::verify_data_dir(empty.path()).unwrap();
    assert!(report.is_clean());
    assert_eq!(report.collections, 0);
}
