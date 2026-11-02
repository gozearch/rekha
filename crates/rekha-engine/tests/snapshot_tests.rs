//! Snapshot and restore.
//!
//! The property under test is the one the previous implementation violated: a
//! snapshot must contain **vectors**, not just ids and metadata. A restored node
//! that answers a query with the wrong distance is worse than one that refuses
//! to start, because it looks healthy.

#![allow(clippy::unreadable_literal)]

use std::sync::Arc;

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::types::{Distance, Embedding, Metadata, MetadataValue};
use rekha_engine::{Durability, Engine, EngineConfig, QueryOptions};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog, Storage};
use tempfile::TempDir;
use uuid::Uuid;

const DIM: usize = 8;

/// Every collection this engine holds, which is the scope a whole-corpus
/// snapshot uses. Production callers pass a shard's collection set instead;
/// these tests are single-group, so "everything local" is the same thing.
fn scope(engine: &Engine) -> Vec<Uuid> {
    engine
        .list_collections("default_tenant", "default_database")
        .expect("list collections")
        .into_iter()
        .map(|r| r.config.id)
        .collect()
}

fn open_engine(dir: &TempDir) -> Engine {
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
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
    .expect("open engine")
}

fn vector(seed: u8) -> Embedding {
    (0..DIM)
        .map(|i| {
            let value = (seed as usize * 17 + i * 3) % 29;
            value as f32 / 29.0
        })
        .collect::<Vec<f32>>()
        .into()
}

fn metadata(tag: &str) -> Metadata {
    let mut map = Metadata::new();
    map.insert("tag".to_string(), MetadataValue::Str(tag.to_string()));
    map
}

fn query_options() -> QueryOptions {
    QueryOptions {
        ef: 0,
        where_filter: None,
        where_document: None,
        oversampling: 4,
    }
}

/// Seed a collection with records whose vectors are already flushed into a
/// segment — that is the case the old snapshot silently dropped.
fn seed(engine: &Engine, cid: Uuid, count: usize) {
    let mut config = CollectionConfig::new("docs".into(), DIM, Distance::L2);
    config.id = cid;
    // A small batch size forces the write buffer into the index early, so the
    // records' embeddings stop living in `Record.embedding`.
    config.hnsw.batch_size = 2;
    engine.create_collection(&config).unwrap();

    for i in 0..count {
        let id = format!("id-{i:03}");
        let emb = vector(i as u8);
        let meta = Some(metadata(&format!("m{i}")));
        let document = Some(format!("document {i}"));
        engine
            .add(&cid, &[id], &[emb], Some(&[meta]), Some(&[document]))
            .unwrap();
    }
}

#[test]
fn snapshot_roundtrip_preserves_vectors_and_distances() {
    let source_dir = TempDir::new().unwrap();
    let source = open_engine(&source_dir);
    let cid = Uuid::new_v4();
    seed(&source, cid, 12);

    // Flush everything into the index so no vector is left in the write buffer.
    source.optimize(&cid).unwrap();

    let archive = source
        .snapshot_bytes(b"cluster-state", &scope(&source))
        .expect("snapshot must succeed once a generation exists");
    assert!(archive.starts_with(b"RKSN"));

    let target_dir = TempDir::new().unwrap();
    let target = open_engine(&target_dir);
    let (state, report) = target.restore_bytes(&archive).expect("restore");
    assert_eq!(state, b"cluster-state");
    assert_eq!(report.collections, 1);
    assert!(report.files > 0);

    // The decisive assertion: identical results, including distances. The old
    // snapshot restored ids and metadata but no embeddings, so every distance
    // here would have been computed from a missing vector.
    let probe = vector(5);
    let before = source.query(&cid, &probe, 5, &query_options()).unwrap();
    let after = target.query(&cid, &probe, 5, &query_options()).unwrap();

    assert_eq!(before.len(), after.len());
    assert_eq!(before.len(), 5);
    for (a, b) in before.iter().zip(after.iter()) {
        assert_eq!(a.id, b.id, "ids diverged after restore");
        assert_eq!(
            a.distance, b.distance,
            "distance for {} diverged after restore: {} vs {}",
            a.id, a.distance, b.distance
        );
    }
}

#[test]
fn snapshot_preserves_metadata_documents_and_config() {
    let source_dir = TempDir::new().unwrap();
    let source = open_engine(&source_dir);
    let cid = Uuid::new_v4();
    seed(&source, cid, 4);
    source.optimize(&cid).unwrap();

    let archive = source.snapshot_bytes(&[], &scope(&source)).unwrap();

    let target_dir = TempDir::new().unwrap();
    let target = open_engine(&target_dir);
    target.restore_bytes(&archive).unwrap();

    let records = target.get_records(&cid, None, None, None, None, 0).unwrap();
    assert_eq!(records.len(), 4);
    for record in &records {
        assert!(record.document.is_some(), "document lost for {}", record.id);
        assert!(record.metadata.is_some(), "metadata lost for {}", record.id);
    }

    let restored = target
        .get_collection_by_id(&cid)
        .unwrap()
        .expect("collection must exist");
    assert_eq!(restored.config.dimension, DIM);
    assert_eq!(restored.config.space, Distance::L2);
    assert_eq!(restored.config.name, "docs");
}

/// A snapshot is a copy of the generation, so a collection that has never been
/// checkpointed cannot be snapshotted from memory alone. This fails loudly
/// rather than shipping a records-only archive.
#[test]
fn snapshot_refuses_when_a_generation_is_missing() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);

    let mut config = CollectionConfig::new("fresh".into(), DIM, Distance::L2);
    config.hnsw.sync_threshold = 1_000_000; // never auto-checkpoints
    engine.create_collection(&config).unwrap();
    engine
        .add(&config.id, &["a".to_string()], &[vector(1)], None, None)
        .unwrap();

    let err = engine
        .snapshot_bytes(&[], &scope(&engine))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no published checkpoint generation"), "{err}");
    assert!(err.contains("vectors live in segments"), "{err}");
}

/// `checkpoint_all` is the step that makes a snapshot safe to pair with a log
/// purge; after it, the snapshot succeeds.
#[test]
fn checkpoint_all_makes_a_snapshot_possible() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);

    let mut config = CollectionConfig::new("fresh".into(), DIM, Distance::L2);
    config.hnsw.sync_threshold = 1_000_000;
    engine.create_collection(&config).unwrap();
    engine
        .add(&config.id, &["a".to_string()], &[vector(1)], None, None)
        .unwrap();

    assert_eq!(engine.checkpoint_all(&scope(&engine)).unwrap(), 1);
    let archive = engine.snapshot_bytes(&[], &scope(&engine)).unwrap();
    assert!(archive.starts_with(b"RKSN"));
}

#[test]
fn a_corrupt_archive_is_rejected_and_leaves_the_node_untouched() {
    let source_dir = TempDir::new().unwrap();
    let source = open_engine(&source_dir);
    let cid = Uuid::new_v4();
    seed(&source, cid, 3);
    source.optimize(&cid).unwrap();
    let archive = source.snapshot_bytes(&[], &scope(&source)).unwrap();

    let target_dir = TempDir::new().unwrap();
    let target = open_engine(&target_dir);
    let other = Uuid::new_v4();
    seed(&target, other, 2);

    let mut corrupt = archive.clone();
    let index = corrupt.len() - 8;
    corrupt[index] ^= 0xff;

    let err = target.restore_bytes(&corrupt).unwrap_err().to_string();
    assert!(
        err.contains("checksum mismatch") || err.contains("truncated"),
        "{err}"
    );

    // The pre-existing collection is still there and still queryable.
    assert_eq!(target.count(&other).unwrap(), 2);
    assert!(
        target.get_collection_by_id(&cid).unwrap().is_none(),
        "a rejected restore must not create anything"
    );
}

/// The archive's own CRC is not the only check: the reconstructed generation is
/// re-validated by the checkpoint loader. A generation whose manifest is wrong
/// must be rejected even though its framing is intact.
#[test]
fn a_generation_that_fails_checkpoint_validation_is_rejected() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);

    // Build a generation directory by hand with a manifest this build does not
    // understand.
    let gen_dir = dir
        .path()
        .join("handmade")
        .join("generations")
        .join("gen-00000000000000000001-deadbeef");
    std::fs::create_dir_all(&gen_dir).unwrap();
    std::fs::write(gen_dir.join("checkpoint.json"), br#"{"format_version":99}"#).unwrap();

    let record = rekha_storage::CollectionRecord {
        config: CollectionConfig::new("handmade".into(), DIM, Distance::L2),
        max_seq_id: 0,
        total_elements: 0,
        shard: None,
    };
    let archive = rekha_engine::encode_snapshot_archive_for_test(&[], vec![(record, gen_dir)])
        .expect("framing must succeed; only the contents are invalid");

    let err = engine.restore_bytes(&archive).unwrap_err().to_string();
    assert!(
        err.contains("failed validation"),
        "expected checkpoint validation to reject it, got: {err}"
    );
}

#[test]
fn restoring_twice_converges() {
    let source_dir = TempDir::new().unwrap();
    let source = open_engine(&source_dir);
    let cid = Uuid::new_v4();
    seed(&source, cid, 6);
    source.optimize(&cid).unwrap();
    let archive = source.snapshot_bytes(&[], &scope(&source)).unwrap();

    let target_dir = TempDir::new().unwrap();
    let target = open_engine(&target_dir);
    target.restore_bytes(&archive).unwrap();
    target.restore_bytes(&archive).unwrap();

    assert_eq!(target.count(&cid).unwrap(), 6);
    let probe = vector(2);
    let before = source.query(&cid, &probe, 3, &query_options()).unwrap();
    let after = target.query(&cid, &probe, 3, &query_options()).unwrap();
    assert_eq!(
        before.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
        after.iter().map(|r| r.id.clone()).collect::<Vec<_>>()
    );
}

#[test]
fn restoring_replaces_a_divergent_collection() {
    let source_dir = TempDir::new().unwrap();
    let source = open_engine(&source_dir);
    let cid = Uuid::new_v4();
    seed(&source, cid, 5);
    source.optimize(&cid).unwrap();
    let archive = source.snapshot_bytes(&[], &scope(&source)).unwrap();

    let target_dir = TempDir::new().unwrap();
    let target = open_engine(&target_dir);
    // Same collection id, different contents: this is the "behind" replica.
    let mut config = CollectionConfig::new("docs".into(), DIM, Distance::L2);
    config.id = cid;
    target.create_collection(&config).unwrap();
    target
        .add(&cid, &["stale".to_string()], &[vector(9)], None, None)
        .unwrap();
    target.optimize(&cid).unwrap();
    assert_eq!(target.count(&cid).unwrap(), 1);

    target.restore_bytes(&archive).unwrap();

    assert_eq!(target.count(&cid).unwrap(), 5);
    assert!(
        !target.get(&cid, &["stale".to_string()]).unwrap()[0].is_some(),
        "the stale record must be gone after a restore"
    );
}
