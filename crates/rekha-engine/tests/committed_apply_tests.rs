//! The cluster apply path: operations that are already durable in the Raft log.
//!
//! These tests pin the two properties the cluster depends on — applying costs
//! no second fsync, and applying twice is the same as applying once — plus the
//! guard that refuses to take the shortcut in single-node mode, where nothing
//! else would have made the write durable.

use std::sync::Arc;

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::op::Operation;
use rekha_core::types::{Distance, Embedding};
use rekha_engine::{ApplyMode, Durability, Engine, EngineConfig};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog, Storage};
use tempfile::TempDir;
use uuid::Uuid;

const DIM: usize = 4;

fn open_engine(dir: &TempDir, apply_mode: ApplyMode) -> Engine {
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            node_id: 0,
            scope: rekha_engine::EngineScope::All,
            durability: Durability::Durable,
            epoch: Epoch(0),
            apply_mode,
        },
    )
    .expect("open engine")
}

fn vector(seed: f32) -> Embedding {
    (0..DIM)
        .map(|i| seed + i as f32 / 10.0)
        .collect::<Vec<f32>>()
        .into()
}

fn add(id: &str, seed: f32) -> Operation {
    Operation::Add {
        id: id.to_string(),
        embedding: vector(seed),
        metadata: None,
        document: Some(format!("doc {id}")),
    }
}

fn create(engine: &Engine, cid: Uuid, batch_size: usize) {
    let mut config = CollectionConfig::new("docs".into(), DIM, Distance::L2);
    config.id = cid;
    config.hnsw.batch_size = batch_size;
    config.hnsw.sync_threshold = 1_000_000; // never auto-checkpoint in these tests
    engine.create_collection(&config).unwrap();
}

/// The guard that stops a future refactor from silently making single-node
/// writes non-durable.
#[test]
fn apply_committed_refuses_in_durable_mode() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir, ApplyMode::Durable);
    let cid = Uuid::new_v4();
    create(&engine, cid, 100);

    let err = engine
        .apply_committed(&cid, &[add("a", 1.0)], 1, DIM)
        .unwrap_err()
        .to_string();

    assert!(err.contains("ApplyMode::Committed"), "{err}");
    assert_eq!(engine.count(&cid).unwrap(), 0, "nothing may be applied");
}

/// The whole point of the cluster apply path: the entry is already durable, so
/// the engine must not append to its WAL.
#[test]
fn apply_committed_applies_without_touching_the_wal() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir, ApplyMode::Committed);
    let cid = Uuid::new_v4();
    create(&engine, cid, 100);

    engine
        .apply_committed(&cid, &[add("a", 1.0), add("b", 2.0)], 7, DIM)
        .unwrap();

    assert_eq!(engine.count(&cid).unwrap(), 2);
    assert_eq!(
        engine.wal_last_seq(&cid).unwrap(),
        0,
        "a committed batch must not be written to the local WAL"
    );

    // The log index is the sequence number.
    let record = engine.get(&cid, &["a".to_string()]).unwrap();
    assert_eq!(record[0].as_ref().unwrap().seq, 7);
}

/// Replay is how a follower catches up and how a snapshot-ahead-of-boundary is
/// made consistent, so applying the same entry twice must be harmless.
#[test]
fn apply_committed_is_idempotent() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir, ApplyMode::Committed);
    let cid = Uuid::new_v4();
    create(&engine, cid, 100);

    let batch = vec![add("a", 1.0)];
    engine.apply_committed(&cid, &batch, 3, DIM).unwrap();
    engine.apply_committed(&cid, &batch, 3, DIM).unwrap();

    assert_eq!(engine.count(&cid).unwrap(), 1);

    // A later version of the same id wins, as it does through the WAL.
    let updated = vec![Operation::Upsert {
        id: "a".to_string(),
        embedding: vector(9.0),
        metadata: None,
        document: None,
    }];
    engine.apply_committed(&cid, &updated, 4, DIM).unwrap();
    assert_eq!(engine.count(&cid).unwrap(), 1);
    let record = engine.get(&cid, &["a".to_string()]).unwrap();
    assert_eq!(record[0].as_ref().unwrap().seq, 4);
}

/// Applied records must become queryable, which means the write buffer has to
/// be flushed on the same path — there is no WAL replay behind them to do it.
#[test]
fn apply_committed_flushes_into_the_index() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir, ApplyMode::Committed);
    let cid = Uuid::new_v4();
    create(&engine, cid, 2); // flush every two records

    let ops: Vec<Operation> = (0..6).map(|i| add(&format!("id-{i}"), i as f32)).collect();
    engine.apply_committed(&cid, &ops, 11, DIM).unwrap();

    let hits = engine
        .query(
            &cid,
            &vector(3.0),
            3,
            &rekha_engine::QueryOptions {
                where_document: None,
                ef: 0,
                where_filter: None,
                oversampling: 4,
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "id-3");
}

#[test]
fn apply_committed_rejects_an_unknown_collection() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir, ApplyMode::Committed);
    assert!(
        engine
            .apply_committed(&Uuid::new_v4(), &[add("a", 1.0)], 1, DIM)
            .is_err()
    );
}
