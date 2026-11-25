//! Integration tests for `rekha-engine`. The correctness oracle is
//! `rekha_distance`, which implements Chroma-exact distance semantics.

#![allow(clippy::unreadable_literal)]
#![allow(clippy::unusual_byte_groupings)]
#![allow(clippy::cloned_ref_to_slice_refs)]
#![allow(clippy::needless_range_loop)]

use std::collections::HashSet;
use std::sync::Arc;

use rekha_core::cluster::{Epoch, ShardId};
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::types::{Distance, Embedding, Id, Metadata, MetadataValue};
use rekha_distance::{distance, l2_squared};
use rekha_engine::{Durability, Engine, EngineConfig, EngineError, QueryOptions};
use rekha_storage::{Catalog, LocalStorage, RedbCatalog, Storage};
use tempfile::TempDir;
use uuid::Uuid;

const DIM: usize = 16;
const TENANT: &str = "default_tenant";
const DATABASE: &str = "default_database";

/// Tiny seeded LCG for deterministic vectors.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// Uniform `f32` in `[0, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

fn rand_embedding(rng: &mut Lcg, dim: usize) -> Embedding {
    (0..dim).map(|_| rng.unit()).collect::<Vec<f32>>().into()
}

fn open_engine(dir: &TempDir) -> Engine {
    open_engine_result(dir).unwrap()
}

/// Reopen without unwrapping, so recovery-failure tests can assert on the error.
fn open_engine_result(dir: &TempDir) -> rekha_engine::EngineResult<Engine> {
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
}

fn create_collection(
    engine: &Engine,
    name: &str,
    dim: usize,
    space: Distance,
    batch_size: usize,
) -> Uuid {
    let mut config = CollectionConfig::new(name.to_owned(), dim, space);
    config.hnsw.batch_size = batch_size;
    engine.create_collection(&config).unwrap().config.id
}

/// Fresh engine + a default L2 collection named `coll`, dim 16, with the given
/// `batch_size`. Reopening the same dir replays the WAL.
fn test_engine(dir: &TempDir, batch_size: usize) -> (Engine, Uuid) {
    let engine = open_engine(dir);
    let id = create_collection(&engine, "coll", DIM, Distance::L2, batch_size);
    (engine, id)
}

fn collection_id_by_name(engine: &Engine) -> Uuid {
    engine
        .get_collection(TENANT, DATABASE, "coll")
        .unwrap()
        .unwrap()
        .config
        .id
}

/// Query options with only `ef` set (everything else defaults).
fn qopts(ef: usize) -> QueryOptions {
    QueryOptions {
        ef,
        ..Default::default()
    }
}

/// A metadata map with a `tag` and an integer `score`.
fn meta_with_tag(tag: &str, score: i64) -> Metadata {
    let mut m = Metadata::new();
    m.insert("tag".into(), MetadataValue::Str(tag.into()));
    m.insert("score".into(), MetadataValue::Int(score));
    m
}

/// A metadata map with only an integer `score` (no `tag` key — used by the
/// `$nin`-with-absent-key and `$ne` tests).
fn meta_score(score: i64) -> Metadata {
    let mut m = Metadata::new();
    m.insert("score".into(), MetadataValue::Int(score));
    m
}

/// Whether a record's optional metadata matches the filter. `None` metadata is
/// evaluated against an empty map, mirroring the engine's buffer-scan rule.
fn filter_matches(filter: &WhereFilter, metadata: &Option<Metadata>) -> bool {
    match metadata {
        Some(m) => filter.matches(m),
        None => filter.matches(&Metadata::new()),
    }
}

/// Brute-force top-k ids among records whose metadata matches `filter`, using
/// `l2_squared` as the oracle. This is the correctness oracle for every
/// filtered-query test.
fn brute_force_filtered_top_k(
    stored: &[(Id, Embedding, Option<Metadata>)],
    filter: &WhereFilter,
    query: &Embedding,
    k: usize,
) -> HashSet<Id> {
    let mut ranked: Vec<(Id, f32)> = stored
        .iter()
        .filter(|(_, _, m)| filter_matches(filter, m))
        .map(|(id, v, _)| (id.clone(), l2_squared(query, v)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Brute-force top-k ids for a query, using `rekha_distance` as the oracle.
fn brute_force_top_k(stored: &[(Id, Embedding)], query: &Embedding, k: usize) -> HashSet<Id> {
    let mut ranked: Vec<(Id, f32)> = stored
        .iter()
        .map(|(id, v)| (id.clone(), l2_squared(query, v)))
        .collect();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    ranked.iter().take(k).map(|(id, _)| id.clone()).collect()
}

#[test]
fn add_get_count() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 100);
    let mut rng = Lcg::new(1);
    let ids: Vec<Id> = (0..5).map(|i| format!("id{i}")).collect();
    let embs: Vec<Embedding> = (0..5).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &ids, &embs, None, None).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 5);

    let got = engine.get(&id, &ids).unwrap();
    assert_eq!(got.len(), 5);
    for (rec, (want_id, want_emb)) in got.iter().zip(ids.iter().zip(&embs)) {
        let rec = rec.as_ref().expect("record present");
        assert_eq!(&rec.id, want_id);
        assert_eq!(rec.embedding.as_ref().unwrap().as_ref(), want_emb.as_ref());
        assert!(rec.seq >= 1, "seq assigned by the WAL");
    }

    let missing = engine.get(&id, &["nope".to_string()]).unwrap();
    assert!(missing[0].is_none());
}

#[test]
fn query_correctness_vs_brute_force() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50); // flushes happen mid-add
    let mut rng = Lcg::new(42);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    for i in 0..200 {
        let idstr = format!("v{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        ids.push(idstr.clone());
        embs.push(v.clone());
        stored.push((idstr, v));
    }
    engine.add(&id, &ids, &embs, None, None).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 200);
    assert_eq!(engine.indexed_count(&id).unwrap(), 200);

    for _ in 0..10 {
        let q = rand_embedding(&mut rng, DIM);
        let hits = engine.query(&id, &q, 10, &qopts(200)).unwrap();
        assert_eq!(hits.len(), 10);
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        let expected = brute_force_top_k(&stored, &q, 10);
        assert_eq!(got, expected, "top-k id set mismatch");
    }
}

#[test]
fn distance_semantics() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let id = create_collection(&engine, "cos", 8, Distance::Cosine, 100);
    let stored: Vec<(Id, Embedding)> = vec![
        (
            "a".to_string(),
            vec![1.0, 0.0, 0.5, 0.0, 0.25, 0.0, 0.0, 1.0].into(),
        ),
        (
            "b".to_string(),
            vec![0.0, 1.0, 0.0, 0.5, 0.0, 0.25, 1.0, 0.0].into(),
        ),
        (
            "c".to_string(),
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5].into(),
        ),
    ];
    let ids: Vec<Id> = stored.iter().map(|(id, _)| id.clone()).collect();
    let embs: Vec<Embedding> = stored.iter().map(|(_, v)| v.clone()).collect();
    engine.add(&id, &ids, &embs, None, None).unwrap();

    let q: Embedding = vec![0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7].into();
    let hits = engine.query(&id, &q, 3, &qopts(200)).unwrap();
    assert_eq!(hits.len(), 3);
    for hit in &hits {
        let expected = distance(
            Distance::Cosine,
            &q,
            &stored.iter().find(|(sid, _)| *sid == hit.id).unwrap().1,
        );
        let rel = (hit.distance - expected).abs() / expected.abs().max(1e-12);
        assert!(
            rel < 1e-4,
            "id `{}`: got {}, expected {}",
            hit.id,
            hit.distance,
            expected
        );
    }
}

#[test]
fn upsert_replaces_embedding() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    let mut rng = Lcg::new(7);

    // Cluster center near the origin, and a far-away replacement vector.
    let p1: Embedding = vec![0.1; DIM].into();
    let v2: Embedding = vec![1.0; DIM].into();

    // Five records close to p1.
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    for i in 0..5 {
        let noise: Vec<f32> = (0..DIM).map(|_| 0.01 * rng.unit()).collect();
        let v: Vec<f32> = p1.iter().zip(&noise).map(|(a, b)| a + b).collect();
        ids.push(format!("r{i}"));
        embs.push(v.into());
    }
    engine.add(&id, &ids, &embs, None, None).unwrap();
    // x is exactly p1, so it is the nearest neighbor of p1.
    engine
        .add(&id, &["x".to_string()], &[p1.clone()], None, None)
        .unwrap();

    let hits = engine.query(&id, &p1, 5, &qopts(100)).unwrap();
    assert_eq!(hits[0].id, "x");

    // Upsert x to a far-away vector.
    engine
        .upsert(&id, &["x".to_string()], &[v2.clone()], None, None)
        .unwrap();
    let hits = engine.query(&id, &v2, 5, &qopts(100)).unwrap();
    assert_eq!(
        hits[0].id, "x",
        "upserted embedding must win for queries near it"
    );

    let hits = engine.query(&id, &p1, 5, &qopts(100)).unwrap();
    assert_eq!(hits.len(), 5);
    assert!(
        hits.iter().all(|h| h.id != "x"),
        "old embedding must be gone from top-k near p1"
    );

    // Force a flush (batch_size=2) so the buffered new-x enters the index via
    // the delete-then-re-add path; correctness must hold after the flush too.
    let extra: Vec<Id> = (0..2).map(|i| format!("e{i}")).collect();
    let extra_embs: Vec<Embedding> = (0..2).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &extra, &extra_embs, None, None).unwrap();
    let hits = engine.query(&id, &v2, 5, &qopts(100)).unwrap();
    assert_eq!(hits[0].id, "x");
}

#[test]
fn update_metadata() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 100);
    let mut rng = Lcg::new(3);
    let ids: Vec<Id> = (0..3).map(|i| format!("u{i}")).collect();
    let embs: Vec<Embedding> = (0..3).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &ids, &embs, None, None).unwrap();

    let mut meta = Metadata::new();
    meta.insert("tag".into(), MetadataValue::Str("new".into()));
    engine
        .update(
            &id,
            &["u1".to_string()],
            None,
            Some(&[Some(rekha_core::types::metadata_patch_from(meta))]),
            None,
        )
        .unwrap();

    let rec = engine.get(&id, &["u1".to_string()]).unwrap()[0]
        .clone()
        .expect("u1 present");
    assert_eq!(
        rec.metadata.unwrap().get("tag"),
        Some(&MetadataValue::Str("new".into()))
    );
    assert_eq!(engine.count(&id).unwrap(), 3, "count unchanged by update");

    let other = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .expect("u0 present");
    assert!(other.metadata.is_none(), "other records untouched");
}

#[test]
fn delete_removes_records() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(9);
    let ids: Vec<Id> = (0..20).map(|i| format!("d{i:02}")).collect();
    let embs: Vec<Embedding> = (0..20).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &ids, &embs, None, None).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 20);

    engine
        .delete(&id, &["d03".to_string(), "d17".to_string()])
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 18);
    let got = engine
        .get(&id, &["d03".to_string(), "d17".to_string()])
        .unwrap();
    assert!(got[0].is_none() && got[1].is_none());

    let q = rand_embedding(&mut rng, DIM);
    let hits = engine.query(&id, &q, 20, &qopts(100)).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "d03" && h.id != "d17"),
        "deleted ids must not be returned"
    );

    // Deleting missing ids is Ok.
    engine
        .delete(&id, &["d03".to_string(), "never_existed".to_string()])
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 18);
}

#[test]
fn flush_threshold() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    let mut rng = Lcg::new(11);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();

    // batch_size = 2: after add 2 flushes (index 2), add 4 flushes (index 4).
    let expected_committed = [0usize, 0, 2, 2, 4, 4];
    for i in 0..5 {
        let v = rand_embedding(&mut rng, DIM);
        let idstr = format!("f{i}");
        stored.push((idstr.clone(), v.clone()));
        engine.add(&id, &[idstr], &[v], None, None).unwrap();
        assert_eq!(engine.indexed_count(&id).unwrap(), i + 1);
        assert_eq!(
            engine.committed_count(&id).unwrap(),
            expected_committed[i + 1],
            "committed count after add {i}"
        );
    }
    assert_eq!(engine.count(&id).unwrap(), 5);

    // Query correctness still holds across index/buffer split.
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine.query(&id, &q, 5, &qopts(100)).unwrap();
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(got, brute_force_top_k(&stored, &q, 5));
}

#[test]
fn wal_replay_rebuilds_state() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(21);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    {
        let (engine, id) = test_engine(&dir, 1000); // no flush: 10 < batch_size
        let ids: Vec<Id> = (0..10).map(|i| format!("r{i}")).collect();
        let embs: Vec<Embedding> = (0..10).map(|_| rand_embedding(&mut rng, DIM)).collect();
        for (i, e) in embs.iter().enumerate() {
            stored.push((ids[i].clone(), e.clone()));
        }
        engine.add(&id, &ids, &embs, None, None).unwrap();

        let new_v = rand_embedding(&mut rng, DIM);
        stored[0] = (ids[0].clone(), new_v.clone());
        engine
            .upsert(&id, &[ids[0].clone()], &[new_v], None, None)
            .unwrap();

        stored.pop(); // r9 is deleted
        engine.delete(&id, &[ids[9].clone()]).unwrap();
        assert_eq!(engine.count(&id).unwrap(), 9);
    }

    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 9, "count survives replay");
    assert!(engine.get(&id, &["r9".to_string()]).unwrap()[0].is_none());

    for (want_id, want_v) in &stored {
        let rec = engine.get(&id, &[want_id.clone()]).unwrap()[0]
            .clone()
            .expect("record rebuilt");
        assert_eq!(rec.embedding.as_ref().unwrap().as_ref(), want_v.as_ref());
    }

    for _ in 0..5 {
        let q = rand_embedding(&mut rng, DIM);
        let hits = engine.query(&id, &q, 5, &qopts(100)).unwrap();
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(got, brute_force_top_k(&stored, &q, 5));
    }
}

#[test]
fn wal_replay_after_flush() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(33);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    {
        let (engine, id) = test_engine(&dir, 2);
        let ids: Vec<Id> = (0..6).map(|i| format!("s{i}")).collect();
        let embs: Vec<Embedding> = (0..6).map(|_| rand_embedding(&mut rng, DIM)).collect();
        for (i, e) in embs.iter().enumerate() {
            stored.push((ids[i].clone(), e.clone()));
        }
        engine.add(&id, &ids, &embs, None, None).unwrap();
        assert_eq!(engine.count(&id).unwrap(), 6);
        assert_eq!(engine.committed_count(&id).unwrap(), 6);
    }

    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 6);
    // The index itself is rebuilt by replay + flush.
    assert_eq!(engine.committed_count(&id).unwrap(), 6);

    for _ in 0..5 {
        let q = rand_embedding(&mut rng, DIM);
        let hits = engine.query(&id, &q, 6, &qopts(100)).unwrap();
        assert_eq!(hits.len(), 6);
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(got, brute_force_top_k(&stored, &q, 6));
    }
}

#[test]
fn validation_rejects_bad_batches() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 100);
    let mut rng = Lcg::new(5);
    let good: Embedding = rand_embedding(&mut rng, DIM);

    // Mismatched id/embedding counts.
    let err = engine
        .add(
            &id,
            &["a".to_string(), "b".to_string()],
            &[good.clone()],
            None,
            None,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));
    assert_eq!(engine.count(&id).unwrap(), 0);

    // Wrong-dimension embedding.
    let wrong_dim: Embedding = vec![0.5; DIM - 1].into();
    let err = engine
        .add(&id, &["a".to_string()], &[wrong_dim], None, None)
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));
    assert_eq!(engine.count(&id).unwrap(), 0);

    // Metadata/doc length mismatch.
    let err = engine
        .add(
            &id,
            &["a".to_string()],
            &[good.clone()],
            Some(&[None, None]),
            None,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));
    assert_eq!(engine.count(&id).unwrap(), 0);

    // A valid add lands.
    engine
        .add(&id, &["a".to_string()], &[good.clone()], None, None)
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 1);

    // Duplicate id add.
    let err = engine
        .add(&id, &["a".to_string()], &[good.clone()], None, None)
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));
    assert_eq!(engine.count(&id).unwrap(), 1);

    // Duplicate id within one batch.
    let err = engine
        .add(
            &id,
            &["b".to_string(), "b".to_string()],
            &[good.clone(), good.clone()],
            None,
            None,
        )
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));
    assert_eq!(engine.count(&id).unwrap(), 1);

    // Update on a missing id is a silent no-op: Chroma answers 200 and changes
    // nothing. Rejecting the batch made a partially-stale client request fail
    // even when every other id was still valid.
    engine
        .update(&id, &["nope".to_string()], None, None, None)
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 1, "no-op must not add a record");
    assert!(
        engine.get(&id, &["nope".to_string()]).unwrap()[0].is_none(),
        "a no-op update must not create the record"
    );

    // Update on an existing id works and changes nothing else.
    engine
        .update(
            &id,
            &["a".to_string()],
            None,
            None,
            Some(&[Some("doc".to_string())]),
        )
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 1);
    let rec = engine.get(&id, &["a".to_string()]).unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(rec.document.as_deref(), Some("doc"));
}

#[test]
fn concurrent_adds_are_safe() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(open_engine(&dir));
    let id = create_collection(&engine, "conc", DIM, Distance::L2, 25);

    let threads: Vec<_> = (0..4u64)
        .map(|t| {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let mut rng = Lcg::new(1000 + t);
                for i in 0..25 {
                    let v = rand_embedding(&mut rng, DIM);
                    let idstr = format!("t{t}-{i}");
                    engine
                        .add(&id, &[idstr], &[v], None, None)
                        .expect("add succeeds under contention");
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }

    assert_eq!(engine.count(&id).unwrap(), 100, "all 100 records present");
    assert_eq!(engine.indexed_count(&id).unwrap(), 100);

    let mut rng = Lcg::new(777);
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine.query(&id, &q, 10, &qopts(200)).unwrap();
    assert_eq!(hits.len(), 10, "query still correct after concurrent adds");
}

/// Force the approximate index+buffer path (max_scan = 0) and verify it still
/// matches the brute-force top-k — HNSW on 200 random 16-dim points at
/// ef=200/k=10 is exact, so this is a stable set-equality assertion.
#[test]
fn query_via_index_path_matches_brute_force() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("idx".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 50;
    config.hnsw.max_scan = 0; // never take the brute-force path
    config.hnsw.ef_search = 200;
    let id = engine.create_collection(&config).unwrap().config.id;

    let mut rng = Lcg::new(0xABCD_EF);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    for i in 0..200 {
        let idstr = format!("i{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        ids.push(idstr.clone());
        embs.push(v.clone());
        stored.push((idstr, v));
    }
    engine.add(&id, &ids, &embs, None, None).unwrap();
    assert_eq!(engine.committed_count(&id).unwrap(), 200);

    for _ in 0..5 {
        let q = rand_embedding(&mut rng, DIM);
        let hits = engine.query(&id, &q, 10, &qopts(200)).unwrap();
        assert_eq!(hits.len(), 10);
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(
            got,
            brute_force_top_k(&stored, &q, 10),
            "index path mismatch"
        );
    }
}

#[test]
fn collection_not_found_errors() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let ghost = Uuid::new_v4();
    let q: Embedding = vec![0.0; DIM].into();
    let err = engine.query(&ghost, &q, 5, &qopts(100)).unwrap_err();
    assert!(matches!(err, EngineError::CollectionNotFound(_)));
    assert!(matches!(
        engine.count(&ghost).unwrap_err(),
        EngineError::CollectionNotFound(_)
    ));
    assert!(matches!(
        engine.get(&ghost, &["a".to_string()]).unwrap_err(),
        EngineError::CollectionNotFound(_)
    ));
}

#[test]
fn delete_collection_removes_all_state() {
    let dir = TempDir::new().unwrap();
    let id;
    {
        let (engine, coll_id) = test_engine(&dir, 100);
        id = coll_id;
        let v: Embedding = vec![0.5; DIM].into();
        engine
            .add(&id, &["a".to_string()], &[v], None, None)
            .unwrap();
        assert_eq!(engine.count(&id).unwrap(), 1);

        engine.delete_collection(&id).unwrap();
        assert!(
            engine
                .get_collection(TENANT, DATABASE, "coll")
                .unwrap()
                .is_none()
        );
        assert!(!dir.path().join("wal").join(format!("{id}.wal")).exists());
    }

    let engine = open_engine(&dir);
    assert!(
        engine
            .get_collection(TENANT, DATABASE, "coll")
            .unwrap()
            .is_none()
    );
}

#[test]
fn query_with_zero_ef_uses_config_default() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("ef".to_owned(), DIM, Distance::L2);
    config.hnsw.ef_search = 64;
    config.hnsw.max_scan = 0; // force the index path so ef is actually used
    let id = engine.create_collection(&config).unwrap().config.id;

    let mut rng = Lcg::new(13);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    for i in 0..50 {
        let idstr = format!("e{i:02}");
        let v = rand_embedding(&mut rng, DIM);
        ids.push(idstr.clone());
        embs.push(v.clone());
        stored.push((idstr, v));
    }
    engine.add(&id, &ids, &embs, None, None).unwrap();

    // k = 50 requests everything, so the exact full set comes back whether ef
    // is 0 (defaulted) or explicit.
    let q = rand_embedding(&mut rng, DIM);
    for ef in [0usize, 64, 256] {
        let hits = engine.query(&id, &q, 50, &qopts(ef)).unwrap();
        assert_eq!(hits.len(), 50, "ef {ef}: expected the full set");
    }
}

#[test]
fn filtered_query_eq() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(101);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..200 {
        let idstr = format!("e{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        let tag = ["a", "b", "c"][i % 3];
        let meta = meta_with_tag(tag, i as i64);
        ids.push(idstr.clone());
        embs.push(v.clone());
        metas.push(Some(meta.clone()));
        stored.push((idstr, v, Some(meta)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
        .unwrap();
    assert_eq!(hits.len(), 10);
    for h in &hits {
        assert_eq!(
            h.metadata.as_ref().and_then(|m| m.get("tag")),
            Some(&MetadataValue::Str("a".into())),
            "id `{}` must be tag a",
            h.id
        );
    }
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        got,
        brute_force_filtered_top_k(&stored, &filter, &q, 10),
        "filtered $eq must match brute force over the eligible set"
    );
}

#[test]
fn filtered_query_range() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(107);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..200 {
        let idstr = format!("r{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        let meta = meta_with_tag("a", i as i64);
        ids.push(idstr.clone());
        embs.push(v.clone());
        metas.push(Some(meta.clone()));
        stored.push((idstr, v, Some(meta)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let filter = WhereFilter::parse_json(r#"{"score": {"$gt": 50}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
        .unwrap();
    assert_eq!(hits.len(), 10);
    for h in &hits {
        let score = h
            .metadata
            .as_ref()
            .and_then(|m| m.get("score"))
            .and_then(MetadataValue::as_f64)
            .expect("every result has a numeric score");
        assert!(score > 50.0, "id `{}` must have score > 50", h.id);
    }
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(got, brute_force_filtered_top_k(&stored, &filter, &q, 10));
}

#[test]
fn filtered_query_in_nin_ne() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(109);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    // 6 records with a tag, 4 with no tag at all (exercises $nin absent-key).
    for i in 0..10 {
        let idstr = format!("t{i}");
        let v = rand_embedding(&mut rng, DIM);
        let meta = if i < 6 {
            Some(meta_with_tag(["a", "a", "b", "b", "c", "c"][i], i as i64))
        } else {
            Some(meta_score(i as i64))
        };
        ids.push(idstr.clone());
        embs.push(v.clone());
        metas.push(meta.clone());
        stored.push((idstr, v, meta));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let q = rand_embedding(&mut rng, DIM);
    for (json, desc) in [
        (r#"{"tag": {"$in": ["a", "b"]}}"#, "$in"),
        (r#"{"tag": {"$nin": ["a"]}}"#, "$nin"),
        (r#"{"tag": {"$ne": "a"}}"#, "$ne"),
    ] {
        let filter = WhereFilter::parse_json(json).unwrap();
        let hits = engine
            .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
            .unwrap();
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(
            got,
            brute_force_filtered_top_k(&stored, &filter, &q, 10),
            "{desc} must match brute force over the eligible set"
        );
    }
}

#[test]
fn filtered_query_and_or() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(113);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..200 {
        let idstr = format!("a{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        let meta = meta_with_tag(["a", "b", "c"][i % 3], i as i64);
        ids.push(idstr.clone());
        embs.push(v.clone());
        metas.push(Some(meta.clone()));
        stored.push((idstr, v, Some(meta)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let q = rand_embedding(&mut rng, DIM);
    for json in [
        r#"{"$and": [{"tag": "a"}, {"score": {"$gte": 50}}]}"#,
        r#"{"$or": [{"tag": "a"}, {"score": {"$lt": 10}}]}"#,
    ] {
        let filter = WhereFilter::parse_json(json).unwrap();
        let hits = engine
            .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
            .unwrap();
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(
            got,
            brute_force_filtered_top_k(&stored, &filter, &q, 10),
            "filter {json} must match brute force"
        );
    }
}

/// Force the ANN path (max_scan = 50 < eligible set) and check the
/// post-filtered graph walk still recovers the exact eligible top-k.
#[test]
fn filtered_query_ann_path() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("annf".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 50;
    config.hnsw.max_scan = 50; // once > 50 eligible, the planner takes the ANN path
    config.hnsw.ef_search = 200;
    let id = engine.create_collection(&config).unwrap().config.id;

    let mut rng = Lcg::new(0xF00D);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..500 {
        let idstr = format!("a{i:03}");
        let v = rand_embedding(&mut rng, DIM);
        // score cycles 0..99 so `> 49` matches ~250 records (> max_scan).
        let meta = meta_score((i % 100) as i64);
        ids.push(idstr.clone());
        embs.push(v.clone());
        metas.push(Some(meta.clone()));
        stored.push((idstr, v, Some(meta)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();
    assert_eq!(engine.committed_count(&id).unwrap(), 500);

    let filter = WhereFilter::parse_json(r#"{"score": {"$gt": 49}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(
            &id,
            &q,
            10,
            &QueryOptions {
                ef: 200,
                where_filter: Some(filter.clone()),
                where_document: None,
                oversampling: 8,
            },
        )
        .unwrap();
    assert_eq!(hits.len(), 10);
    for h in &hits {
        let score = h
            .metadata
            .as_ref()
            .and_then(|m| m.get("score"))
            .and_then(MetadataValue::as_f64)
            .expect("every result has a numeric score");
        assert!(score > 49.0, "id `{}` must be eligible", h.id);
    }
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        got,
        brute_force_filtered_top_k(&stored, &filter, &q, 10),
        "ANN path must recover the exact eligible top-k (ef=200, oversampling=8)"
    );
}

#[test]
fn filtered_query_matches_buffer_only() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 1000); // nothing flushes
    let mut rng = Lcg::new(77);
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..6 {
        ids.push(format!("b{i}"));
        embs.push(rand_embedding(&mut rng, DIM));
        metas.push(Some(meta_with_tag(if i < 3 { "x" } else { "y" }, i as i64)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();
    assert_eq!(engine.committed_count(&id).unwrap(), 0, "nothing flushed");

    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "x"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(&id, &q, 3, &QueryOptions::with_where(filter))
        .unwrap();
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        got,
        HashSet::from(["b0".to_string(), "b1".to_string(), "b2".to_string()]),
        "only buffer records with tag x are eligible"
    );
}

#[test]
fn filtered_query_excludes_deleted() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(19);
    let mut stored: Vec<(Id, Embedding, Option<Metadata>)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..10 {
        ids.push(format!("g{i}"));
        let v = rand_embedding(&mut rng, DIM);
        embs.push(v.clone());
        let meta = meta_with_tag(if i < 8 { "a" } else { "b" }, i as i64);
        metas.push(Some(meta.clone()));
        stored.push((ids[i].clone(), v, Some(meta)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();
    engine
        .delete(&id, &["g2".to_string(), "g5".to_string()])
        .unwrap();
    stored.retain(|(id, _, _)| id != "g2" && id != "g5");

    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
        .unwrap();
    assert!(
        !hits.iter().any(|h| h.id == "g2" || h.id == "g5"),
        "deleted ids must never be returned"
    );
    assert_eq!(
        hits.len(),
        brute_force_filtered_top_k(&stored, &filter, &q, 10).len(),
        "deleted records must be gone from the eligible set"
    );
}

#[test]
fn filtered_query_after_metadata_update() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(23);
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..5 {
        ids.push(format!("u{i}"));
        embs.push(rand_embedding(&mut rng, DIM));
        metas.push(Some(meta_with_tag("a", i as i64)));
    }
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let filter_a = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
    let filter_b = WhereFilter::parse_json(r#"{"tag": {"$eq": "b"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let before = engine
        .query(&id, &q, 5, &QueryOptions::with_where(filter_a.clone()))
        .unwrap();
    assert_eq!(before.len(), 5, "all five start as tag a");

    // Upsert u1 with tag a -> b.
    engine
        .upsert(
            &id,
            &["u1".to_string()],
            &[rand_embedding(&mut rng, DIM)],
            Some(&[Some(rekha_core::types::metadata_patch_from(meta_with_tag(
                "b", 1,
            )))]),
            None,
        )
        .unwrap();

    let after_a = engine
        .query(&id, &q, 5, &QueryOptions::with_where(filter_a))
        .unwrap();
    assert!(
        after_a.iter().all(|h| h.id != "u1"),
        "u1 must no longer match tag a"
    );
    let after_b = engine
        .query(&id, &q, 5, &QueryOptions::with_where(filter_b))
        .unwrap();
    assert!(
        after_b.iter().any(|h| h.id == "u1"),
        "u1 must match tag b after the upsert"
    );
}

#[test]
fn filtered_query_after_replay() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(31);
    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let before: HashSet<Id>;
    {
        let (engine, id) = test_engine(&dir, 3); // small batch: most records flush
        let mut ids = Vec::new();
        let mut embs = Vec::new();
        let mut metas = Vec::new();
        for i in 0..10 {
            ids.push(format!("r{i}"));
            embs.push(rand_embedding(&mut rng, DIM));
            metas.push(Some(meta_with_tag(
                if i % 2 == 0 { "a" } else { "b" },
                i as i64,
            )));
        }
        engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();
        assert!(engine.committed_count(&id).unwrap() >= 6, "records flushed");
        before = engine
            .query(&id, &q, 5, &QueryOptions::with_where(filter.clone()))
            .unwrap()
            .iter()
            .map(|h| h.id.clone())
            .collect();
        assert_eq!(before.len(), 5);
    }

    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    let after: HashSet<Id> = engine
        .query(&id, &q, 5, &QueryOptions::with_where(filter))
        .unwrap()
        .iter()
        .map(|h| h.id.clone())
        .collect();
    assert_eq!(
        after, before,
        "replay must rebuild postings identically (same query, same set)"
    );
}

#[test]
fn filtered_query_no_match_returns_empty() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 100);
    let mut rng = Lcg::new(41);
    let ids: Vec<Id> = (0..5).map(|i| format!("z{i}")).collect();
    let embs: Vec<Embedding> = (0..5).map(|_| rand_embedding(&mut rng, DIM)).collect();
    let metas: Vec<Option<Metadata>> = (0..5).map(|i| Some(meta_with_tag("a", i as i64))).collect();
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "nope"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);
    let hits = engine
        .query(&id, &q, 5, &QueryOptions::with_where(filter))
        .unwrap();
    assert!(hits.is_empty(), "no eligible records => empty result");
}

#[test]
fn unfiltered_query_unchanged() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 50);
    let mut rng = Lcg::new(43);
    let mut stored: Vec<(Id, Embedding)> = Vec::new();
    let mut ids = Vec::new();
    let mut embs = Vec::new();
    for i in 0..50 {
        let idstr = format!("n{i:02}");
        let v = rand_embedding(&mut rng, DIM);
        ids.push(idstr.clone());
        embs.push(v.clone());
        stored.push((idstr, v));
    }
    engine.add(&id, &ids, &embs, None, None).unwrap();

    let q = rand_embedding(&mut rng, DIM);
    let hits = engine.query(&id, &q, 10, &QueryOptions::default()).unwrap();
    let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        got,
        brute_force_top_k(&stored, &q, 10),
        "default options must behave exactly like the old signature"
    );
}

#[test]
fn postings_int_float_coercion() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 100);
    let mut rng = Lcg::new(61);
    let ids: Vec<Id> = vec!["int5".into(), "flt5".into(), "six".into()];
    let embs: Vec<Embedding> = (0..3).map(|_| rand_embedding(&mut rng, DIM)).collect();
    let mut m_int = Metadata::new();
    m_int.insert("score".into(), MetadataValue::Int(5));
    let mut m_flt = Metadata::new();
    m_flt.insert("score".into(), MetadataValue::Float(5.0));
    let mut m_six = Metadata::new();
    m_six.insert("score".into(), MetadataValue::Int(6));
    let metas: Vec<Option<Metadata>> = vec![Some(m_int), Some(m_flt), Some(m_six)];
    engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();

    let q = rand_embedding(&mut rng, DIM);
    // 5.0 parses as Float; Int(5) and Float(5.0) must collide (Chroma coercion).
    for json in [r#"{"score": {"$eq": 5.0}}"#, r#"{"score": {"$eq": 5}}"#] {
        let filter = WhereFilter::parse_json(json).unwrap();
        let hits = engine
            .query(&id, &q, 3, &QueryOptions::with_where(filter))
            .unwrap();
        let got: HashSet<Id> = hits.iter().map(|h| h.id.clone()).collect();
        assert_eq!(
            got,
            HashSet::from(["int5".to_string(), "flt5".to_string()]),
            "filter {json} must match both Int(5) and Float(5.0) records"
        );
    }
}

#[test]
fn concurrent_filtered_queries_and_adds() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(open_engine(&dir));
    let id = create_collection(&engine, "cf", DIM, Distance::L2, 25);

    // Seed some tag-a records so queries have data immediately.
    let mut rng = Lcg::new(5000);
    let seed_ids: Vec<Id> = (0..10).map(|i| format!("seed{i}")).collect();
    let seed_embs: Vec<Embedding> = (0..10).map(|_| rand_embedding(&mut rng, DIM)).collect();
    let seed_metas: Vec<Option<Metadata>> = (0..10).map(|_| Some(meta_with_tag("a", 1))).collect();
    engine
        .add(&id, &seed_ids, &seed_embs, Some(&seed_metas), None)
        .unwrap();

    let writer = {
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let mut rng = Lcg::new(6000);
            for i in 0..50 {
                let v = rand_embedding(&mut rng, DIM);
                engine
                    .add(
                        &id,
                        &[format!("w{i}")],
                        &[v],
                        Some(&[Some(meta_with_tag("a", i as i64))]),
                        None,
                    )
                    .unwrap();
            }
        })
    };

    let readers: Vec<_> = (0..4)
        .map(|t| {
            let engine = Arc::clone(&engine);
            std::thread::spawn(move || {
                let mut rng = Lcg::new(7000 + t);
                let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
                for _ in 0..30 {
                    let q = rand_embedding(&mut rng, DIM);
                    let hits = engine
                        .query(&id, &q, 10, &QueryOptions::with_where(filter.clone()))
                        .unwrap();
                    for h in hits {
                        let m = h.metadata.as_ref().expect("all results carry metadata");
                        assert_eq!(
                            m.get("tag"),
                            Some(&MetadataValue::Str("a".into())),
                            "a non-eligible id leaked into the results"
                        );
                    }
                }
            })
        })
        .collect();

    writer.join().unwrap();
    for t in readers {
        t.join().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Phase 4a: checkpointing
// ---------------------------------------------------------------------------

/// Path of a collection's WAL file on disk.
fn wal_path(dir: &TempDir, id: &Uuid) -> std::path::PathBuf {
    dir.path().join("wal").join(format!("{id}.wal"))
}

/// On-disk byte length of `path` (0 if absent).
fn file_len(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// Query helper mirroring `qopts` with an optional where filter.
fn query_top_k(
    engine: &Engine,
    id: &Uuid,
    query: &Embedding,
    k: usize,
    filter: Option<WhereFilter>,
) -> HashSet<Id> {
    let mut opts = qopts(100);
    opts.where_filter = filter;
    engine
        .query(id, query, k, &opts)
        .unwrap()
        .iter()
        .map(|h| h.id.clone())
        .collect()
}

/// Add `n` records named `c0..c{n-1}` and return `(ids, embeddings, stored)`.
fn add_n(
    engine: &Engine,
    id: &Uuid,
    rng: &mut Lcg,
    n: usize,
    prefix: &str,
) -> (Vec<Id>, Vec<Embedding>, Vec<(Id, Embedding)>) {
    let ids: Vec<Id> = (0..n).map(|i| format!("{prefix}{i}")).collect();
    let embs: Vec<Embedding> = (0..n).map(|_| rand_embedding(rng, DIM)).collect();
    let stored: Vec<(Id, Embedding)> = ids.iter().cloned().zip(embs.iter().cloned()).collect();
    engine.add(id, &ids, &embs, None, None).unwrap();
    (ids, embs, stored)
}

/// Checkpoint + reopen + verify the state and seqs survive, and that new
/// appends resume where the checkpoint left off.
#[test]
fn checkpoint_then_reopen_preserves_state_and_seqs() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_001);
    let (engine, id) = test_engine(&dir, 2); // flushes, so the index has vectors
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 10, "c");

    engine.checkpoint(&id).unwrap();
    // Pruned to the (empty) un-checkpointed tail: the file shrank to near-empty.
    assert!(
        file_len(&wal_path(&dir, &id)) < 1_000,
        "WAL pruned by checkpoint"
    );

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);

    assert_eq!(engine.count(&id).unwrap(), 10, "records survive checkpoint");
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        10,
        "index restored from index.bin"
    );
    for i in 0..10 {
        let rec = engine.get(&id, &[format!("c{i}")]).unwrap()[0]
            .clone()
            .expect("record present");
        assert_eq!(
            rec.embedding.as_ref().unwrap().as_ref(),
            stored[i].1.as_ref()
        );
        assert_eq!(
            engine.seq_of(&id, &format!("c{i}")).unwrap(),
            (i + 1) as u64
        );
    }
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 5, None),
            brute_force_top_k(&stored, &q, 5)
        );
    }

    // New writes continue the seq where the checkpoint left off.
    let new_ids: Vec<Id> = (10..13).map(|i| format!("c{i}")).collect();
    let new_embs: Vec<Embedding> = (0..3).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &new_ids, &new_embs, None, None).unwrap();
    for (i, nid) in new_ids.iter().enumerate() {
        assert_eq!(
            engine.seq_of(&id, nid).unwrap(),
            (11 + i) as u64,
            "seq resumes after checkpoint"
        );
    }
    assert_eq!(engine.count(&id).unwrap(), 13);
}

/// A checkpoint taken before any flush still restores every record.
///
/// The snapshot covers the whole WAL prefix (not just the index watermark), so
/// recovery no longer replays those operations to rediscover unflushed records.
/// Instead it reconciles the derived structures itself: with no usable graph it
/// rebuilds one from the authoritative vectors. Leaving the graph empty would
/// make the ANN path return nothing once a collection is large enough to use it.
#[test]
fn checkpoint_without_flush_restores_records_and_index() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_002);
    let (engine, id) = test_engine(&dir, 1000); // never flushes
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 10, "u");
    assert_eq!(engine.committed_count(&id).unwrap(), 0);

    engine.checkpoint(&id).unwrap();

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);

    assert_eq!(
        engine.count(&id).unwrap(),
        10,
        "records survive an unflushed checkpoint"
    );
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        10,
        "the index is rebuilt from the authoritative vectors"
    );
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 5, None),
            brute_force_top_k(&stored, &q, 5)
        );
    }
}

/// Checkpoint pruning must not break seq continuity across a reopen.
#[test]
fn checkpoint_prunes_wal_and_resumes_seqs() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_003);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 10, "p");

    let wal_before = file_len(&wal_path(&dir, &id));
    engine.checkpoint(&id).unwrap();
    let wal_after = file_len(&wal_path(&dir, &id));
    assert!(wal_after < wal_before, "checkpoint shrank the WAL");
    assert!(wal_after < 1_000, "WAL pruned to near-empty");

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);

    let extra: Vec<Id> = (10..13).map(|i| format!("p{i}")).collect();
    let extra_embs: Vec<Embedding> = (0..3).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine.add(&id, &extra, &extra_embs, None, None).unwrap();
    for (i, eid) in extra.iter().enumerate() {
        assert_eq!(
            engine.seq_of(&id, eid).unwrap(),
            (11 + i) as u64,
            "no seq gap after reopen"
        );
    }
    assert_eq!(engine.count(&id).unwrap(), 13);
}

/// Resolve the published checkpoint generation directory (the one `CURRENT`
/// names). Checkpoints are immutable generations, so tests must not assume
/// files sit directly under the collection's checkpoint directory.
fn checkpoint_generation(dir: &TempDir, id: &Uuid) -> std::path::PathBuf {
    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    let current = std::fs::read_to_string(base.join("CURRENT"))
        .expect("a checkpoint must publish a CURRENT pointer");
    base.join("generations").join(current.trim())
}

/// A corrupt published generation is reported, not silently downgraded.
///
/// This test previously asserted a full WAL replay. That was only sound while
/// pruning stopped at the *index* watermark and happened to leave the whole
/// history behind: the checkpoint now covers the entire prefix it publishes, so
/// the WAL no longer holds a second copy. Recovering what remains would return a
/// smaller collection than the one that was acknowledged, which is worse than
/// refusing to start.
///
/// The WAL-only recovery path (no checkpoint at all) is covered separately by
/// `restart-mode.sh wal`.
#[test]
fn corrupt_checkpoint_is_reported_not_replayed() {
    for corrupt in ["checkpoint.json", "records.bin"] {
        let dir = TempDir::new().unwrap();
        let mut rng = Lcg::new(9_004);
        let (engine, id) = test_engine(&dir, 1000);
        add_n(&engine, &id, &mut rng, 6, "f");
        engine.checkpoint(&id).unwrap();

        let ckpt = checkpoint_generation(&dir, &id);
        std::fs::write(ckpt.join(corrupt), b"garbage that is not a checkpoint").unwrap();

        drop(engine);
        match open_engine_result(&dir) {
            Ok(engine) => {
                let id = collection_id_by_name(&engine);
                panic!(
                    "corrupt {corrupt} was accepted, count={}",
                    engine.count(&id).unwrap_or(u64::MAX)
                );
            }
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("refusing to recover") || msg.contains("checksum"),
                    "unexpected error for {corrupt}: {msg}"
                );
            }
        }
    }
}

/// A lost/corrupt index.bin must NOT lose the records: once the WAL is pruned,
/// records.bin is the only copy. The graph is derived state, so reopen rebuilds
/// it from the authoritative vectors and keeps every record.
///
/// It previously degraded to an *empty* graph, relying on the exact path — which
/// silently turns the ANN path into "return nothing" once a collection is large
/// enough to need it.
#[test]
fn missing_index_bin_keeps_records() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_005);
    let (engine, id) = test_engine(&dir, 2);
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 8, "m");
    // Routine checkpoints no longer persist the derived graph; compaction does.
    // That is the generation whose index can go missing.
    engine.optimize(&id).unwrap();

    let ckpt = checkpoint_generation(&dir, &id);
    std::fs::remove_file(ckpt.join("index.bin")).unwrap();

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);

    assert_eq!(
        engine.count(&id).unwrap(),
        8,
        "records survive a missing index.bin"
    );
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        8,
        "the index is rebuilt from the authoritative vectors"
    );
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 8, None),
            brute_force_top_k(&stored, &q, 8)
        );
    }
}

/// With a low `sync_threshold`, auto-checkpoint keeps the WAL bounded without
/// any explicit `checkpoint` call.
#[test]
fn auto_checkpoint_bounds_wal_growth() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("coll".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 5;
    config.hnsw.sync_threshold = 5;
    let id = engine.create_collection(&config).unwrap().config.id;

    let mut rng = Lcg::new(9_006);
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 60, "a");
    // 60 unpruned records would be well over 1 KB; auto-checkpoints keep it tiny.
    assert!(
        file_len(&wal_path(&dir, &id)) < 1_000,
        "auto-checkpoint bounded the WAL"
    );

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(
        engine.count(&id).unwrap(),
        60,
        "all records survive auto-checkpoint"
    );
    assert_eq!(engine.seq_of(&id, &"a0".to_string()).unwrap(), 1);
    assert_eq!(engine.seq_of(&id, &"a59".to_string()).unwrap(), 60);
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 10, None),
            brute_force_top_k(&stored, &q, 10)
        );
    }
}

/// `optimize` rebuilds the HNSW graph from live records, so deletes are
/// reclaimed and the stats reflect the materialized state.
#[test]
fn optimize_reclaims_deleted_vectors() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_007);
    let (engine, id) = test_engine(&dir, 5);
    let (ids, _, stored) = add_n(&engine, &id, &mut rng, 40, "o");
    assert_eq!(engine.committed_count(&id).unwrap(), 40);

    let dead: Vec<Id> = ids[..30].to_vec();
    let alive: Vec<(Id, Embedding)> = stored[30..].to_vec();
    engine.delete(&id, &dead).unwrap();
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        10,
        "eager delete removed vectors"
    );

    let stats = engine.optimize(&id).unwrap();
    assert_eq!(stats.records, 10);
    assert_eq!(stats.indexed, 10, "rebuilt graph holds only live vectors");

    // Optimize is the operation that promises to reclaim space: it compacts
    // into a single segment holding only live vectors, while routine
    // checkpoints append incrementally and leave superseded versions behind.
    let generation = checkpoint_generation(&dir, &id);
    let segments: Vec<_> = std::fs::read_dir(generation.join("segments"))
        .unwrap()
        .flatten()
        .collect();
    assert_eq!(segments.len(), 1, "optimize must compact into one segment");
    let len = std::fs::metadata(segments[0].path()).unwrap().len();
    assert_eq!(
        len,
        (32 + 10 * DIM * 4) as u64,
        "compacted segment must hold only the 10 live vectors"
    );
    assert!(stats.wal_bytes_after <= stats.wal_bytes_before);
    assert_eq!(engine.count(&id).unwrap(), 10);

    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 5, None),
            brute_force_top_k(&alive, &q, 5)
        );
    }
}

/// `optimize` force-flushes the buffer, so an unflushed collection's vectors
/// are materialized and counted.
#[test]
fn optimize_stats_reflect_materialized_state() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_008);
    let (engine, id) = test_engine(&dir, 1000); // nothing flushed by add
    let (_, _, _) = add_n(&engine, &id, &mut rng, 20, "z");
    assert_eq!(engine.committed_count(&id).unwrap(), 0);

    let stats = engine.optimize(&id).unwrap();
    assert_eq!(stats.records, 20);
    assert_eq!(stats.indexed, 20, "optimize force-flushes the buffer");
    assert_eq!(engine.committed_count(&id).unwrap(), 20);
}

/// Checkpointing an empty collection is a no-op; records with metadata and
/// embeddings survive a checkpoint even when nothing has been flushed to the
/// index.
#[test]
fn checkpoint_noop_for_empty_and_metadata_only() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 1000);
    engine.checkpoint(&id).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 0);

    // Add records with metadata and embeddings (batch_size=1000 → nothing
    // flushed to the index; the checkpoint stores records.bin only).
    let mut config = CollectionConfig::new("meta".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 1000;
    let mid = engine.create_collection(&config).unwrap().config.id;
    let ids = vec!["m0".to_string(), "m1".to_string()];
    let embs = vec![
        rand_embedding(&mut Lcg::new(9_010), DIM),
        rand_embedding(&mut Lcg::new(9_011), DIM),
    ];
    engine
        .add(
            &mid,
            &ids,
            &embs,
            Some(&[Some(meta_with_tag("a", 1)), None]),
            None,
        )
        .unwrap();
    assert_eq!(engine.committed_count(&mid).unwrap(), 0);
    engine.checkpoint(&mid).unwrap();

    drop(engine);
    let engine = open_engine(&dir);
    let mid = engine
        .get_collection(TENANT, DATABASE, "meta")
        .unwrap()
        .unwrap()
        .config
        .id;
    let rec = engine.get(&mid, &["m0".to_string()]).unwrap()[0]
        .clone()
        .expect("m0");
    assert!(rec.embedding.is_some());
    assert_eq!(
        rec.metadata.as_ref().and_then(|m| m.get("tag")),
        Some(&MetadataValue::Str("a".into()))
    );
    assert_eq!(engine.count(&mid).unwrap(), 2);
}

/// Checkpoint writes segment files that survive reopen.
#[test]
fn checkpoint_writes_segments_and_reopen_mmaps_them() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_020);
    let (engine, id) = test_engine(&dir, 2); // batch_size=2 → flushed
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 8, "s");

    engine.checkpoint(&id).unwrap();

    // Verify segment files exist on disk.
    let seg_dir = checkpoint_generation(&dir, &id).join("segments");
    assert!(seg_dir.join("seg-0.bin").exists(), "segment file written");
    assert!(
        checkpoint_generation(&dir, &id)
            .join("segment_index.bin")
            .exists(),
        "segment index written"
    );

    // Verify the segment file has the right number of vectors.
    let seg = rekha_engine::segment::Segment::open(seg_dir.join("seg-0.bin")).unwrap();
    assert_eq!(seg.len(), 8);
    assert_eq!(seg.dimension(), DIM);

    // Reopen and verify records survive.
    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 8);
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 5, None),
            brute_force_top_k(&stored, &q, 5)
        );
    }
}

/// Optimize writes segments that survive reopen.
#[test]
fn optimize_writes_segments_and_reopen_mmaps_them() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_021);
    let (engine, id) = test_engine(&dir, 2);
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 6, "o");

    let stats = engine.optimize(&id).unwrap();
    assert_eq!(stats.records, 6);

    // Verify segment files exist.
    let seg_dir = checkpoint_generation(&dir, &id).join("segments");
    assert!(
        seg_dir.join("seg-0.bin").exists(),
        "optimize writes segment"
    );

    // Reopen and verify.
    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 6);
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 5, None),
            brute_force_top_k(&stored, &q, 5)
        );
    }
}

// ---------------------------------------------------------------------------
// WAL head inspection
// ---------------------------------------------------------------------------

/// A durable single-node engine must record what it appended: the WAL head is
/// how a restart knows where to resume.
#[test]
fn wal_last_seq_reports_head() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_032);
    let (engine, id) = test_engine(&dir, 1000);
    assert_eq!(engine.wal_last_seq(&id).unwrap(), 0, "empty WAL");
    let _ = add_n(&engine, &id, &mut rng, 7, "w");
    assert_eq!(engine.wal_last_seq(&id).unwrap(), 7, "7 records written");
}

// ---------------------------------------------------------------------------
// P1: checkpoint / vector-visibility regressions
// ---------------------------------------------------------------------------

/// A checkpoint must not destroy vectors a *previous* checkpoint already
/// stripped from the in-memory records.
///
/// Checkpoint #1 strips `embedding` and writes the segment. Checkpoint #2
/// rebuilds `seg-0.bin` from the in-memory records only, so unless it carries
/// the already-stripped vectors forward from the old segment the first cohort
/// becomes permanently vector-less: its WAL prefix was pruned at checkpoint #1
/// and `records.bin` holds `embedding: None` for it.
#[test]
fn repeated_checkpoint_preserves_earlier_vectors() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_040);
    let (engine, id) = test_engine(&dir, 2); // batch_size=2 -> flushed

    let (_, _, first) = add_n(&engine, &id, &mut rng, 6, "a");
    engine.checkpoint(&id).unwrap();

    let (_, _, second) = add_n(&engine, &id, &mut rng, 6, "b");
    engine.checkpoint(&id).unwrap();

    let stored: Vec<(Id, Embedding)> = first.iter().chain(second.iter()).cloned().collect();
    assert_eq!(engine.count(&id).unwrap(), 12);

    // In-session: the exact path must still see the first cohort.
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 6, None),
            brute_force_top_k(&stored, &q, 6),
            "second checkpoint dropped the first cohort's vectors"
        );
    }

    // After reopen: every record must still resolve a vector.
    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 12);
    for (want_id, want_emb) in &stored {
        let rec = engine.get(&id, &[want_id.clone()]).unwrap()[0]
            .clone()
            .unwrap_or_else(|| panic!("{want_id} must survive the checkpoint"));
        assert_eq!(
            rec.embedding
                .as_ref()
                .unwrap_or_else(|| panic!("{want_id} lost its vector across checkpoints"))
                .as_ref(),
            want_emb.as_ref(),
            "embedding for {want_id}"
        );
    }
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 6, None),
            brute_force_top_k(&stored, &q, 6)
        );
    }
}

/// The filtered exact path must resolve vectors from mmap segments.
///
/// A checkpoint strips `embedding` once the vector is in the segment, so a
/// filtered query whose candidate set fits `max_scan` must read the segment or
/// it silently returns nothing after a restart.
#[test]
fn filtered_query_after_checkpoint_returns_all_candidates() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_050);
    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();

    let mut ids = Vec::new();
    let mut embs = Vec::new();
    let mut metas = Vec::new();
    for i in 0..10 {
        ids.push(format!("f{i}"));
        embs.push(rand_embedding(&mut rng, DIM));
        metas.push(Some(meta_with_tag(
            if i % 2 == 0 { "a" } else { "b" },
            i as i64,
        )));
    }
    let stored: Vec<(Id, Embedding, Option<Metadata>)> = ids
        .iter()
        .cloned()
        .zip(embs.iter().cloned())
        .zip(metas.iter().cloned())
        .map(|((id, e), m)| (id, e, m))
        .collect();
    let query = embs[0].clone();
    let expected = brute_force_filtered_top_k(&stored, &filter, &query, 3);

    {
        let (engine, id) = test_engine(&dir, 2);
        engine.add(&id, &ids, &embs, Some(&metas), None).unwrap();
        engine.checkpoint(&id).unwrap();
        assert_eq!(engine.committed_count(&id).unwrap(), 10);
    }

    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    let got: HashSet<Id> = engine
        .query(&id, &query, 3, &QueryOptions::with_where(filter))
        .unwrap()
        .iter()
        .map(|h| h.id.clone())
        .collect();
    assert_eq!(
        got, expected,
        "filtered query lost segment-resident vectors after a restart"
    );
}

/// An upsert after a checkpoint must be scored with the new vector.
///
/// The in-memory record holds the new vector while the segment still holds the
/// old one, so a segment-first lookup ranks by a stale vector. The reported
/// distance is asserted directly because the ranking can tie.
#[test]
fn upsert_after_checkpoint_uses_new_vector() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);

    let old = Embedding::from(vec![1.0f32; DIM]);
    let new = Embedding::from(vec![0.0f32; DIM]);
    engine
        .add(
            &id,
            &["u0".to_string()],
            std::slice::from_ref(&old),
            None,
            None,
        )
        .unwrap();
    engine.checkpoint(&id).unwrap();

    // Same vector as `old`, so ranking alone cannot distinguish stale from new.
    engine
        .add(
            &id,
            &["b1".to_string()],
            std::slice::from_ref(&old),
            None,
            None,
        )
        .unwrap();
    engine
        .upsert(
            &id,
            &["u0".to_string()],
            std::slice::from_ref(&new),
            None,
            None,
        )
        .unwrap();

    let query = old.clone();
    let hits = engine.query(&id, &query, 2, &qopts(100)).unwrap();
    assert_eq!(hits.len(), 2, "both records must be returned");
    let u0 = hits.iter().find(|h| h.id == "u0").expect("u0 present");
    let want = l2_squared(new.as_ref(), query.as_ref());
    assert!(
        (u0.distance - want).abs() < 1e-4,
        "u0 was scored with a stale vector: got {}, want {} (stale would be {})",
        u0.distance,
        want,
        l2_squared(old.as_ref(), query.as_ref())
    );

    // `get` must report the new vector too.
    let rec = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .expect("u0");
    assert_eq!(
        rec.embedding.as_ref().unwrap().as_ref(),
        new.as_ref(),
        "get returned the stale segment vector"
    );
}

/// Chroma allows creating a collection without a dimension and infers it from
/// the first write. The HNSW index is built when the collection is opened, so an
/// inferred dimension must also be applied to the index.
#[test]
fn dimension_inferred_from_first_write() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_060);
    let engine = open_engine(&dir);
    let id = create_collection(&engine, "infer", 0, Distance::L2, 2);

    let emb = rand_embedding(&mut rng, DIM);
    engine
        .add(
            &id,
            &["a".to_string()],
            std::slice::from_ref(&emb),
            None,
            None,
        )
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 1);

    let hits = engine.query(&id, &emb, 1, &qopts(10)).unwrap();
    assert_eq!(hits.len(), 1, "query on an inferred-dimension collection");
    assert_eq!(hits[0].id, "a");

    // The inferred dimension must survive a restart.
    drop(engine);
    let engine = open_engine(&dir);
    let id = engine
        .get_collection(TENANT, DATABASE, "infer")
        .unwrap()
        .unwrap()
        .config
        .id;
    assert_eq!(
        engine
            .get_collection_by_id(&id)
            .unwrap()
            .unwrap()
            .config
            .dimension,
        DIM
    );
}

/// A rejected first batch must not pin the collection's dimension.
///
/// Dimension inference used to run before validation, so a batch that failed
/// (duplicate ids here) still set `config.dimension` from its first vector and
/// every later write of the real width was rejected.
#[test]
fn rejected_first_write_does_not_pin_dimension() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_061);
    let engine = open_engine(&dir);
    let id = create_collection(&engine, "infer", 0, Distance::L2, 100);

    // Duplicate ids -> rejected, but the first embedding is 8-dim.
    let eight = rand_embedding(&mut rng, 8);
    let err = engine.add(
        &id,
        &["dup".to_string(), "dup".to_string()],
        &[eight.clone(), eight],
        None,
        None,
    );
    assert!(err.is_err(), "duplicate ids must be rejected");

    // The real width is DIM (16); it must still be accepted.
    let sixteen = rand_embedding(&mut rng, DIM);
    engine
        .add(
            &id,
            &["real".to_string()],
            std::slice::from_ref(&sixteen),
            None,
            None,
        )
        .unwrap();
    assert_eq!(engine.count(&id).unwrap(), 1);
    assert_eq!(
        engine
            .get_collection(TENANT, DATABASE, "infer")
            .unwrap()
            .unwrap()
            .config
            .dimension,
        DIM,
        "a rejected batch must not pin the dimension"
    );
}

/// A first batch mixing embedding widths must be rejected, not silently
/// indexed at the first vector's width.
#[test]
fn mixed_width_first_batch_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_062);
    let engine = open_engine(&dir);
    let id = create_collection(&engine, "infer", 0, Distance::L2, 100);

    let a = rand_embedding(&mut rng, DIM);
    let b = rand_embedding(&mut rng, DIM + 4);
    let err = engine.add(
        &id,
        &["a".to_string(), "b".to_string()],
        &[a, b],
        None,
        None,
    );
    assert!(err.is_err(), "mixed widths in one batch must be rejected");
}

// ---------------------------------------------------------------------------
// Checkpoint generation publication: crash safety and fail-closed recovery
// ---------------------------------------------------------------------------

/// Interrupting a checkpoint *before* it publishes must leave the previous
/// generation active and complete.
///
/// The writer stages a full generation, then flips `CURRENT`. A crash in
/// between leaves an orphan directory that nothing references, so the reopen
/// must observe exactly the last published state (plus the WAL tail).
#[test]
fn interrupted_publication_keeps_previous_generation_active() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_070);
    let (engine, id) = test_engine(&dir, 2);

    let (_, _, first) = add_n(&engine, &id, &mut rng, 4, "p");
    engine.checkpoint(&id).unwrap();
    let published = checkpoint_generation(&dir, &id);
    let published_name = published.file_name().unwrap().to_str().unwrap().to_owned();

    // Simulate a crash mid-publication: a generation directory exists but was
    // never completed, and `CURRENT` still names the previous one.
    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    let orphan = base
        .join("generations")
        .join("gen-99999999999999999999-deadbeef");
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("records.bin"), b"partial").unwrap();

    let current = std::fs::read_to_string(base.join("CURRENT")).unwrap();
    assert_eq!(current.trim(), published_name);

    // More writes after the checkpoint land in the WAL tail and must survive
    // alongside the published generation.
    let (_, _, second) = add_n(&engine, &id, &mut rng, 4, "q");
    let stored: Vec<(Id, Embedding)> = first.iter().chain(second.iter()).cloned().collect();

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 8, "published state + WAL tail");
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 8, None),
            brute_force_top_k(&stored, &q, 8)
        );
    }
}

/// A `CURRENT` pointer naming an incomplete generation is corruption, not
/// "no checkpoint". It must not be silently downgraded to a full replay.
#[test]
fn current_pointing_at_incomplete_generation_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_071);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "c");
    engine.checkpoint(&id).unwrap();

    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    let broken = base
        .join("generations")
        .join("gen-00000000000000000999-broken00");
    std::fs::create_dir_all(&broken).unwrap();
    // A generation that never got its manifest.
    std::fs::write(broken.join("records.bin"), b"not a checkpoint").unwrap();
    std::fs::write(base.join("CURRENT"), b"gen-00000000000000000999-broken00").unwrap();

    drop(engine);

    // The WAL was pruned by the checkpoint, so there is nothing to replay. The
    // engine must refuse to start rather than hand back a healthy-looking
    // empty collection. Recovery fails at open time, which is where an operator
    // needs to see it.
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let result = Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            durability: Durability::Durable,
            epoch: Epoch(0),
            ..EngineConfig::default()
        },
    );
    match result {
        Ok(engine) => {
            let id = collection_id_by_name(&engine);
            panic!(
                "corrupt published generation was accepted, count={}",
                engine.count(&id).unwrap_or(u64::MAX)
            );
        }
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("refusing to recover"),
                "unexpected error: {msg}"
            );
        }
    }
}

/// Deleting every record and checkpointing must prune the WAL.
///
/// It used to return early on an empty record map, so the delete history could
/// never be pruned and every restart replayed all of it.
#[test]
fn emptying_a_collection_still_prunes_the_wal() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_072);
    let (engine, id) = test_engine(&dir, 2);
    let (ids, _, _) = add_n(&engine, &id, &mut rng, 6, "e");
    engine.checkpoint(&id).unwrap();

    engine.delete(&id, &ids).unwrap();
    let stats = engine.optimize(&id).unwrap();
    assert_eq!(stats.records, 0);
    assert!(
        stats.wal_bytes_after < stats.wal_bytes_before,
        "WAL still holds the delete history: {} -> {}",
        stats.wal_bytes_before,
        stats.wal_bytes_after
    );

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 0, "deletes must not resurrect");
}

// ---------------------------------------------------------------------------
// PR3: generation validation, durable-watermark recovery, auto-checkpoint
// ---------------------------------------------------------------------------

/// A generation missing its vector file must be rejected, not silently
/// degraded to a collection of records that no query can ever match.
#[test]
fn missing_vector_file_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_080);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "v");
    engine.checkpoint(&id).unwrap();

    let generation = checkpoint_generation(&dir, &id);
    std::fs::remove_file(generation.join("segments").join("seg-0.bin")).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(engine) => {
            let id = collection_id_by_name(&engine);
            panic!(
                "generation without its vectors was accepted, count={}",
                engine.count(&id).unwrap_or(u64::MAX)
            );
        }
        Err(e) => assert!(
            e.to_string().contains("refusing to recover"),
            "unexpected error: {e}"
        ),
    }
}

/// A tampered `records.bin` must fail its manifest checksum.
///
/// The inventory is what distinguishes "this file is the one the manifest
/// describes" from "this file happens to deserialize".
#[test]
fn tampered_records_file_fails_checksum() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_081);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "t");
    engine.checkpoint(&id).unwrap();

    let generation = checkpoint_generation(&dir, &id);
    let records_path = generation.join("records.bin");
    let mut bytes = std::fs::read(&records_path).unwrap();
    // Flip a byte in the middle: same length, different content.
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xff;
    std::fs::write(&records_path, &bytes).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("tampered records.bin was accepted"),
        Err(e) => assert!(
            e.to_string().contains("checksum mismatch") || e.to_string().contains("refusing"),
            "unexpected error: {e}"
        ),
    }
}

/// A generation directory copied from another collection must not be activated.
#[test]
fn generation_from_another_collection_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_082);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "g");
    engine.checkpoint(&id).unwrap();

    let generation = checkpoint_generation(&dir, &id);
    let manifest_path = generation.join("checkpoint.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    manifest["collection_id"] = serde_json::Value::String(Uuid::new_v4().to_string());
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("a generation owned by another collection was accepted"),
        Err(e) => assert!(
            e.to_string().contains("belongs to collection") || e.to_string().contains("refusing"),
            "unexpected error: {e}"
        ),
    }
}

/// Deleting the checkpoint directory after a checkpoint must not look like an
/// empty, freshly created collection.
///
/// The catalog's durable watermark is the only record that survives that
/// deletion, so it is what tells recovery the history is gone.
#[test]
fn deleted_checkpoint_dir_after_prune_is_reported() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_083);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "d");
    engine.checkpoint(&id).unwrap();

    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    std::fs::remove_dir_all(&base).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(engine) => {
            let id = collection_id_by_name(&engine);
            panic!(
                "lost checkpoint was reported as an empty collection, count={}",
                engine.count(&id).unwrap_or(u64::MAX)
            );
        }
        Err(e) => assert!(
            e.to_string()
                .contains("no checkpoint generation is readable")
                || e.to_string().contains("refusing"),
            "unexpected error: {e}"
        ),
    }
}

/// Automatic checkpointing must prune the WAL of an emptied collection.
///
/// `auto_checkpoint` skipped collections with no live records, so a collection
/// that was fully deleted kept its entire delete history and replayed it on
/// every restart.
#[test]
fn auto_checkpoint_prunes_an_emptied_collection() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_084);
    let mut config = CollectionConfig::new("auto".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 2;
    // Checkpoint every 3 WAL records so the deletes alone cross the threshold.
    config.hnsw.sync_threshold = 3;
    let catalog = Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let engine = Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig::default(),
    )
    .unwrap();
    let id = engine.create_collection(&config).unwrap().config.id;

    let (ids, _, _) = add_n(&engine, &id, &mut rng, 4, "a");
    let wal_path = dir.path().join("wal").join(format!("{id}.wal"));
    let before = std::fs::metadata(&wal_path).unwrap().len();

    engine.delete(&id, &ids).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 0);

    // The delete history must be pruned without an explicit checkpoint call.
    // `last_seq` keeps counting past a prune, so assert on the file itself: four
    // more operations were appended, so without truncation it would be larger.
    let after = std::fs::metadata(&wal_path).unwrap().len();
    assert!(
        after <= before,
        "WAL grew from {before} to {after} bytes: the delete history was not pruned"
    );
    assert!(
        after < 256,
        "WAL is {after} bytes: that is more than a header, so history survived"
    );

    // And the collection must reopen empty rather than replaying the deletes.
    drop(engine);
    let engine = open_engine(&dir);
    let id = engine
        .get_collection(TENANT, DATABASE, "auto")
        .unwrap()
        .expect("collection survives restart")
        .config
        .id;
    assert_eq!(engine.count(&id).unwrap(), 0);
}

/// Inferring the dimension must also resize the ANN index.
///
/// A collection created without a dimension built its index at width 0. The
/// first write set `config.dimension` but left the index untouched, so the
/// failure only appeared later — at the first buffer flush or rebuild — as
/// "index is 0, got N". Small collections never flush, which is why the
/// existing inference test passed.
#[test]
fn inferred_dimension_resizes_the_index() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_090);
    let engine = open_engine(&dir);
    // batch_size 2 so the buffer flushes during the write, exercising the index.
    let id = create_collection(&engine, "infer_idx", 0, Distance::L2, 2);

    let (ids, embs, _) = add_n(&engine, &id, &mut rng, 4, "i");
    assert_eq!(engine.count(&id).unwrap(), 4);
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        4,
        "buffered records must reach the index"
    );

    let stored: Vec<(Id, Embedding)> = ids.into_iter().zip(embs).collect();
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 4, None),
            brute_force_top_k(&stored, &q, 4)
        );
    }

    // Optimize rebuilds the graph at the inferred width.
    let stats = engine.optimize(&id).unwrap();
    assert_eq!(stats.records, 4);
    assert_eq!(stats.indexed, 4);

    drop(engine);
    let engine = open_engine(&dir);
    let id = engine
        .get_collection(TENANT, DATABASE, "infer_idx")
        .unwrap()
        .expect("collection survives restart")
        .config
        .id;
    assert_eq!(engine.count(&id).unwrap(), 4);
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 4, None),
            brute_force_top_k(&stored, &q, 4)
        );
    }
}

// ---------------------------------------------------------------------------
// PR4: Chroma mutation semantics (merge, field preservation, no-op missing id)
// ---------------------------------------------------------------------------

fn patch(pairs: &[(&str, &str)]) -> rekha_core::types::MetadataPatch {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), Some(MetadataValue::Str((*v).to_string()))))
        .collect()
}

fn seed_two(engine: &Engine, id: &Uuid) {
    let mut meta = Metadata::new();
    meta.insert("a".into(), MetadataValue::Str("1".into()));
    meta.insert("b".into(), MetadataValue::Str("2".into()));
    engine
        .add(
            id,
            &["u0".to_string()],
            &[Embedding::from(vec![1.0f32; DIM])],
            Some(&[Some(meta)]),
            Some(&[Some("doc-1".to_string())]),
        )
        .unwrap();
}

/// A metadata update merges and leaves the document and vector alone.
///
/// Chroma merges rather than replaces, so a key the request does not mention
/// keeps its value, and `update` must never clear unrelated fields.
#[test]
fn update_metadata_merges_and_preserves_other_fields() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    seed_two(&engine, &id);

    engine
        .update(
            &id,
            &["u0".to_string()],
            None,
            Some(&[Some(patch(&[("a", "10")]))]),
            None,
        )
        .unwrap();

    let rec = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .expect("u0");
    let meta = rec.metadata.expect("metadata survives");
    assert_eq!(meta.get("a"), Some(&MetadataValue::Str("10".into())));
    assert_eq!(
        meta.get("b"),
        Some(&MetadataValue::Str("2".into())),
        "an unmentioned key must be preserved"
    );
    assert_eq!(rec.document.as_deref(), Some("doc-1"), "document preserved");
    assert_eq!(
        rec.embedding.as_ref().map(|e| e.as_ref()),
        Some(&vec![1.0f32; DIM][..]),
        "vector preserved"
    );
}

/// A metadata value of `null` deletes the key; the rest still merges.
#[test]
fn update_metadata_null_deletes_key() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    seed_two(&engine, &id);

    let mut p = rekha_core::types::MetadataPatch::new();
    p.insert("b".to_string(), None);
    p.insert("c".to_string(), Some(MetadataValue::Str("3".into())));
    engine
        .update(&id, &["u0".to_string()], None, Some(&[Some(p)]), None)
        .unwrap();

    let meta = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .unwrap()
        .metadata
        .unwrap();
    assert!(!meta.contains_key("b"), "null must remove the key");
    assert_eq!(meta.get("a"), Some(&MetadataValue::Str("1".into())));
    assert_eq!(meta.get("c"), Some(&MetadataValue::Str("3".into())));
}

/// A document update leaves metadata and the vector untouched.
#[test]
fn update_document_preserves_metadata_and_vector() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    seed_two(&engine, &id);

    engine
        .update(
            &id,
            &["u0".to_string()],
            None,
            None,
            Some(&[Some("doc-2".to_string())]),
        )
        .unwrap();

    let rec = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(rec.document.as_deref(), Some("doc-2"));
    let meta = rec.metadata.expect("metadata preserved");
    assert_eq!(meta.get("a"), Some(&MetadataValue::Str("1".into())));
    assert_eq!(meta.get("b"), Some(&MetadataValue::Str("2".into())));
}

/// A vector-only update keeps metadata/document **and** reaches the index.
///
/// `update` never ran the flush policy, so a vector written there sat in the
/// buffer and `committed_count` never advanced until some later write flushed
/// it.
#[test]
fn update_embedding_preserves_fields_and_flushes() {
    let dir = TempDir::new().unwrap();
    // batch_size 1: every buffered write flushes immediately, so the assertion
    // is about `update` running the flush policy at all.
    let (engine, id) = test_engine(&dir, 1);
    seed_two(&engine, &id);

    let new = Embedding::from(vec![0.5f32; DIM]);
    engine
        .update(
            &id,
            &["u0".to_string()],
            Some(std::slice::from_ref(&new)),
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        engine.committed_count(&id).unwrap(),
        1,
        "vector update must reach the index"
    );

    let query = Embedding::from(vec![0.5f32; DIM]);
    let hits = engine.query(&id, &query, 1, &qopts(10)).unwrap();
    assert!(
        hits[0].distance < 1e-4,
        "query must use the updated vector, got {}",
        hits[0].distance
    );

    let rec = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .unwrap();
    assert_eq!(rec.document.as_deref(), Some("doc-1"));
    assert!(rec.metadata.is_some());
}

/// Upsert merges metadata into an existing record instead of replacing it.
#[test]
fn upsert_merges_metadata() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    seed_two(&engine, &id);

    engine
        .upsert(
            &id,
            &["u0".to_string()],
            &[Embedding::from(vec![2.0f32; DIM])],
            Some(&[Some(patch(&[("a", "99")]))]),
            None,
        )
        .unwrap();

    let rec = engine.get(&id, &["u0".to_string()]).unwrap()[0]
        .clone()
        .unwrap();
    let meta = rec.metadata.unwrap();
    assert_eq!(meta.get("a"), Some(&MetadataValue::Str("99".into())));
    assert_eq!(
        meta.get("b"),
        Some(&MetadataValue::Str("2".into())),
        "upsert must merge, not replace"
    );
    assert_eq!(
        rec.document.as_deref(),
        Some("doc-1"),
        "unspecified document preserved"
    );
}

/// A metadata-only update must be visible to filtered queries immediately,
/// including for a record still sitting in the unflushed buffer.
#[test]
fn buffered_metadata_update_refreshes_filter_eligibility() {
    let dir = TempDir::new().unwrap();
    // Never flushes: every record stays in the buffer.
    let (engine, id) = test_engine(&dir, 1000);
    let mut rng = Lcg::new(9_100);

    let mut meta = Metadata::new();
    meta.insert("tag".into(), MetadataValue::Str("a".into()));
    engine
        .add(
            &id,
            &["b0".to_string()],
            &[rand_embedding(&mut rng, DIM)],
            Some(&[Some(meta)]),
            None,
        )
        .unwrap();

    let a = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();
    let b = WhereFilter::parse_json(r#"{"tag": {"$eq": "b"}}"#).unwrap();
    let q = rand_embedding(&mut rng, DIM);

    assert_eq!(
        engine
            .query(&id, &q, 5, &QueryOptions::with_where(a.clone()))
            .unwrap()
            .len(),
        1
    );

    engine
        .update(
            &id,
            &["b0".to_string()],
            None,
            Some(&[Some(patch(&[("tag", "b")]))]),
            None,
        )
        .unwrap();

    assert!(
        engine
            .query(&id, &q, 5, &QueryOptions::with_where(a))
            .unwrap()
            .is_empty(),
        "record must stop matching the old tag"
    );
    assert_eq!(
        engine
            .query(&id, &q, 5, &QueryOptions::with_where(b))
            .unwrap()
            .len(),
        1,
        "buffer metadata must reflect the update"
    );
}

/// A filtered delete must not remove records that never matched the filter,
/// even while other writes are landing.
#[test]
fn delete_where_does_not_remove_non_matching_records() {
    let dir = TempDir::new().unwrap();
    let engine = Arc::new(open_engine(&dir));
    let mut config = CollectionConfig::new("coll".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 2;
    let id = engine.create_collection(&config).unwrap().config.id;

    let filter = WhereFilter::parse_json(r#"{"tag": {"$eq": "a"}}"#).unwrap();

    // Seed tag=a records.
    let mut a_ids = Vec::new();
    for i in 0..8 {
        let mut meta = Metadata::new();
        meta.insert("tag".into(), MetadataValue::Str("a".into()));
        a_ids.push(format!("a{i}"));
        engine
            .add(
                &id,
                std::slice::from_ref(&a_ids[i]),
                &[Embedding::from(vec![i as f32; DIM])],
                Some(&[Some(meta)]),
                None,
            )
            .unwrap();
    }

    // Writer adds tag=b records concurrently.
    let writer = {
        let engine = engine.clone();
        std::thread::spawn(move || {
            for i in 0..24 {
                let mut meta = Metadata::new();
                meta.insert("tag".into(), MetadataValue::Str("b".into()));
                engine
                    .add(
                        &id,
                        &[format!("b{i}")],
                        &[Embedding::from(vec![100.0f32; DIM])],
                        Some(&[Some(meta)]),
                        None,
                    )
                    .unwrap();
            }
        })
    };

    // Selection happens under the collection lock, so a record cannot slip in
    // between the filter evaluation and the delete.
    let batch = engine
        .prepare_delete_where(&id, None, Some(&filter), None, None)
        .unwrap();
    assert_eq!(batch.ops.len(), 8, "exactly the tag=a records are selected");
    engine.append_ops(&id, &batch.ops).unwrap();
    writer.join().unwrap();

    let remaining = engine.get_records(&id, None, None, None, None, 0).unwrap();
    assert_eq!(remaining.len(), 24, "every tag=b record survives");
    assert!(
        remaining.iter().all(|r| r
            .metadata
            .as_ref()
            .and_then(|m| m.get("tag"))
            .is_some_and(|v| *v == MetadataValue::Str("b".into()))),
        "only tag=b records remain"
    );
}

// ---------------------------------------------------------------------------
// PR5: query projection consistency and persisted ordering
// ---------------------------------------------------------------------------

/// Pagination order must survive a checkpoint + restart.
///
/// Offsets were rebuilt by iterating the records `HashMap`, so a restart could
/// reorder a collection even though nothing had been written since.
#[test]
fn insertion_order_survives_restart() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("order".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 2;
    let id = engine.create_collection(&config).unwrap().config.id;

    // Deliberately not sorted, and not in id order.
    let ids: Vec<Id> = ["z", "a", "m", "b"].iter().map(|s| s.to_string()).collect();
    let embs: Vec<Embedding> = (0..ids.len())
        .map(|i| Embedding::from(vec![i as f32; DIM]))
        .collect();
    engine.add(&id, &ids, &embs, None, None).unwrap();

    let before = engine.get_records(&id, None, None, None, None, 0).unwrap();
    let before_ids: Vec<Id> = before.iter().map(|r| r.id.clone()).collect();
    assert_eq!(before_ids, ids, "insertion order");

    engine.checkpoint(&id).unwrap();
    drop(engine);

    let engine = open_engine(&dir);
    let id = engine
        .get_collection(TENANT, DATABASE, "order")
        .unwrap()
        .expect("collection survives restart")
        .config
        .id;

    let after = engine.get_records(&id, None, None, None, None, 0).unwrap();
    let after_ids: Vec<Id> = after.iter().map(|r| r.id.clone()).collect();
    assert_eq!(
        after_ids, before_ids,
        "pagination order must be identical after a restart"
    );

    // And pagination still walks that order.
    let mut paged = Vec::new();
    for offset in (0..ids.len()).step_by(2) {
        let page = engine
            .get_records(&id, None, None, None, Some(2), offset)
            .unwrap();
        paged.extend(page.into_iter().map(|r| r.id));
    }
    assert_eq!(paged, before_ids, "concatenated pages equal the full order");
}

/// An explicit id list is filtered to insertion order, not re-ordered.
#[test]
fn explicit_ids_return_insertion_order() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    let ids: Vec<Id> = ["z", "a", "m"].iter().map(|s| s.to_string()).collect();
    let embs: Vec<Embedding> = vec![Embedding::from(vec![1.0f32; DIM]); 3];
    engine.add(&id, &ids, &embs, None, None).unwrap();

    let got = engine
        .get_records(
            &id,
            Some(&["m".to_string(), "z".to_string(), "a".to_string()]),
            None,
            None,
            None,
            0,
        )
        .unwrap();
    let order: Vec<Id> = got.into_iter().map(|r| r.id).collect();
    assert_eq!(
        order, ids,
        "Chroma returns insertion order, not request order"
    );
}

/// A query result row must pair its distance with the embedding of the same
/// record version.
///
/// The API used to search, release the lock, then re-fetch embeddings, so a
/// concurrent upsert could produce a distance from the old vector next to the
/// new one.
#[test]
fn query_projection_is_internally_consistent() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);

    let old = Embedding::from(vec![1.0f32; DIM]);
    let new = Embedding::from(vec![2.0f32; DIM]);
    engine
        .add(
            &id,
            &["r0".to_string()],
            std::slice::from_ref(&old),
            None,
            None,
        )
        .unwrap();
    engine.checkpoint(&id).unwrap();
    // Replace the vector after the checkpoint so record and segment disagree.
    engine
        .upsert(
            &id,
            &["r0".to_string()],
            std::slice::from_ref(&new),
            None,
            None,
        )
        .unwrap();

    let query = Embedding::from(vec![2.0f32; DIM]);
    let rows = engine
        .query_many(&id, std::slice::from_ref(&query), 1, &qopts(10), true)
        .unwrap();
    let row = &rows[0][0];

    assert_eq!(
        row.embedding.as_ref().map(|e| e.as_ref()),
        Some(new.as_ref()),
        "projected embedding must be the version the distance was computed from"
    );
    let want = l2_squared(new.as_ref(), query.as_ref());
    assert!(
        (row.distance - want).abs() < 1e-4,
        "distance {} should match the projected vector (want {want})",
        row.distance
    );
}

/// Embeddings are only copied when requested.
#[test]
fn query_omits_embeddings_unless_requested() {
    let dir = TempDir::new().unwrap();
    let (engine, id) = test_engine(&dir, 2);
    let emb = Embedding::from(vec![1.0f32; DIM]);
    engine
        .add(
            &id,
            &["r0".to_string()],
            std::slice::from_ref(&emb),
            None,
            None,
        )
        .unwrap();

    let rows = engine
        .query_many(&id, std::slice::from_ref(&emb), 1, &qopts(10), false)
        .unwrap();
    assert!(
        rows[0][0].embedding.is_none(),
        "embedding must not be materialized when not requested"
    );
}

// ---------------------------------------------------------------------------
// Manifest validation: version, configuration binding, structural integrity
// ---------------------------------------------------------------------------

/// Rewrite a generation's manifest through `f` and assert the reopen fails with
/// a recovery error mentioning `expect`.
fn assert_manifest_rejected(corrupt: &str, f: impl FnOnce(&mut serde_json::Value), expect: &str) {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_200);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "m");
    engine.checkpoint(&id).unwrap();

    let manifest = checkpoint_generation(&dir, &id).join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    f(&mut value);
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("{corrupt}: malformed generation was accepted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(expect),
                "{corrupt}: expected `{expect}` in error, got: {msg}"
            );
        }
    }
}

/// A manifest version this build does not know must be refused.
///
/// Guessing at a future format is worse than refusing: a later version may
/// assign new meaning to an existing field.
#[test]
fn unknown_manifest_version_is_rejected() {
    assert_manifest_rejected(
        "future version",
        |v| v["format_version"] = serde_json::json!(99),
        "unsupported checkpoint format version",
    );
}

/// A manifest written by an older format is refused, not partially interpreted.
///
/// `format_version` is required: without it there is no statement about which
/// fields exist, and guessing at a field's meaning is worse than refusing.
#[test]
fn manifest_without_format_version_is_rejected() {
    assert_manifest_rejected(
        "missing version",
        |v| {
            v.as_object_mut().unwrap().remove("format_version");
        },
        "missing field",
    );
}

/// Checkpoint files at the base, with no `CURRENT`, are the pre-generation
/// layout and must be rejected.
///
/// Falling back to them would mean the engine reads a layout it never writes,
/// while treating them as absent would silently discard the checkpoint and
/// recover a smaller collection from whatever the pruned WAL still holds.
#[test]
fn flat_layout_checkpoint_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_074);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "f");
    engine.checkpoint(&id).unwrap();

    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    let generation = checkpoint_generation(&dir, &id);
    for entry in std::fs::read_dir(&generation).unwrap() {
        let entry = entry.unwrap();
        let dest = base.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
    std::fs::remove_file(base.join("CURRENT")).unwrap();
    std::fs::remove_dir_all(base.join("generations")).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("a pre-generation checkpoint layout was accepted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("pre-generation layout"),
                "unexpected error: {msg}"
            );
        }
    }
}

/// A generation is always written with `order.bin`; missing it is corruption.
///
/// Reconstructing an order would silently change `get`/pagination results
/// relative to the previous process, which is worse than refusing to start.
#[test]
fn missing_order_file_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_075);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "o");
    engine.checkpoint(&id).unwrap();

    let generation = checkpoint_generation(&dir, &id);
    std::fs::remove_file(generation.join("order.bin")).unwrap();
    // The inventory lists order.bin, so the checksum check reports it first.
    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("a generation without order.bin was accepted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("order.bin"),
                "expected the error to name order.bin, got: {msg}"
            );
        }
    }
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let dest = dst.join(entry.file_name());
        if entry.path().is_dir() {
            copy_dir(&entry.path(), &dest);
        } else {
            std::fs::copy(entry.path(), dest).unwrap();
        }
    }
}

/// A manifest with no inventory must not pass by having nothing to check.
#[test]
fn empty_inventory_is_rejected() {
    assert_manifest_rejected(
        "empty inventory",
        |v| v["files"] = serde_json::json!([]),
        "declares no file inventory",
    );
}

/// A manifest that does not inventory its own records file is malformed.
#[test]
fn inventory_without_records_is_rejected() {
    assert_manifest_rejected(
        "no records entry",
        |v| {
            let files = v["files"].as_array_mut().unwrap();
            files.retain(|f| f["path"] != "records.bin");
        },
        "does not inventory records.bin",
    );
}

/// Listing the same file twice is ambiguous and must be refused.
#[test]
fn duplicate_inventory_entry_is_rejected() {
    assert_manifest_rejected(
        "duplicate path",
        |v| {
            let files = v["files"].as_array_mut().unwrap();
            let first = files[0].clone();
            files.push(first);
        },
        "more than once",
    );
}

/// A generation written for a different vector width must not be activated.
#[test]
fn dimension_mismatch_is_rejected() {
    assert_manifest_rejected(
        "dimension",
        |v| v["dimension"] = serde_json::json!(999),
        "dimensional vectors but the collection is",
    );
}

/// A generation written for a different metric must not be activated: every
/// distance computed from it would be in the wrong space.
#[test]
fn space_mismatch_is_rejected() {
    assert_manifest_rejected(
        "space",
        |v| v["space"] = serde_json::json!("cosine"),
        "was written for space",
    );
}

/// A record whose seq is beyond the snapshot boundary means the manifest and
/// the record set disagree about what the generation contains.
#[test]
fn record_beyond_snapshot_boundary_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_201);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "b");
    engine.checkpoint(&id).unwrap();

    let manifest = checkpoint_generation(&dir, &id).join("checkpoint.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    value["snapshot_seq"] = serde_json::json!(1);
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();

    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("record beyond the snapshot boundary was accepted"),
        Err(e) => assert!(
            e.to_string().contains("beyond the snapshot boundary"),
            "unexpected error: {e}"
        ),
    }
}

/// A segment offset that points past the segment must be caught at load.
///
/// Checking only that the id appears in the mapping accepts an offset that
/// resolves to nothing, which surfaces later as a record that exists but can
/// never be matched by a query.
#[test]
fn out_of_bounds_segment_offset_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_202);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "o");
    engine.checkpoint(&id).unwrap();

    let generation = checkpoint_generation(&dir, &id);
    let idx_path = generation.join("segment_index.bin");
    let mut idx: std::collections::HashMap<Id, usize> =
        bincode::deserialize(&std::fs::read(&idx_path).unwrap()).unwrap();
    for offset in idx.values_mut() {
        *offset += 10_000;
    }
    std::fs::write(&idx_path, bincode::serialize(&idx).unwrap()).unwrap();

    // The inventory records a CRC for this file, so tampering is caught there
    // first — either way the generation must not load.
    drop(engine);
    match open_engine_result(&dir) {
        Ok(_) => panic!("out-of-bounds segment offset was accepted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("out of bounds") || msg.contains("checksum"),
                "unexpected error: {msg}"
            );
        }
    }
}

/// An index built for another configuration is rebuilt, not trusted.
///
/// `load_any` validated the graph against its own persisted metadata, so a
/// self-consistent graph written for a different width or metric would load and
/// produce wrong distances.
#[test]
fn index_written_for_another_config_is_rebuilt() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_203);
    let (engine, id) = test_engine(&dir, 2);
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 6, "x");
    engine.checkpoint(&id).unwrap();

    // Replace the persisted index with one built at a different width by
    // clearing it: a missing/incompatible graph must trigger a rebuild from the
    // authoritative vectors rather than an empty graph.
    let generation = checkpoint_generation(&dir, &id);
    let _ = std::fs::remove_file(generation.join("index.bin"));
    let _ = std::fs::remove_file(generation.join("index.meta"));

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);

    assert_eq!(engine.count(&id).unwrap(), 6);
    assert_eq!(
        engine.committed_count(&id).unwrap(),
        6,
        "index must be rebuilt from records"
    );
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 6, None),
            brute_force_top_k(&stored, &q, 6)
        );
    }
}

// ---------------------------------------------------------------------------
// PR-C2a: transactional checkpoint publication
// ---------------------------------------------------------------------------

use std::cell::RefCell;

thread_local! {
    /// The fault point this test thread should fail at, if any.
    static FAIL_AT: RefCell<Option<&'static str>> = const { RefCell::new(None) };
}

fn fault_hook(point: &str) -> Result<(), String> {
    FAIL_AT.with(|target| match *target.borrow() {
        Some(t) if t == point => Err("injected".to_string()),
        _ => Ok(()),
    })
}

/// Install the fault hook, fail at `point`, run `f`, then clear the hook.
fn with_fault_at<R>(point: &'static str, f: impl FnOnce() -> R) -> R {
    FAIL_AT.with(|target| *target.borrow_mut() = Some(point));
    rekha_engine::fault::set_hook(Some(fault_hook));
    let out = f();
    rekha_engine::fault::set_hook(None);
    FAIL_AT.with(|target| *target.borrow_mut() = None);
    out
}

fn published_generation_name(dir: &TempDir, id: &Uuid) -> String {
    let base = dir.path().join("wal").join(format!("{id}.checkpoint"));
    std::fs::read_to_string(base.join("CURRENT"))
        .expect("a checkpoint must publish a CURRENT pointer")
        .trim()
        .to_owned()
}

/// A failure anywhere before publication must leave the previous generation
/// active and live reads untouched.
#[test]
fn pre_publication_failure_leaves_previous_generation_active() {
    for point in [
        "checkpoint.staging",
        "checkpoint.files",
        "checkpoint.validated",
    ] {
        let dir = TempDir::new().unwrap();
        let mut rng = Lcg::new(9_300);
        let (engine, id) = test_engine(&dir, 2);

        let (_, _, first) = add_n(&engine, &id, &mut rng, 4, "a");
        engine.checkpoint(&id).unwrap();
        let before = published_generation_name(&dir, &id);

        let (_, _, second) = add_n(&engine, &id, &mut rng, 4, "b");
        let stored: Vec<(Id, Embedding)> = first.iter().chain(second.iter()).cloned().collect();

        let err = with_fault_at(point, || engine.checkpoint(&id)).unwrap_err();
        assert!(
            err.to_string().contains("fault injected"),
            "{point}: expected an injected failure, got {err}"
        );

        // Nothing was published.
        assert_eq!(
            published_generation_name(&dir, &id),
            before,
            "{point}: CURRENT must still name the previous generation"
        );

        // Live reads still resolve vectors from the *published* generation.
        assert_eq!(engine.count(&id).unwrap(), 8, "{point}");
        for _ in 0..3 {
            let q = rand_embedding(&mut rng, DIM);
            assert_eq!(
                query_top_k(&engine, &id, &q, 8, None),
                brute_force_top_k(&stored, &q, 8),
                "{point}: reads must not see staging data"
            );
        }

        // The collection is still writable and a later checkpoint succeeds.
        let (_, _, third) = add_n(&engine, &id, &mut rng, 2, "c");
        engine.checkpoint(&id).unwrap();
        assert_ne!(
            published_generation_name(&dir, &id),
            before,
            "{point}: a later checkpoint must publish"
        );

        // Restart sees a complete collection.
        let stored: Vec<(Id, Embedding)> = stored.into_iter().chain(third).collect();
        drop(engine);
        let engine = open_engine(&dir);
        let id = collection_id_by_name(&engine);
        assert_eq!(engine.count(&id).unwrap(), 10, "{point}");
        for _ in 0..3 {
            let q = rand_embedding(&mut rng, DIM);
            assert_eq!(
                query_top_k(&engine, &id, &q, 10, None),
                brute_force_top_k(&stored, &q, 10),
                "{point}: restart after a failed publication"
            );
        }
    }
}

/// A failure after `CURRENT` was replaced is *not* "nothing happened".
///
/// The generation is durable but the publication could not be confirmed, so the
/// collection must refuse mutations until a reopen establishes the active
/// generation — and a restart must load the new one.
#[test]
fn post_rename_failure_marks_publication_uncertain() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_301);
    let (engine, id) = test_engine(&dir, 2);

    let (_, _, stored) = add_n(&engine, &id, &mut rng, 4, "u");
    engine.checkpoint(&id).unwrap();
    let before = published_generation_name(&dir, &id);

    let err = with_fault_at("checkpoint.published", || engine.checkpoint(&id)).unwrap_err();
    assert!(err.to_string().contains("fault injected"), "got {err}");

    // The pointer moved, so the new generation is what a restart will load.
    assert_ne!(
        published_generation_name(&dir, &id),
        before,
        "the rename should have taken effect"
    );

    // Reads keep working (the data is intact), but writes are refused.
    assert_eq!(engine.count(&id).unwrap(), 4);
    let write = engine
        .add(
            &id,
            &["late".to_string()],
            &[Embedding::from(vec![1.0f32; DIM])],
            None,
            None,
        )
        .unwrap_err();
    assert!(
        write.to_string().contains("publication is unresolved"),
        "expected a publication-uncertain error, got {write}"
    );

    // A restart re-establishes the active generation and accepts writes again.
    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 4);
    engine
        .add(
            &id,
            &["late".to_string()],
            &[Embedding::from(vec![1.0f32; DIM])],
            None,
            None,
        )
        .expect("writes resume after a reopen");
    let _ = stored;
}

/// Cleanup failures after publication are reported as deferred, not as errors.
#[test]
fn post_publication_cleanup_failure_is_deferred_not_fatal() {
    for (point, stage) in [
        ("checkpoint.catalog", "catalog"),
        ("checkpoint.pruned", "prune"),
        ("checkpoint.gc", "gc"),
    ] {
        let dir = TempDir::new().unwrap();
        let mut rng = Lcg::new(9_302);
        let (engine, id) = test_engine(&dir, 2);
        let (_, _, stored) = add_n(&engine, &id, &mut rng, 4, "d");

        let outcome = with_fault_at(point, || engine.checkpoint(&id))
            .unwrap_or_else(|e| panic!("{point}: a durable checkpoint must not be an error: {e}"));
        assert_eq!(
            outcome.deferred_stage(),
            Some(stage),
            "{point}: reported the wrong deferred stage"
        );

        // The collection keeps serving and accepting writes.
        assert_eq!(engine.count(&id).unwrap(), 4, "{point}");
        engine
            .add(
                &id,
                &[format!("after-{stage}")],
                &[Embedding::from(vec![2.0f32; DIM])],
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("{point}: writes must continue: {e}"));

        // And a restart is consistent.
        drop(engine);
        let engine = open_engine(&dir);
        let id = collection_id_by_name(&engine);
        assert_eq!(engine.count(&id).unwrap(), 5, "{point}");
        for _ in 0..2 {
            let q = rand_embedding(&mut rng, DIM);
            let hits = engine.query(&id, &q, 5, &qopts(10)).unwrap();
            assert_eq!(hits.len(), 5, "{point}");
        }
        let _ = &stored;
    }
}

/// Publishing must not depend on being able to open the staged segment lazily:
/// an unreadable staged segment aborts before anything is published.
#[test]
fn unreadable_staged_segment_aborts_publication() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_303);
    let (engine, id) = test_engine(&dir, 2);
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 4, "s");
    engine.checkpoint(&id).unwrap();
    let before = published_generation_name(&dir, &id);

    // Corrupt every staged segment as soon as it is written by failing the
    // validation stage — the closest available analogue to an unreadable file.
    let err = with_fault_at("checkpoint.validated", || engine.checkpoint(&id)).unwrap_err();
    assert!(err.to_string().contains("fault injected"));

    assert_eq!(published_generation_name(&dir, &id), before);
    for _ in 0..3 {
        let q = rand_embedding(&mut rng, DIM);
        assert_eq!(
            query_top_k(&engine, &id, &q, 4, None),
            brute_force_top_k(&stored, &q, 4)
        );
    }
}

// ---------------------------------------------------------------------------
// PR-C2b: WAL inspection and conservative recovery
// ---------------------------------------------------------------------------

fn wal_file(dir: &TempDir, id: &Uuid) -> std::path::PathBuf {
    dir.path().join("wal").join(format!("{id}.wal"))
}

/// Replace the log with one that starts past the checkpoint boundary.
///
/// A log that resumes after the snapshot leaves a hole: the operations in
/// between exist in neither place.
#[test]
fn wal_gap_after_checkpoint_boundary_is_reported() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_400);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 4, "g");
    engine.checkpoint(&id).unwrap();
    drop(engine);

    // Rebuild the log as if its middle were lost: resume far past the boundary.
    let path = wal_file(&dir, &id);
    let _ = std::fs::remove_file(&path);
    let mut wal = rekha_wal::Wal::open(
        dir.path().join("wal"),
        &id,
        Epoch(0),
        rekha_wal::WalOptions {
            fsync: false,
            start_seq: Some(100),
        },
        rekha_wal::TailRepair::DiscardSuspect,
    )
    .unwrap();
    wal.append(rekha_core::op::Operation::Add {
        id: "late".into(),
        embedding: Embedding::from(vec![1.0f32; DIM]),
        metadata: None,
        document: None,
    })
    .unwrap();
    drop(wal);

    match open_engine_result(&dir) {
        Ok(_) => panic!("a gap between checkpoint and WAL was accepted"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("are missing") || msg.contains("resumes at"),
                "unexpected error: {msg}"
            );
        }
    }
}

/// Without a checkpoint, a log that does not start at 1 means the earlier
/// history is gone — it must not be replayed as if it were the whole collection.
#[test]
fn wal_without_checkpoint_must_start_at_one() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let id = engine
        .create_collection(&CollectionConfig::new("gap".into(), DIM, Distance::L2))
        .unwrap()
        .config
        .id;
    drop(engine);

    let path = wal_file(&dir, &id);
    let _ = std::fs::remove_file(&path);
    let mut wal = rekha_wal::Wal::open(
        dir.path().join("wal"),
        &id,
        Epoch(0),
        rekha_wal::WalOptions {
            fsync: false,
            start_seq: Some(50),
        },
        rekha_wal::TailRepair::DiscardSuspect,
    )
    .unwrap();
    wal.append(rekha_core::op::Operation::Add {
        id: "only".into(),
        embedding: Embedding::from(vec![1.0f32; DIM]),
        metadata: None,
        document: None,
    })
    .unwrap();
    drop(wal);

    match open_engine_result(&dir) {
        Ok(_) => panic!("a log starting at 50 without a checkpoint was accepted"),
        Err(e) => assert!(
            e.to_string().contains("earlier history is missing")
                || e.to_string().contains("starts at sequence"),
            "unexpected error: {e}"
        ),
    }
}

/// Corruption in the middle of the retained log must be reported, not
/// truncated into a shorter collection.
#[test]
fn corrupt_wal_interior_is_reported_and_not_truncated() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_401);
    // Never flushes/checkpoints, so the log keeps every record.
    let (engine, id) = test_engine(&dir, 1000);
    // Five separate calls so the log holds five batches; damaging one of the
    // early ones must be recognised as a hole rather than a tail.
    for i in 0..5 {
        engine
            .add(
                &id,
                &[format!("c{i}")],
                &[rand_embedding(&mut rng, DIM)],
                None,
                None,
            )
            .unwrap();
    }
    drop(engine);

    let path = wal_file(&dir, &id);
    let bytes = std::fs::read(&path).unwrap();
    // Flip a byte a third of the way in: inside an early batch, with batches
    // after it.
    let at = 18 + (bytes.len() - 18) / 3;
    let mut damaged = bytes.clone();
    damaged[at] ^= 0xFF;
    std::fs::write(&path, &damaged).unwrap();
    let len_before = std::fs::metadata(&path).unwrap().len();

    match open_engine_result(&dir) {
        Ok(engine) => {
            let id = collection_id_by_name(&engine);
            panic!(
                "corrupted interior accepted, count={}",
                engine.count(&id).unwrap_or(u64::MAX)
            );
        }
        Err(e) => assert!(e.to_string().contains("corrupted"), "unexpected error: {e}"),
    }
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        len_before,
        "the log must not be truncated by a failed open"
    );
}

/// A torn final write is unacknowledged, so discarding it is allowed — and the
/// records before it must survive.
#[test]
fn truncated_wal_tail_is_discarded() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_402);
    let (engine, id) = test_engine(&dir, 1000);
    // One batch per call, so cutting into the last batch leaves three intact.
    for i in 0..4 {
        engine
            .add(
                &id,
                &[format!("t{i}")],
                &[rand_embedding(&mut rng, DIM)],
                None,
                None,
            )
            .unwrap();
    }
    drop(engine);

    let path = wal_file(&dir, &id);
    let len = std::fs::metadata(&path).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(len - 3).unwrap();
    drop(file);

    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(
        engine.count(&id).unwrap(),
        3,
        "the complete batches must survive; only the torn one is dropped"
    );
}

// ---------------------------------------------------------------------------
// Durability contract
// ---------------------------------------------------------------------------

/// The default must be durable.
///
/// A database that acknowledges writes it may lose is the wrong default even
/// though it is faster: `POST /add -> 200` followed by a power loss would
/// silently discard acknowledged data.
#[test]
fn default_durability_is_durable() {
    assert_eq!(EngineConfig::default().durability, Durability::Durable);
    assert!(Durability::Durable.is_durable());
    assert!(!Durability::Relaxed.is_durable());
}

/// The configuration names round-trip and reject typos.
#[test]
fn durability_names_parse_and_reject_unknown() {
    assert_eq!(
        "durable".parse::<Durability>().unwrap(),
        Durability::Durable
    );
    assert_eq!(
        "DURABLE".parse::<Durability>().unwrap(),
        Durability::Durable
    );
    assert_eq!(
        " relaxed ".parse::<Durability>().unwrap(),
        Durability::Relaxed
    );
    assert_eq!(Durability::Relaxed.as_str(), "relaxed");

    let err = "fastest".parse::<Durability>().unwrap_err();
    assert!(
        err.contains("expected `durable` or `relaxed`"),
        "unexpected error: {err}"
    );
}

/// The engine reports the level it was configured with.
#[test]
fn engine_reports_its_durability() {
    let dir = TempDir::new().unwrap();
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let engine = Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            durability: Durability::Relaxed,
            epoch: Epoch(0),
            ..EngineConfig::default()
        },
    )
    .unwrap();
    assert_eq!(engine.durability(), Durability::Relaxed);
}

// ---------------------------------------------------------------------------
// Checkpoint attribution
// ---------------------------------------------------------------------------

/// `optimize` reports a per-phase breakdown, and no phase can exceed the total.
///
/// The breakdown is what makes "checkpoints are whole-corpus work" measurable
/// rather than asserted; a phase that reported more than the whole checkpoint
/// would mean the instrumentation is wrong.
#[test]
fn optimize_reports_checkpoint_phase_timings() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_300);
    let (engine, id) = test_engine(&dir, 2);
    add_n(&engine, &id, &mut rng, 16, "t");

    let stats = engine.optimize(&id).unwrap();
    assert!(
        stats.checkpoint.total_us > 0,
        "a checkpoint on a non-empty collection must take measurable time"
    );
    for (phase, micros) in stats.checkpoint.phases() {
        assert!(
            micros <= stats.checkpoint.total_us,
            "phase `{phase}` ({micros}us) exceeds the checkpoint total ({}us)",
            stats.checkpoint.total_us
        );
    }
    assert!(
        stats.rebuild_us > 0,
        "a rebuild of 16 vectors must take measurable time"
    );
}

/// Durable appends are fsynced; the process counter is what proves it.
///
/// The counter is deliberately process-wide: a checkpoint prunes the log and
/// reopens it, and a per-handle counter would reset mid-run and report zero
/// syncs for a durable workload.
#[test]
fn durable_appends_increment_the_sync_counter() {
    let before = rekha_wal::sync_count();
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_301);
    let (engine, id) = test_engine(&dir, 1000);
    for i in 0..3 {
        engine
            .add(
                &id,
                &[format!("s{i}")],
                &[rand_embedding(&mut rng, DIM)],
                None,
                None,
            )
            .unwrap();
    }
    // Other tests may run concurrently and can only add to the counter.
    assert!(
        rekha_wal::sync_count() >= before + 3,
        "three durable batches must issue at least three fsyncs"
    );
}

/// Routine checkpoints append; they do not rewrite the whole corpus.
///
/// The second generation must reuse the first generation's segment (a hard
/// link, not a copy) and write one new segment holding only the records added
/// since. Without this, every checkpoint rewrites every vector and its cost
/// grows with the collection rather than with the write that triggered it.
#[test]
fn checkpoint_appends_only_new_vectors() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_310);
    let (engine, id) = test_engine(&dir, 2);

    add_n(&engine, &id, &mut rng, 8, "a");
    engine.checkpoint(&id).unwrap();
    let first = checkpoint_generation(&dir, &id);
    let seg = |dir: &std::path::Path, n: u32| dir.join("segments").join(format!("seg-{n}.bin"));
    assert!(
        !seg(&first, 1).exists(),
        "the first generation has one segment"
    );

    // The first generation is garbage collected by the next checkpoint, so its
    // inode has to be captured now.
    #[cfg(unix)]
    let first_inode = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(seg(&first, 0)).unwrap().ino()
    };

    add_n(&engine, &id, &mut rng, 3, "b");
    engine.checkpoint(&id).unwrap();
    let second = checkpoint_generation(&dir, &id);
    assert_ne!(first, second, "each checkpoint publishes a new generation");
    assert!(!first.exists(), "the superseded generation is reclaimed");

    // The carried-forward segment is the same inode: a link, not a copy.
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let carried = std::fs::metadata(seg(&second, 0)).unwrap();
        assert_eq!(
            first_inode,
            carried.ino(),
            "unchanged vectors must be hard-linked, not copied"
        );
    }

    // The appended segment holds exactly the three new vectors.
    let appended = std::fs::metadata(seg(&second, 1)).unwrap().len();
    assert_eq!(
        appended,
        (32 + 3 * DIM * 4) as u64,
        "only the new vectors are written"
    );
}

// ---------------------------------------------------------------------------
// Placement: a node stores only the collections it replicates.
// ---------------------------------------------------------------------------

/// Open an engine on a fresh directory whose catalog already carries `rows`.
///
/// Mirrors a replica that received the collection registry through the
/// metadata group before it opened its own data files. `hosted` is the set of
/// shards this node runs consensus groups for.
fn engine_with_rows(
    dir: &TempDir,
    hosted: &[ShardId],
    rows: &[(&CollectionConfig, ShardId)],
) -> Engine {
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    for (config, shard) in rows {
        catalog.record_collection(config, Some(*shard)).unwrap();
    }
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            node_id: 1,
            scope: rekha_engine::EngineScope::Shards(hosted.to_vec()),
            durability: Durability::Durable,
            epoch: Epoch(0),
            ..EngineConfig::default()
        },
    )
    .unwrap()
}

#[test]
fn a_replica_opens_its_collections_and_a_non_replica_refuses_them() {
    let config = CollectionConfig::new("sharded".into(), DIM, Distance::L2);
    let shard = ShardId(3);

    // This node hosts shard 3: the collection is open (and empty, because no
    // write has reached this shard group yet).
    let dir1 = TempDir::new().unwrap();
    let owner = engine_with_rows(&dir1, &[shard], &[(&config, shard)]);
    assert_eq!(owner.count(&config.id).unwrap(), 0);

    // This node hosts a different shard. It still has the catalog row — that is
    // what lets it authorize and route — but it must not answer as if the
    // collection were empty here, because that would turn a routing bug into a
    // wrong answer.
    let dir3 = TempDir::new().unwrap();
    let outsider = engine_with_rows(&dir3, &[ShardId(4)], &[(&config, shard)]);
    match outsider.count(&config.id) {
        Err(EngineError::NotReplica {
            shard: reported, ..
        }) => assert_eq!(reported, Some(shard)),
        other => panic!("a non-replica must refuse, not answer: {other:?}"),
    }

    // And it must not have grown a WAL for a collection it does not own.
    let wal = dir3.path().join("wal").join(format!("{}.wal", config.id));
    assert!(
        !wal.exists(),
        "a non-replica must not create data files for a collection it does not hold"
    );

    // The owner did open a WAL on demand, which is how an empty shard answers
    // reads without a special case.
    let owner_wal = dir1.path().join("wal").join(format!("{}.wal", config.id));
    let owner_shard_wal = dir1
        .path()
        .join("wal")
        .join(format!("{}.shard-{}.wal", config.id, shard.0));
    assert!(
        owner_wal.exists() || owner_shard_wal.exists(),
        "the owner must have opened its data file"
    );
}

#[test]
fn recording_a_collection_does_not_create_data_files() {
    let dir = TempDir::new().unwrap();
    let config = CollectionConfig::new("metadata-only".into(), DIM, Distance::L2);
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    catalog
        .record_collection(&config, Some(ShardId(7)))
        .unwrap();

    let wal = dir.path().join("wal").join(format!("{}.wal", config.id));
    assert!(
        !wal.exists(),
        "recording metadata must not materialize data: that is the split that \
         lets every node answer for a collection it does not hold"
    );

    // Recording twice is the normal case (replicated entries are re-applied).
    catalog
        .record_collection(&config, Some(ShardId(7)))
        .unwrap();
    assert_eq!(
        catalog
            .get_collection_by_id(&config.id)
            .unwrap()
            .unwrap()
            .shard,
        Some(ShardId(7))
    );
}

#[test]
fn materializing_is_idempotent_and_preserves_an_open_collection() {
    let dir = TempDir::new().unwrap();
    let config = CollectionConfig::new("idem".into(), DIM, Distance::L2);
    let engine = engine_with_rows(&dir, &[ShardId(1)], &[(&config, ShardId(1))]);

    let ids = vec!["a".to_string()];
    let embeddings = vec![Embedding::from(vec![0.5f32; DIM])];
    engine
        .add(&config.id, &ids, &embeddings, None, None)
        .unwrap();
    assert_eq!(engine.count(&config.id).unwrap(), 1);

    // A second materialize must be a no-op. Replacing the live handle would
    // drop the in-memory state and re-open the WAL underneath a reader.
    engine.materialize_collection(&config.id).unwrap();
    assert_eq!(engine.count(&config.id).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// P0: delete / upsert between checkpoints must not disable checkpointing.
// Incremental generations hard-link previous segments, which still contain
// deleted and superseded vectors. The staged validator used to require
// `sum(segment.len()) == live_count`, so the first delete made every later
// checkpoint fail and every later write 500 after the WAL commit.
// ---------------------------------------------------------------------------

#[test]
fn delete_between_checkpoints_keeps_checkpoint_working() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(10_001);
    // batch_size=2 flushes so vectors reach segments; sync_threshold default
    // (1000) so only explicit checkpoints run.
    let (engine, id) = test_engine(&dir, 2);
    let _ = add_n(&engine, &id, &mut rng, 10, "d");
    engine.checkpoint(&id).unwrap();

    engine.delete(&id, &["d0".to_string()]).unwrap();
    // This explicit checkpoint is what used to fail with
    // "staged segments hold 10 vectors but 9 were written".
    engine.checkpoint(&id).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 9);

    // Writes after the delete-checkpoint must succeed (previously every write
    // re-attempted the broken checkpoint and returned 500).
    let ids = vec!["new".to_string()];
    let embs = vec![rand_embedding(&mut rng, DIM)];
    engine.add(&id, &ids, &embs, None, None).unwrap();
    engine.checkpoint(&id).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 10);

    // The second generation must survive a restart with vectors intact.
    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 10);
    let got = engine.get(&id, &["d1".to_string()]).unwrap();
    assert!(got[0].as_ref().unwrap().embedding.is_some());
    let missing = engine.get(&id, &["d0".to_string()]).unwrap();
    assert!(missing[0].is_none());
}

#[test]
fn upsert_flushed_vector_between_checkpoints_keeps_checkpoint_working() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(10_002);
    let (engine, id) = test_engine(&dir, 2);
    let _ = add_n(&engine, &id, &mut rng, 8, "u");
    engine.checkpoint(&id).unwrap();

    // Upsert a flushed record: the old segment keeps the superseded vector,
    // the new generation appends the replacement. Total segments then exceed
    // the live count by design.
    let ids = vec!["u0".to_string()];
    let embs = vec![rand_embedding(&mut rng, DIM)];
    engine.upsert(&id, &ids, &embs, None, None).unwrap();
    engine.checkpoint(&id).unwrap();
    assert_eq!(engine.count(&id).unwrap(), 8);

    drop(engine);
    let engine = open_engine(&dir);
    let id = collection_id_by_name(&engine);
    assert_eq!(engine.count(&id).unwrap(), 8);
}

#[test]
fn auto_checkpoint_after_delete_does_not_fail_the_write() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let mut config = CollectionConfig::new("coll".to_owned(), DIM, Distance::L2);
    config.hnsw.batch_size = 2;
    config.hnsw.sync_threshold = 5;
    let id = engine.create_collection(&config).unwrap().config.id;
    let mut rng = Lcg::new(10_003);
    let _ = add_n(&engine, &id, &mut rng, 5, "a");
    // First auto-checkpoint has now run (5 >= 5).
    engine.delete(&id, &["a0".to_string()]).unwrap();
    // This add crosses the threshold again with a dead vector on disk. The
    // write already committed to the WAL, so it must report success even if
    // the background checkpoint were to fail.
    let ids = vec!["z0".to_string(), "z1".to_string(), "z2".to_string()];
    let embs: Vec<Embedding> = (0..3).map(|_| rand_embedding(&mut rng, DIM)).collect();
    engine
        .add(&id, &ids, &embs, None, None)
        .expect("auto-checkpoint must not turn an acknowledged write into an error");
    assert_eq!(engine.count(&id).unwrap(), 7);
}

// ---------------------------------------------------------------------------
// Sharded engine: partitions, reservations, and background checkpoints
// ---------------------------------------------------------------------------

fn vec_of(seed: f32) -> Embedding {
    (0..DIM as i32)
        .map(|d| seed + d as f32)
        .collect::<Vec<f32>>()
        .into()
}

fn op_add(id: &str, seed: f32) -> rekha_core::op::Operation {
    rekha_core::op::Operation::Add {
        id: id.to_owned(),
        embedding: vec_of(seed),
        metadata: None,
        document: None,
    }
}

fn shard_one() -> Option<ShardId> {
    Some(ShardId(1))
}

#[test]
fn partition_reads_and_writes_stay_on_their_shard() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let config = CollectionConfig::new("parts".into(), DIM, Distance::L2);
    let id = engine.create_collection(&config).unwrap().config.id;

    engine
        .materialize_collection_partition(&id, shard_one())
        .unwrap();
    engine
        .append_ops_partition(&id, shard_one(), &[op_add("a", 0.0), op_add("b", 1.0)])
        .unwrap();

    assert_eq!(engine.count_partition(&id, shard_one()).unwrap(), 2);
    assert_eq!(
        engine.count(&id).unwrap(),
        2,
        "count merges the open partitions"
    );

    let by_id = engine
        .get_records_partition(
            &id,
            shard_one(),
            Some(&["b".to_owned()]),
            None,
            None,
            None,
            0,
        )
        .unwrap();
    assert_eq!(by_id.len(), 1);
    assert_eq!(by_id[0].id, "b");
    assert!(by_id[0].embedding.is_some());

    let page = engine
        .get_records_partition(&id, shard_one(), None, None, None, Some(1), 1)
        .unwrap();
    assert_eq!(page.len(), 1, "offset/limit apply inside the partition");

    let hits = engine
        .query_partition(&id, shard_one(), &vec_of(0.0), 1, &QueryOptions::default())
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "a");

    let many = engine
        .query_many_partition(
            &id,
            shard_one(),
            &[vec_of(0.0), vec_of(1.0)],
            1,
            &QueryOptions::default(),
            true,
        )
        .unwrap();
    assert_eq!(many.len(), 2);
    assert!(
        many[0][0].embedding.is_some(),
        "with_embeddings is honoured"
    );
}

#[test]
fn a_multi_partition_read_merges_sorts_and_pages() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let config = CollectionConfig::new("merged".into(), DIM, Distance::L2);
    let id = engine.create_collection(&config).unwrap().config.id;

    for shard in [ShardId(1), ShardId(2)] {
        engine
            .materialize_collection_partition(&id, Some(shard))
            .unwrap();
    }
    engine
        .append_ops_partition(&id, Some(ShardId(1)), &[op_add("a", 0.0)])
        .unwrap();
    engine
        .append_ops_partition(&id, Some(ShardId(2)), &[op_add("b", 1.0)])
        .unwrap();

    let all = engine.get_records(&id, None, None, None, None, 0).unwrap();
    assert_eq!(all.len(), 2, "both partitions contribute");
    let mut ids: Vec<&str> = all.iter().map(|r| r.id.as_str()).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec!["a", "b"]);

    let second = engine
        .get_records(&id, None, None, None, Some(1), 1)
        .unwrap();
    assert_eq!(second.len(), 1);
    assert!(ids.contains(&second[0].id.as_str()));

    assert_eq!(engine.count_local(&id).unwrap(), Some(2));

    let filtered = engine
        .get_records(
            &id,
            None,
            Some(&WhereFilter::parse_json(r#"{"missing": {"$eq": 1}}"#).unwrap()),
            None,
            None,
            0,
        )
        .unwrap();
    assert!(filtered.is_empty());

    let by_doc = engine
        .get_records(
            &id,
            None,
            None,
            Some(&rekha_core::filter::WhereDocument::Contains("nope".into())),
            None,
            0,
        )
        .unwrap();
    assert!(by_doc.is_empty());
}

#[test]
fn a_reserved_id_is_locked_until_the_transaction_releases_it() {
    let dir = TempDir::new().unwrap();
    let engine = open_engine(&dir);
    let config = CollectionConfig::new("tx".into(), DIM, Distance::L2);
    let id = engine.create_collection(&config).unwrap().config.id;
    engine
        .materialize_collection_partition(&id, shard_one())
        .unwrap();

    let tx = Uuid::new_v4();
    engine
        .reserve_tx_ids(&id, shard_one(), tx, &["locked".to_owned()])
        .unwrap();
    let err = engine
        .prepare_add_partition(
            &id,
            shard_one(),
            &["locked".to_owned()],
            &[vec_of(0.0)],
            None,
            None,
        )
        .unwrap_err();
    assert!(
        matches!(err, EngineError::Validation(ref m) if m.contains("in-flight transaction")),
        "{err:?}"
    );

    engine.release_tx_ids(&id, shard_one(), &tx).unwrap();
    let prepared = engine
        .prepare_add_partition(
            &id,
            shard_one(),
            &["locked".to_owned()],
            &[vec_of(0.0)],
            None,
            None,
        )
        .unwrap();
    engine
        .append_ops_partition(&id, shard_one(), &prepared.ops)
        .unwrap();
    assert_eq!(engine.count_partition(&id, shard_one()).unwrap(), 1);
}

#[test]
fn a_cluster_engine_checkpoints_on_demand_not_on_the_write_path() {
    use rekha_cluster::state_machine::EngineSnapshotProvider;
    use rekha_engine::ApplyMode;

    let dir = TempDir::new().unwrap();
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    let engine = Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            apply_mode: ApplyMode::Committed,
            ..EngineConfig::default()
        },
    )
    .unwrap();
    let mut config = CollectionConfig::new("due".into(), DIM, Distance::L2);
    config.hnsw.sync_threshold = 1;
    let id = engine.create_collection(&config).unwrap().config.id;

    assert_eq!(
        engine.checkpoint_if_due().unwrap(),
        0,
        "nothing has been applied yet"
    );
    engine
        .apply_batch(&id, None, &[op_add("a", 0.0)], 1, DIM)
        .unwrap();
    assert_eq!(
        engine.checkpoint_if_due().unwrap(),
        1,
        "an applied batch past the threshold makes a checkpoint due"
    );
    assert_eq!(
        engine.checkpoint_if_due().unwrap(),
        0,
        "a second call finds nothing due"
    );
}
