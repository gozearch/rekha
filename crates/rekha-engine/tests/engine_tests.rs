//! Integration tests for `rekha-engine`. The correctness oracle is
//! `rekha_distance`, which implements Chroma-exact distance semantics.

use std::collections::HashSet;
use std::sync::Arc;

use rekha_core::cluster::Epoch;
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::types::{Distance, Embedding, Id, Metadata, MetadataValue};
use rekha_distance::{distance, l2_squared};
use rekha_engine::{Engine, EngineConfig, EngineError, QueryOptions};
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
    let catalog: Arc<dyn Catalog> =
        Arc::new(RedbCatalog::open(dir.path().join("catalog.redb")).unwrap());
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path().join("objects")));
    Engine::open(
        catalog,
        storage,
        dir.path().join("wal"),
        EngineConfig {
            wal_fsync: false,
            epoch: Epoch(0),
        },
    )
    .unwrap()
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
        .update(&id, &["u1".to_string()], Some(&[Some(meta)]), None)
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

    // Update on a missing id.
    let empty_meta: Option<Metadata> = None;
    let err = engine
        .update(&id, &["nope".to_string()], Some(&[empty_meta]), None)
        .unwrap_err();
    assert!(matches!(err, EngineError::Validation(_)));

    // Update on an existing id works and changes nothing else.
    engine
        .update(
            &id,
            &["a".to_string()],
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
            Some(&[Some(meta_with_tag("b", 1))]),
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

/// A checkpoint taken before any flush stores the records but an empty index;
/// the reopen replays the (kept) WAL tail into the buffer.
#[test]
fn checkpoint_without_flush_reopen_replays_tail() {
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
    assert_eq!(engine.committed_count(&id).unwrap(), 0, "index stays empty");
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

/// A corrupt checkpoint file must degrade to a full WAL replay, never a panic.
#[test]
fn corrupt_checkpoint_falls_back_to_full_replay() {
    for corrupt in ["checkpoint.json", "records.bin"] {
        let dir = TempDir::new().unwrap();
        let mut rng = Lcg::new(9_004);
        // Unflushed records: the checkpoint prunes nothing (flushed_seq = 0), so
        // the WAL still holds the full history for the fallback replay.
        let (engine, id) = test_engine(&dir, 1000);
        let (_, _, stored) = add_n(&engine, &id, &mut rng, 6, "f");
        engine.checkpoint(&id).unwrap();

        let ckpt = dir.path().join("wal").join(format!("{id}.checkpoint"));
        std::fs::write(ckpt.join(corrupt), b"garbage that is not a checkpoint").unwrap();

        drop(engine);
        let engine = open_engine(&dir);
        let id = collection_id_by_name(&engine);
        assert_eq!(
            engine.count(&id).unwrap(),
            6,
            "full replay recovers from corrupt {corrupt}"
        );
        for _ in 0..3 {
            let q = rand_embedding(&mut rng, DIM);
            assert_eq!(
                query_top_k(&engine, &id, &q, 6, None),
                brute_force_top_k(&stored, &q, 6)
            );
        }
    }
}

/// A lost/corrupt index.bin must NOT lose the records: once the WAL is pruned,
/// records.bin is the only copy. Reopen degrades to an empty index (queries
/// fall back to the exact brute-force path) and keeps every record.
#[test]
fn missing_index_bin_keeps_records() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_005);
    let (engine, id) = test_engine(&dir, 2); // flushed -> checkpoint carries index.bin
    let (_, _, stored) = add_n(&engine, &id, &mut rng, 8, "m");
    engine.checkpoint(&id).unwrap();

    let ckpt = dir.path().join("wal").join(format!("{id}.checkpoint"));
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
        0,
        "index degrades to empty"
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
    let seg_dir = dir
        .path()
        .join("wal")
        .join(format!("{id}.checkpoint"))
        .join("segments");
    assert!(seg_dir.join("seg-0.bin").exists(), "segment file written");
    assert!(
        dir.path()
            .join("wal")
            .join(format!("{id}.checkpoint"))
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
    let seg_dir = dir
        .path()
        .join("wal")
        .join(format!("{id}.checkpoint"))
        .join("segments");
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
// WAL shipping: wal_delta / wal_last_seq / apply_remote_ops
// ---------------------------------------------------------------------------

/// Engine::wal_delta returns records from the WAL starting at from_seq.
#[test]
fn wal_delta_returns_records_from_seq() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_030);
    let (engine, id) = test_engine(&dir, 1000);
    let _ = add_n(&engine, &id, &mut rng, 5, "d");

    // All 5 records are in the WAL (not flushed).
    let delta = engine.wal_delta(&id, 1).unwrap();
    assert_eq!(delta.records.len(), 5);
    assert_eq!(delta.target_seq, 5);

    // From seq 3: should get records 3, 4, 5.
    let delta = engine.wal_delta(&id, 3).unwrap();
    assert_eq!(delta.records.len(), 3);

    // From seq 10 (beyond last): empty.
    let delta = engine.wal_delta(&id, 10).unwrap();
    assert!(delta.records.is_empty());
}

/// Engine::wal_last_seq reports the correct WAL head.
#[test]
fn wal_last_seq_reports_head() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_032);
    let (engine, id) = test_engine(&dir, 1000);
    assert_eq!(engine.wal_last_seq(&id).unwrap(), 0, "empty WAL");
    let _ = add_n(&engine, &id, &mut rng, 7, "w");
    assert_eq!(engine.wal_last_seq(&id).unwrap(), 7, "7 records written");
}

/// Engine::apply_remote_ops applies operations without WAL append.
#[test]
fn apply_remote_ops_applies_without_wal() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_031);
    let (engine, id) = test_engine(&dir, 1000);

    // Apply remote ops directly (no local WAL append).
    let ops: Vec<_> = (0..3)
        .map(|i| {
            let emb: Embedding = rand_embedding(&mut rng, DIM);
            (
                i as u64 + 1,
                rekha_core::op::Operation::Add {
                    id: format!("r{i}"),
                    embedding: emb,
                    metadata: None,
                    document: None,
                },
            )
        })
        .collect();
    engine.apply_remote_ops(&id, ops).unwrap();

    assert_eq!(engine.count(&id).unwrap(), 3);
}

/// Round-trip: leader wal_delta → decode_ops → follower apply_remote_ops.
#[test]
fn wal_delta_roundtrip_to_follower() {
    let dir = TempDir::new().unwrap();
    let mut rng = Lcg::new(9_033);
    let (engine, id) = test_engine(&dir, 1000);
    let (ids, embs, _) = add_n(&engine, &id, &mut rng, 4, "f");

    // Leader ships delta from seq 1.
    let delta = engine.wal_delta(&id, 1).unwrap();
    assert_eq!(delta.records.len(), 4);

    // Follower applies the decoded ops.
    let dir2 = TempDir::new().unwrap();
    let (engine2, id2) = test_engine(&dir2, 1000);
    let ops = delta.decode_ops();
    engine2.apply_remote_ops(&id2, ops).unwrap();
    assert_eq!(engine2.count(&id2).unwrap(), 4);

    // Verify the follower's records match the leader's.
    for (want_id, want_emb) in ids.iter().zip(&embs) {
        let rec = engine2.get(&id2, &[want_id.clone()]).unwrap()[0]
            .clone()
            .expect("record present on follower");
        assert_eq!(rec.embedding.as_ref().unwrap().as_ref(), want_emb.as_ref());
    }
}
