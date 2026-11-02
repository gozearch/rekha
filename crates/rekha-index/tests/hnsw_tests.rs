//! Integration tests for `UsearchIndex`. The correctness oracle is
//! `rekha_distance`, which implements Chroma-exact distance semantics.

use std::collections::HashSet;

use rekha_core::types::{Distance, Embedding};
use rekha_distance::{distance, l2_squared};
use rekha_index::{Index, IndexError, UsearchIndex};

/// xorshift64 — deterministic, dependency-free RNG.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Uniform `f32` in `[0, 1)`.
fn rand_f32(state: &mut u64) -> f32 {
    (xorshift(state) >> 40) as f32 / (1u64 << 24) as f32
}

fn rand_embedding(state: &mut u64, dim: usize) -> Embedding {
    (0..dim)
        .map(|_| rand_f32(state))
        .collect::<Vec<f32>>()
        .into()
}

/// Relative difference, tolerant of a zero/denormal expected value.
fn rel_diff(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(1e-12)
}

#[test]
fn chroma_distance_semantics_match_rekha_distance() {
    for space in [Distance::L2, Distance::Cosine, Distance::Ip] {
        let mut index = UsearchIndex::new(space, 8).unwrap();
        let stored = vec![
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
        for (id, v) in &stored {
            index.add(id, v).unwrap();
        }
        let query: Embedding = vec![0.2, 0.8, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7].into();
        let hits = index.search(&query, 3, 200).unwrap();
        assert_eq!(hits.len(), 3);
        for hit in &hits {
            let expected = distance(
                space,
                &query,
                &stored.iter().find(|(id, _)| *id == hit.id).unwrap().1,
            );
            assert!(
                rel_diff(hit.distance, expected) < 1e-4,
                "{space:?}: id `{}` got {}, expected {}",
                hit.id,
                hit.distance,
                expected
            );
        }
    }
}

#[test]
fn knn_matches_brute_force() {
    let dim = 16;
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut index = UsearchIndex::new(Distance::L2, dim).unwrap();
    let mut stored = Vec::new();
    for i in 0..200 {
        let id = format!("v{i:03}");
        let v = rand_embedding(&mut rng, dim);
        index.add(&id, &v).unwrap();
        stored.push((id, v));
    }
    for q in 0..20 {
        let query = rand_embedding(&mut rng, dim);
        let hits = index.search(&query, 10, 200).unwrap();
        assert_eq!(hits.len(), 10, "query {q}: expected 10 hits");
        let mut expected: Vec<(String, f32)> = stored
            .iter()
            .map(|(id, v)| (id.clone(), l2_squared(&query, v)))
            .collect();
        expected.sort_by(|a, b| a.1.total_cmp(&b.1));
        let expected_ids: HashSet<&String> = expected.iter().take(10).map(|(id, _)| id).collect();
        let got_ids: HashSet<&String> = hits.iter().map(|h| &h.id).collect();
        assert_eq!(got_ids, expected_ids, "query {q}: id set mismatch");
    }
}

#[test]
fn add_search_delete_lifecycle() {
    let mut index = UsearchIndex::new(Distance::L2, 4).unwrap();
    let a: Embedding = vec![0.1, 0.2, 0.3, 0.4].into();
    let b: Embedding = vec![0.9, 0.8, 0.7, 0.6].into();
    let c: Embedding = vec![0.5, 0.5, 0.5, 0.5].into();

    index.add(&"a".to_string(), &a).unwrap();
    index.add(&"b".to_string(), &b).unwrap();
    index.add(&"c".to_string(), &c).unwrap();
    assert_eq!(index.len(), 3);
    assert!(index.contains(&"a".to_string()));

    let hits = index.search(&b, 3, 64).unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "b", "a point is its own nearest neighbor");

    index.delete(&"a".to_string()).unwrap();
    assert_eq!(index.len(), 2);
    assert!(!index.contains(&"a".to_string()));
    let hits = index.search(&a, 3, 64).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "a"),
        "deleted id must not be returned"
    );

    index.delete(&"a".to_string()).unwrap();
    assert_eq!(index.len(), 2, "delete of an absent id is idempotent");
}

#[test]
fn duplicate_id_rejected_and_readd_after_delete() {
    let mut index = UsearchIndex::new(Distance::L2, 4).unwrap();
    let v: Embedding = vec![1.0, 2.0, 3.0, 4.0].into();
    index.add(&"dup".to_string(), &v).unwrap();
    assert!(matches!(
        index.add(&"dup".to_string(), &v),
        Err(IndexError::DuplicateId(_))
    ));
    index.delete(&"dup".to_string()).unwrap();
    index.add(&"dup".to_string(), &v).unwrap();
    assert!(index.contains(&"dup".to_string()));
    assert_eq!(index.len(), 1);
}

#[test]
fn dimension_mismatch_rejected() {
    let mut index = UsearchIndex::new(Distance::L2, 4).unwrap();
    let wrong: Embedding = vec![1.0, 2.0].into();
    assert!(matches!(
        index.add(&"x".to_string(), &wrong),
        Err(IndexError::DimensionMismatch { index: 4, got: 2 })
    ));
    assert_eq!(index.len(), 0);
}

#[test]
fn empty_index_search_is_empty() {
    let index = UsearchIndex::new(Distance::L2, 8).unwrap();
    assert!(index.is_empty());
    let q: Embedding = vec![0.0; 8].into();
    let hits = index.search(&q, 10, 100).unwrap();
    assert!(hits.is_empty());
}

#[test]
fn ef_does_not_change_tiny_index_results() {
    let dim = 8;
    let mut rng = 0x0123_4567_89AB_CDEFu64;
    let mut index = UsearchIndex::new(Distance::L2, dim).unwrap();
    for i in 0..50 {
        index
            .add(&format!("n{i:02}"), &rand_embedding(&mut rng, dim))
            .unwrap();
    }
    let query = rand_embedding(&mut rng, dim);
    // k = 50 requests the entire collection, so the beam is at least 50 for
    // every ef (usearch clamps expansion to max(ef, k)); on 50 points the
    // result must be the full set regardless of ef.
    let sets: Vec<HashSet<String>> = [1usize, 16, 256]
        .into_iter()
        .map(|ef| {
            let hits = index.search(&query, 50, ef).unwrap();
            assert_eq!(hits.len(), 50, "ef {ef}: expected the full set back");
            hits.into_iter().map(|h| h.id).collect()
        })
        .collect();
    assert_eq!(sets[0], sets[1]);
    assert_eq!(sets[1], sets[2]);
}

#[test]
fn cosine_distances_match_rekha_distance() {
    let dim = 8;
    let mut rng = 0x0F0F_0F0F_0F0F_0F0Fu64;
    let mut index = UsearchIndex::new(Distance::Cosine, dim).unwrap();
    let mut stored = Vec::new();
    for i in 0..20 {
        let id = format!("c{i}");
        let v = rand_embedding(&mut rng, dim);
        index.add(&id, &v).unwrap();
        stored.push((id, v));
    }
    let query = rand_embedding(&mut rng, dim);
    let hits = index.search(&query, 5, 128).unwrap();
    assert_eq!(hits.len(), 5);
    for hit in &hits {
        let expected = distance(
            Distance::Cosine,
            &query,
            &stored.iter().find(|(id, _)| *id == hit.id).unwrap().1,
        );
        assert!(
            rel_diff(hit.distance, expected) < 1e-4,
            "id `{}` got {}, expected {}",
            hit.id,
            hit.distance,
            expected
        );
    }
}

#[test]
fn save_load_roundtrip_preserves_maps_and_search() {
    use std::path::PathBuf;

    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("idx").join("index.bin");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    let dim = 16;
    let mut rng = 0xDEAD_BEEF_CAFE_F00Du64;
    let mut stored = Vec::new();
    let mut index = UsearchIndex::new(Distance::L2, dim).unwrap();
    for i in 0..100 {
        let id = format!("s{i:03}");
        let v = rand_embedding(&mut rng, dim);
        index.add(&id, &v).unwrap();
        stored.push((id, v));
    }

    // Search before the save for a later set-equality comparison.
    let query = rand_embedding(&mut rng, dim);
    let before: HashSet<String> = index
        .search(&query, 10, 200)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(before.len(), 10);

    index.save(&path).unwrap();
    assert!(path.exists());
    assert!(path.with_extension("meta").exists());

    let mut loaded = UsearchIndex::load(&path, Distance::L2, dim).unwrap();
    assert_eq!(loaded.len(), 100);
    assert_eq!(loaded.dimension(), dim);
    assert_eq!(loaded.space(), Distance::L2);
    for i in 0..100 {
        let id = format!("s{i:03}");
        assert!(loaded.contains(&id), "id {id} must be present after load");
    }

    // Same query, same top-k id set.
    let after: HashSet<String> = loaded
        .search(&query, 10, 200)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(after, before, "top-k id set must survive save/load");

    // Maps stay consistent after the round-trip: add/delete still work.
    let extra: Embedding = rand_embedding(&mut rng, dim);
    loaded.add(&"newbie".to_string(), &extra).unwrap();
    assert!(loaded.contains(&"newbie".to_string()));
    loaded.delete(&"s000".to_string()).unwrap();
    assert!(!loaded.contains(&"s000".to_string()));
    assert_eq!(loaded.len(), 100);

    // Fresh id gets a label above everything the round-trip restored.
    let again: Embedding = rand_embedding(&mut rng, dim);
    loaded.add(&"fresh".to_string(), &again).unwrap();
    assert!(loaded.contains(&"fresh".to_string()));
}

#[test]
fn save_load_preserves_ids_after_delete() {
    use std::path::PathBuf;

    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("index.bin");

    let dim = 8;
    let mut rng = 0x1234_5678_9ABC_DEF0u64;
    let mut index = UsearchIndex::new(Distance::L2, dim).unwrap();
    for i in 0..10 {
        index
            .add(&format!("d{i}"), &rand_embedding(&mut rng, dim))
            .unwrap();
    }
    // Delete two, then save: the graph stores the live set and the meta maps
    // agree with it.
    index.delete(&"d3".to_string()).unwrap();
    index.delete(&"d7".to_string()).unwrap();
    assert_eq!(index.len(), 8);

    index.save(&path).unwrap();
    let loaded = UsearchIndex::load(&path, Distance::L2, dim).unwrap();
    assert_eq!(loaded.len(), 8);
    assert!(!loaded.contains(&"d3".to_string()));
    assert!(!loaded.contains(&"d7".to_string()));
    for i in [0, 1, 2, 4, 5, 6, 8, 9] {
        assert!(loaded.contains(&format!("d{i}")));
    }
}

#[test]
fn load_with_wrong_dimension_errors() {
    use std::path::PathBuf;

    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("index.bin");

    let mut index = UsearchIndex::new(Distance::L2, 8).unwrap();
    index
        .add(&"a".to_string(), &rand_embedding(&mut 7u64, 8))
        .unwrap();
    index.save(&path).unwrap();

    assert!(matches!(
        UsearchIndex::load(&path, Distance::L2, 16),
        Err(IndexError::Corrupt(_))
    ));
    assert!(matches!(
        UsearchIndex::load(&path, Distance::Cosine, 8),
        Err(IndexError::Corrupt(_))
    ));
}

#[test]
fn load_garbage_file_errors_without_panicking() {
    use std::path::PathBuf;

    let dir = tempfile::tempdir().unwrap();
    let path: PathBuf = dir.path().join("index.bin");
    std::fs::write(&path, b"this is not a usearch index").unwrap();
    std::fs::write(
        path.with_extension("meta"),
        b"definitely not bincode IndexMeta",
    )
    .unwrap();

    assert!(UsearchIndex::load(&path, Distance::L2, 8).is_err());
    // Missing meta file is also an error, never a panic.
    let ghost: PathBuf = dir.path().join("ghost.bin");
    assert!(UsearchIndex::load(&ghost, Distance::L2, 8).is_err());
}
