use rekha_cluster::chord::{hash_to_chord_id, ChordNode};
use rekha_coordinator::{Coordinator, PeerPool};
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, SearchParams};
use rekha_storage::RekhaStore;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;

fn ivf_config(dim: u32) -> IvfConfig {
    IvfConfig {
        dim,
        nlist: 4,
        nprobe: 4,
        pq_m: 2,
        pq_k: 8,
        replication_factor: 3,
        distance_metric: DistanceMetric::L2,
    }
}

#[tokio::test]
async fn test_single_node_insert_and_search() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            "n1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let results = coord
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty(), "search should return results");
    assert_eq!(results[0].id, 1, "nearest vector should be id=1");
}

#[tokio::test]
async fn test_multi_collection() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();

    coord
        .create_collection(
            "dim4",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();
    coord
        .create_collection(
            "dim8",
            ivf_config(8),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "dim4",
            1,
            vec![0.5; 4],
            None,
            1000,
            "n1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();
    coord
        .insert(
            "dim8",
            1,
            vec![0.5; 8],
            None,
            1000,
            "n1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let r4 = coord
        .search("dim4", vec![0.5; 4], 5, SearchParams::default())
        .await
        .unwrap();
    let r8 = coord
        .search("dim8", vec![0.5; 8], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!r4.is_empty(), "dim=4 collection should work");
    assert!(!r8.is_empty(), "dim=8 collection should work");
}

#[tokio::test]
async fn test_delete() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "test",
            1,
            vec![0.5; 4],
            None,
            1000,
            "n1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();
    let deleted = coord
        .delete("test", &[1], 1001, "n1", ConsistencyLevel::One, false)
        .await
        .unwrap();
    assert_eq!(deleted, 1);

    let results = coord
        .search("test", vec![0.5; 4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(results.is_empty(), "deleted vector should not be found");
}

#[tokio::test]
async fn test_fetch() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "test",
            1,
            vec![0.5; 4],
            Some(vec![1, 2, 3]),
            1000,
            "n1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let results = coord.fetch("test", &[1], true).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 1);
    assert_eq!(results[0].payload, Some(vec![1, 2, 3]));
}

#[tokio::test]
async fn test_list_collections() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();

    let names = coord.list_collections().await.unwrap();
    assert!(
        names.contains(&"default".to_string()),
        "default collection should exist after init"
    );

    coord
        .create_collection(
            "extra",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();
    let names = coord.list_collections().await.unwrap();
    assert!(names.contains(&"extra".to_string()));
}

#[tokio::test]
async fn test_rebuild_index() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 0..20 {
        coord
            .insert(
                "test",
                i,
                vec![0.1 * i as f32; 4],
                None,
                1000,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    coord.rebuild_index("test").await.unwrap();

    let results = coord
        .search("test", vec![0.1; 4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty(), "search should work after rebuild");
}

#[tokio::test]
async fn test_export_collection() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=5 {
        coord
            .insert(
                "test",
                i,
                vec![i as f32 * 0.1; 4],
                Some(vec![i as u8; 4]),
                1000 + i as i64,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    let exported = coord
        .export_collection("test", 0, 10, true, true)
        .await
        .unwrap();
    assert_eq!(exported.len(), 5);
    for (idx, ev) in exported.iter().enumerate() {
        assert_eq!(ev.id, (idx + 1) as u64);
        assert_eq!(ev.vector.len(), 4);
        assert_eq!(ev.payload, Some(vec![(idx + 1) as u8; 4]));
        assert_eq!(ev.timestamp, 1000 + (idx + 1) as i64);
    }

    let exported = coord
        .export_collection("test", 0, 10, true, false)
        .await
        .unwrap();
    assert_eq!(exported.len(), 5);
    for ev in exported {
        assert!(!ev.vector.is_empty());
        assert!(ev.payload.is_none());
    }

    let exported = coord
        .export_collection("test", 0, 10, false, true)
        .await
        .unwrap();
    assert_eq!(exported.len(), 5);
    for ev in exported {
        assert!(ev.vector.is_empty());
        assert!(ev.payload.is_some());
    }

    let exported = coord
        .export_collection("test", 2, 2, true, true)
        .await
        .unwrap();
    assert_eq!(exported.len(), 2);
    assert_eq!(exported[0].id, 3);
    assert_eq!(exported[1].id, 4);
}

#[tokio::test]
async fn test_export_stream() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=10 {
        coord
            .insert(
                "test",
                i,
                vec![i as f32 * 0.1; 4],
                None,
                1000 + i as i64,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    let mut rx = coord.export_stream("test", 0, 10, true, true, 3);
    let mut total = 0;
    while let Some(batch) = rx.recv().await {
        let batch = batch.unwrap();
        total += batch.len();
        assert!(batch.len() <= 3);
        for ev in batch {
            assert!(!ev.vector.is_empty());
        }
    }
    assert_eq!(total, 10);

    let mut rx = coord.export_stream("test", 2, 5, true, true, 2);
    let mut total = 0;
    while let Some(batch) = rx.recv().await {
        let batch = batch.unwrap();
        total += batch.len();
    }
    assert_eq!(total, 5);
}

#[tokio::test]
async fn test_transfer_shard_out() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=15 {
        coord
            .insert(
                "test",
                i,
                vec![i as f32 * 0.1; 4],
                None,
                1000 + i as i64,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let chunks = coord.transfer_shard_out("test", 5).await.unwrap();
    assert!(!chunks.is_empty());
    assert!(chunks.last().unwrap().final_chunk);

    let first = &chunks[0];
    assert!(!first.centroids.is_empty());
    assert_eq!(first.nlist, 4);
    assert_eq!(first.nprobe, 4);
    assert_eq!(first.total_dim, 4);

    let mut total_vectors = 0;
    for chunk in &chunks {
        for batch in &chunk.vector_batches {
            total_vectors += batch.vectors.len();
        }
    }
    assert_eq!(total_vectors, 15);
}

#[tokio::test]
async fn test_transfer_shard_stream() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=12 {
        coord
            .insert(
                "test",
                i,
                vec![i as f32 * 0.1; 4],
                None,
                1000 + i as i64,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut rx = coord.transfer_shard_stream("test", 5);
    let mut chunks = Vec::new();
    while let Some(chunk) = rx.recv().await {
        chunks.push(chunk.unwrap());
    }
    assert!(!chunks.is_empty());
    assert!(chunks.last().unwrap().final_chunk);
    assert!(!chunks[0].centroids.is_empty());
    assert_eq!(chunks[0].nlist, 4);

    let mut total_vectors = 0;
    for chunk in &chunks {
        for batch in &chunk.vector_batches {
            total_vectors += batch.vectors.len();
        }
    }
    assert_eq!(total_vectors, 12);
}

#[tokio::test]
async fn test_repair_collection() {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(rekha_cluster::Membership::new("n1", 5000)));
    let coord = Arc::new(Coordinator::new(
        store.clone(),
        membership,
        1,
        "n1".to_string(),
        true,
        3600,
        ConsistencyLevel::Quorum,
        3,
        Arc::new(ChordNode::new(hash_to_chord_id(b"n1"), "127.0.0.1:5000")),
        Arc::new(PeerPool::new()),
        86400,
    ));
    coord.initialize().await.unwrap();
    coord
        .create_collection(
            "test",
            ivf_config(4),
            "n1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=5 {
        coord
            .insert(
                "test",
                i,
                vec![i as f32 * 0.1; 4],
                None,
                1000 + i as i64,
                "n1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    let progress = coord.repair_collection("test").await.unwrap();
    assert_eq!(progress.total, 5);
    assert_eq!(progress.repaired, 0);
    assert_eq!(progress.current_node, "n1");
}
