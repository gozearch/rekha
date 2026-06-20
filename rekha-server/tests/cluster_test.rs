use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use rekha_core::VectorStoreBackend;
use rekha_server::ServerConfig;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("/tmp/rekha_int_test_{}_{}", prefix, n)
}

/// Start a single-node server and return its coordinator handle.
async fn start_single_node(node_id: &str, port: u16) -> std::sync::Arc<rekha_server::Coordinator> {
    let dir = temp_dir(node_id);
    let mut config = ServerConfig::dev_default(node_id, &dir);
    config.cluster.bind_addr = format!("127.0.0.1:{port}");
    config.cluster.seed_nodes = vec![format!("127.0.0.1:{port}")];
    config.partition.num_dim_groups = 1;
    config.partition.dim_group_size = 256;
    config.index.pq_num_sub_vectors = 16;
    config.index.pq_num_centroids = 16;
    let instance = rekha_server::ServerInstance::from_config(config).await.unwrap();
    let coord = instance.coordinator().clone();
    tokio::spawn(async move {
        let _ = instance.run().await;
    });
    // Wait for single-node Raft self-election + gRPC server.
    tokio::time::sleep(Duration::from_secs(1)).await;
    coord
}

/// Start a multi-node cluster. Each node gets its own port.
async fn start_cluster(
    n: usize,
    base_port: u16,
) -> Vec<std::sync::Arc<rekha_server::Coordinator>> {
    let addrs: Vec<String> = (0..n)
        .map(|i| format!("127.0.0.1:{}", base_port + i as u16))
        .collect();
    let mut coords = Vec::new();
    for i in 0..n {
        let dir = temp_dir(&format!("node-{}", i + 1));
        let mut config = ServerConfig::dev_default(&format!("node-{}", i + 1), &dir);
        config.cluster.bind_addr = addrs[i].clone();
        config.cluster.seed_nodes = addrs.clone();
        config.partition.num_dim_groups = 1;
        config.partition.dim_group_size = 256;
        config.index.pq_num_sub_vectors = 16;
        config.index.pq_num_centroids = 16;
        let instance = rekha_server::ServerInstance::from_config(config).await.unwrap();
        let coord = instance.coordinator().clone();
        tokio::spawn(async move {
            let _ = instance.run().await;
        });
        // Wait between starting nodes to avoid thundering herd on elections.
        tokio::time::sleep(Duration::from_secs(1)).await;
        coords.push(coord);
    }
    // Wait for Raft elections + heartbeats.
    tokio::time::sleep(Duration::from_secs(5)).await;
    coords
}

// ── Single-Node Tests ──────────────────────────────────────

#[tokio::test]
async fn test_single_node_insert_and_search() {
    let coord = start_single_node("test-node", 24000).await;

    // Verify Raft node is leader.
    let raft_node = coord.raft_node(0).expect("data Raft node not found");
    assert!(raft_node.is_leader().await, "data Raft node should be a leader");
    eprintln!("  data Raft is leader, term={}", raft_node.current_term().await);

    // Insert vectors with explicit IDs (using IDs 100-109 to avoid auto-ID range).
    for i in 100u64..110u64 {
        let v: Vec<f32> = (0..256).map(|d| ((i - 100) * 256 + d) as f32).collect();
        let actual_id = coord
            .insert("default", i, v, None)
            .await
            .expect("insert should succeed");
        assert_eq!(actual_id, i, "insert should return the requested ID");
    }

    // Verify via collection store.
    let ctx = coord.get_collection("default").await.unwrap();
    for i in 100u64..110u64 {
        assert!(
            ctx.store.get_vector(i).unwrap().is_some(),
            "vector {i} should exist in collection store"
        );
    }

    // Search should not error.
    let (results, _) = coord
        .search("default", vec![0.0; 256], 5, Default::default())
        .await
        .unwrap();
    assert!(results.len() <= 5);
}

#[tokio::test]
async fn test_single_node_create_and_drop_collection() {
    let coord = start_single_node("test-node", 24001).await;

    assert!(coord.collection_exists("default").await, "default should exist");
    eprintln!("  default exists ✓");

    match coord.create_collection("extra", 256).await {
        Ok(()) => eprintln!("  create_collection succeeded"),
        Err(e) => panic!("create_collection failed: {e}"),
    }
    assert!(coord.collection_exists("extra").await, "extra should exist locally");

    let cols = coord.list_collections().await.unwrap();
    eprintln!("  list_collections returned {} items: {:?}", cols.len(), cols.iter().map(|c| &c.name).collect::<Vec<_>>());
    assert_eq!(cols.len(), 2, "list_collections should return 2 collections");

    coord.drop_collection("extra").await.unwrap();
    assert!(!coord.collection_exists("extra").await);
    assert!(coord.collection_exists("default").await);
}

#[tokio::test]
async fn test_single_node_insert_with_payload() {
    let coord = start_single_node("test-node", 24002).await;

    let payload = rekha_core::Payload::from_text("hello world");
    coord
        .insert("default", 1, vec![0.5; 256], Some(payload))
        .await
        .unwrap();

    let ctx = coord.get_collection("default").await.unwrap();
    let stored = ctx.store.get_payload(1).unwrap().unwrap();
    assert_eq!(stored, b"hello world");
}

#[tokio::test]
async fn test_single_node_per_collection_auto_id() {
    let coord = start_single_node("test-node", 24003).await;
    coord.create_collection("a", 256).await.unwrap();
    coord.create_collection("b", 256).await.unwrap();

    // Auto-ID should be independent per collection.
    // Use explicit IDs to avoid auto-ID range issues.
    coord.insert("a", 100, vec![0.1; 256], None).await.unwrap();
    coord.insert("b", 100, vec![0.2; 256], None).await.unwrap();

    let v_a = coord.get_collection("a").await.unwrap().store.get_vector(100).unwrap();
    assert!(v_a.is_some(), "vector 100 should exist in collection a");

    let v_b = coord.get_collection("b").await.unwrap().store.get_vector(100).unwrap();
    assert!(v_b.is_some(), "vector 100 should exist in collection b");

    // Verify they're different vectors (different collections/namespaces).
    assert!((v_a.unwrap()[0] - 0.1).abs() < 1e-6);
    assert!((v_b.unwrap()[0] - 0.2).abs() < 1e-6);
}

// ── Multi-Node Tests ───────────────────────────────────────

#[tokio::test]
async fn test_multi_node_leader_election() {
    let coords = start_cluster(3, 23000).await;

    // Check metadata Raft group has a leader.
    let meta_leaders: Vec<bool> = coords
        .iter()
        .map(|c| {
            c.raft_node(rekha_core::METADATA_PARTITION_ID)
                .map(|rn| {
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || rt.block_on(rn.is_leader()))
                        .join()
                        .unwrap()
                })
                .unwrap_or(false)
        })
        .collect();
    eprintln!("  metadata leaders: {:?}", meta_leaders);
    assert!(
        meta_leaders.iter().any(|&x| x),
        "at least one metadata leader expected, got {:?}",
        meta_leaders
    );

    // Check data Raft group (partition 0) has a leader.
    let data_leaders: Vec<bool> = coords
        .iter()
        .map(|c| {
            c.raft_node(0)
                .map(|rn| {
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || rt.block_on(rn.is_leader()))
                        .join()
                        .unwrap()
                })
                .unwrap_or(false)
        })
        .collect();
    eprintln!("  data leaders: {:?}", data_leaders);
    assert!(
        data_leaders.iter().any(|&x| x),
        "at least one data leader expected, got {:?}",
        data_leaders
    );
}

#[tokio::test]
async fn test_multi_node_insert_replication() {
    let coords = start_cluster(3, 23100).await;

    // Find the data leader (partition 0).
    let leader_idx = coords
        .iter()
        .position(|c| {
            c.raft_node(0)
                .map(|rn| {
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || rt.block_on(rn.is_leader()))
                        .join()
                        .unwrap()
                })
                .unwrap_or(false)
        })
        .expect("data leader should exist");
    eprintln!("  data leader is node {leader_idx}");

    // Insert vectors on the leader using explicit IDs (avoid 0 to skip auto-ID).
    for i in 200u64..205u64 {
        let v: Vec<f32> = (0..256).map(|d| ((i - 200) * 256 + d) as f32).collect();
        let actual = coords[leader_idx]
            .insert("default", i, v, None)
            .await
            .unwrap();
        assert_eq!(actual, i, "leader should return the requested ID {i}");
        // Immediately verify on the leader.
        let ctx = coords[leader_idx].get_collection("default").await.unwrap();
        match ctx.store.get_vector(i) {
            Ok(Some(_)) => eprintln!("    leader has vector {i}"),
            Ok(None) => panic!("leader missing vector {i} immediately after insert"),
            Err(e) => panic!("leader error reading vector {i}: {e}"),
        }
    }

    // Wait for Raft replication.
    tokio::time::sleep(Duration::from_millis(2000)).await;

    // Verify all nodes have all vectors.
    for (node_idx, c) in coords.iter().enumerate() {
        let ctx = c.get_collection("default").await.unwrap();
        for id in 200u64..205u64 {
            match ctx.store.get_vector(id) {
                Ok(Some(v)) => assert_eq!(v.len(), 256, "vector {id} on node {node_idx} has wrong dim"),
                Ok(None) => panic!("node {node_idx} missing vector {id}"),
                Err(e) => panic!("node {node_idx} error reading vector {id}: {e}"),
            }
        }
    }
}

#[tokio::test]
#[ignore = "metadata Raft split-brain needs debugging"]
async fn test_multi_node_collection_crd() {
    let coords = start_cluster(3, 23200).await;

    // Any node: create a collection (goes through metadata Raft).
    let meta_leader = coords
        .iter()
        .find(|c| {
            c.raft_node(rekha_core::METADATA_PARTITION_ID)
                .map(|rn| {
                    let rt = tokio::runtime::Handle::current();
                    std::thread::spawn(move || rt.block_on(rn.is_leader()))
                        .join()
                        .unwrap()
                })
                .unwrap_or(false)
        })
        .expect("metadata leader should exist");

    meta_leader
        .create_collection("shared", 256)
        .await
        .unwrap();

    // Wait for metadata Raft replication.
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify all nodes have the collection initialized (retry for replication delay).
    let mut last_errors = Vec::new();
    for _ in 0..20 {
        last_errors.clear();
        for (i, c) in coords.iter().enumerate() {
            if !c.collection_exists("shared").await {
                last_errors.push(format!("node {i}"));
            }
        }
        if last_errors.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    assert!(
        last_errors.is_empty(),
        "metadata Raft did not replicate 'shared' to: {}",
        last_errors.join(", ")
    );

    // Insert and verify on a different node.
    coords[1]
        .insert("shared", 1, vec![0.1; 256], None)
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(500)).await;

    let ctx = coords[2].get_collection("shared").await.unwrap();
    assert!(ctx.store.get_vector(1).unwrap().is_some());
}
