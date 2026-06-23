use rekha_client::RekhaClient;
use rekha_core::{ConsistencyLevel, NodeInfo, NodeStatus, ScoredPoint, SearchParams, VectorStoreBackend};
use rekha_coordinator::{Coordinator, CoordinatorConfig};
use rekha_index::RekhaIndex;
use rekha_storage::RocksVectorStore;
use rekha_server::proto::rekha_server::RekhaServer as RekhaGrpcServer;
use rekha_server::service::RekhaService;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::oneshot;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

struct TestNode {
    node_id: String,
    addr: String,
    coordinator: Arc<Coordinator>,
    _temp_dir: TempDir,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl TestNode {
    async fn start(node_id: &str, peer_timeout_ms: u64) -> Self {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path();

        let store = Arc::new(RocksVectorStore::open(data_dir).unwrap());
        let config = CoordinatorConfig {
            node_id: node_id.to_string(),
            bind_addr: "0.0.0.0:0".into(),
            seed_nodes: vec![],
            default_write_consistency: "QUORUM".into(),
            hinted_handoff_enabled: true,
            max_hint_window_secs: 10800,
            gc_grace_seconds: 864000,
            peer_timeout_ms,
        };
        let coordinator = Arc::new(Coordinator::new(config, store));
        coordinator.initialize(RekhaIndex::new().unwrap()).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let actual_addr = listener.local_addr().unwrap().to_string();

        let service = RekhaService::new(coordinator.clone());
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        let incoming = TcpListenerStream::new(listener);
        tokio::spawn(async move {
            Server::builder()
                .add_service(RekhaGrpcServer::new(service))
                .serve_with_incoming_shutdown(incoming, async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .ok();
        });

        // Wait for the server to accept connections
        for _ in 0..50 {
            if RekhaClient::connect(&[actual_addr.clone()]).await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        Self {
            node_id: node_id.to_string(),
            addr: actual_addr,
            coordinator,
            _temp_dir: temp_dir,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    fn node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.node_id.clone(),
            address: self.addr.clone(),
            partition_id: 0,
            dim_groups: vec![0, 1, 2, 3],
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        }
    }

    async fn client(&self) -> RekhaClient {
        RekhaClient::connect(&[self.addr.clone()]).await.unwrap()
    }
}

impl Drop for TestNode {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

async fn register_peers(nodes: &[&TestNode]) {
    for (i, node) in nodes.iter().enumerate() {
        for (j, other) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }
            node.coordinator.register_peer(other.node_info()).await;
        }
    }
}

async fn create_coll(client: &RekhaClient, name: &str, dim: u32, rf: u64) {
    client
        .create_collection(name, dim, 16, 4, rf, ConsistencyLevel::One)
        .await
        .unwrap();
}

async fn insert_vec(client: &RekhaClient, coll: &str, dim: u32, cl: ConsistencyLevel) -> u64 {
    client
        .insert(0, vec![0.5f32; dim as usize], coll, None, cl)
        .await
        .unwrap()
}

async fn search_local(client: &RekhaClient, coll: &str, dim: u32, k: usize) -> Vec<ScoredPoint> {
    let params = SearchParams {
        ef_search: 64,
        nprobe: 4,
        include_payloads: false,
        local_only: true,
    };
    let (results, _) = client
        .search_with_params(vec![0.5f32; dim as usize], coll, k, params, ConsistencyLevel::One)
        .await
        .unwrap();
    results
}

// ============================================================
// Tests
// ============================================================

#[tokio::test]
async fn test_two_node_replication() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 2).await;
    insert_vec(&c1, "test", 4, ConsistencyLevel::One).await;

    let c2 = n2.client().await;
    let results = search_local(&c2, "test", 4, 5).await;
    assert!(!results.is_empty(), "node-2 should find the vector replicated from node-1");
}

#[tokio::test]
async fn test_three_node_quorum_insert() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    let n3 = TestNode::start("n3", 10000).await;
    register_peers(&[&n1, &n2, &n3]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 3).await;

    let id = c1
        .insert(0, vec![0.5; 4], "test", None, ConsistencyLevel::Quorum)
        .await
        .unwrap();
    assert!(id > 0, "QUORUM insert with 3 nodes should succeed (needs 2/3 acks)");
}

#[tokio::test]
async fn test_consistency_all_fails_on_node_down() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 3).await;

    let result = c1
        .insert(0, vec![0.5; 4], "test", None, ConsistencyLevel::All)
        .await;
    assert!(result.is_err(), "ALL with 2/3 nodes should fail");
}

#[tokio::test]
async fn test_failover_search() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    let n3 = TestNode::start("n3", 10000).await;
    register_peers(&[&n1, &n2, &n3]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 3).await;
    for _ in 0..10 {
        insert_vec(&c1, "test", 4, ConsistencyLevel::One).await;
    }

    let c3 = n3.client().await;
    let results = search_local(&c3, "test", 4, 5).await;
    assert!(!results.is_empty(), "node-3 should have replicas of the data");
}

#[tokio::test]
async fn test_hinted_handoff_stores_hints() {
    let n1 = TestNode::start("n1", 10000).await;

    // Register n2 as a peer BEFORE starting n2's server
    // This simulates the scenario where n2 is down at registration time,
    // so PeerPool can't connect to it — but n2 is still Healthy in Membership.
    let n2_info = NodeInfo {
        node_id: "n2".into(),
        address: "127.0.0.1:1".into(), // non-routable address — will fail to connect
        partition_id: 0,
        dim_groups: vec![0, 1, 2, 3],
        is_leader: false,
        raft_term: 0,
        commit_index: 0,
        storage_bytes: 0,
        status: NodeStatus::Healthy,
        last_heartbeat: 0,
    };
    n1.coordinator.register_peer(n2_info).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 2).await;

    // Insert — n2 is Healthy (in Membership ring) but not in PeerPool (connect failed),
    // so the fan-out will try and store a hint when it can't find the client in the pool.
    let id = n1.coordinator
        .insert("test", 0, vec![0.5; 4], None, 0, ConsistencyLevel::One)
        .await
        .unwrap();
    assert!(id > 0);

    // Verify a hint was stored for n2
    let hint_store = n1.coordinator.store().hint_store();
    let hints = hint_store.iter_hints_for_node("n2").unwrap();
    assert!(!hints.is_empty(), "insert with unreachable gRPC peer should store a hint");
    assert_eq!(hints[0].id, id);
}

#[tokio::test]
async fn test_cross_node_search_merge() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 2).await;
    for _ in 0..5 {
        insert_vec(&c1, "test", 4, ConsistencyLevel::One).await;
    }

    let params = SearchParams {
        ef_search: 64,
        nprobe: 4,
        include_payloads: false,
        local_only: false,
    };
    let (results, _) = c1
        .search_with_params(vec![0.5; 4], "test", 10, params, ConsistencyLevel::One)
        .await
        .unwrap();
    assert!(!results.is_empty(), "cross-node search should return merged results");
}

#[tokio::test]
async fn test_collection_ddl_replicated() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "shared", 4, 2).await;

    let colls = n2.client().await.list_collections().await.unwrap();
    assert!(colls.contains(&"shared".to_string()), "collection metadata should appear on n2");
}

#[tokio::test]
async fn test_multi_collection_different_dims() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "dim4", 4, 2).await;
    create_coll(&c1, "dim8", 8, 2).await;

    insert_vec(&c1, "dim4", 4, ConsistencyLevel::One).await;
    insert_vec(&c1, "dim8", 8, ConsistencyLevel::One).await;

    let c2 = n2.client().await;
    let d4r = search_local(&c2, "dim4", 4, 5).await;
    let d8r = search_local(&c2, "dim8", 8, 5).await;
    assert!(!d4r.is_empty(), "dim=4 collection should work on n2");
    assert!(!d8r.is_empty(), "dim=8 collection should work on n2");
}

#[tokio::test]
async fn test_delete_replicates_to_peer() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "test", 4, 2).await;
    let id = insert_vec(&c1, "test", 4, ConsistencyLevel::One).await;

    // Verify it exists on n2
    let ns = n2.coordinator.store().as_ref().clone().with_namespace("test".into());
    let before = ns.get_vector_record(id).unwrap();
    assert!(before.is_some(), "vector should be replicated to n2");
    assert!(!before.unwrap().is_tombstone);

    // Use coordinator API directly for delete (avoids gRPC timeout issues)
    n1.coordinator
        .delete("test", &[id], 0, ConsistencyLevel::One)
        .await
        .unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify it's tombstoned on n2
    let after = ns.get_vector_record(id).unwrap();
    assert!(after.is_some(), "vector should still have a record on n2");
    assert!(after.unwrap().is_tombstone, "deleted vector should be tombstoned on n2");
}

#[tokio::test]
async fn test_collection_exists_across_nodes() {
    let n1 = TestNode::start("n1", 10000).await;
    let n2 = TestNode::start("n2", 10000).await;
    register_peers(&[&n1, &n2]).await;

    let c1 = n1.client().await;
    create_coll(&c1, "exists_test", 4, 2).await;

    let c2 = n2.client().await;
    assert!(c2.collection_exists("exists_test").await.unwrap());
    assert!(!c2.collection_exists("nonexistent").await.unwrap());
}
