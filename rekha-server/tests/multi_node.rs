// Multi-node integration tests for Rekha's distributed write/read path.
// These tests verify replication, read fan-out, failover, and LWW resolution
// across real gRPC servers and coordinator instances.
//
// RULE: If a test fails, fix the race or the underlying bug.
// Do NOT add `#[ignore]` to a failing test — that defeats the purpose.
// If the test harness is the problem, fix the harness.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rekha_cluster::chord::hash_to_chord_id;
use rekha_cluster::Membership;
use rekha_coordinator::{Coordinator, PeerPool};
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, SearchParams};
use rekha_server::RekhaService;
use rekha_storage::RekhaStore;
use tempfile::TempDir;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

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

struct TestServer {
    _dir: Option<TempDir>,
    port: u16,
    handle: Option<tokio::task::JoinHandle<()>>,
    cancel: CancellationToken,
    pub coordinator: Arc<Coordinator>,
}

impl TestServer {
    pub async fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            handle.abort();
            let _ = handle.await;
        }
    }

    pub async fn restart(mut self) -> TestServer {
        self.stop().await;
        tokio::time::sleep(Duration::from_millis(200)).await;

        let coordinator = self.coordinator.clone();
        let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = tcp_listener.local_addr().unwrap().port();

        let service = RekhaService::new(coordinator.clone());
        let cancel = CancellationToken::new();
        let _server_cancel = cancel.clone();

        let handle = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(rekha_proto::proto::rekha_server::RekhaServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(tcp_listener))
                .await
                .ok();
        });

        let ready_addr = format!("127.0.0.1:{}", port);
        for _ in 0..50 {
            if std::net::TcpStream::connect(&ready_addr).is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        TestServer {
            _dir: self._dir.take(),
            port,
            handle: Some(handle),
            cancel,
            coordinator,
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            handle.abort();
            // block_in_place releases the current worker thread so we can
            // block on the task handle. Without this, RocksDB file descriptors
            // leak until the runtime reclaims the aborted task.
            let _ = tokio::task::block_in_place(move || {
                if let Ok(rt) = tokio::runtime::Handle::try_current() {
                    rt.block_on(handle)
                } else {
                    Ok(())
                }
            });
        }
    }
}

async fn start_server(name: &str, seed_addrs: &[String]) -> TestServer {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(Membership::new(name, 5000)));
    let node_id: u64 = {
        let mut h = std::hash::DefaultHasher::new();
        name.hash(&mut h);
        h.finish()
    };

    let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = tcp_listener.local_addr().unwrap().port();
    let listen_addr = format!("127.0.0.1:{}", port);

    let chord = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(format!("{}:{}", name, &listen_addr).as_bytes()),
        &listen_addr,
    ));
    chord.set_successor(name, &listen_addr);

    // Pre-populate successor list with all seed addresses (production does this too)
    for seed in seed_addrs {
        if seed != &listen_addr {
            let mut sl = chord.successor_list.write().await;
            if !sl.contains(seed) {
                sl.push(seed.clone());
                chord.successor_addresses.write().await.push(seed.clone());
            }
        }
    }

    let peer_pool = Arc::new(PeerPool::new());
    let coordinator = Arc::new(Coordinator::new(
        store,
        membership,
        node_id,
        name.to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord,
        peer_pool,
        86400,
    ));
    coordinator.initialize().await.unwrap();

    let service = RekhaService::new(coordinator.clone());
    let cancel = CancellationToken::new();
    let _server_cancel = cancel.clone();

    let handle = tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(rekha_proto::proto::rekha_server::RekhaServer::new(service))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(tcp_listener))
            .await
            .ok();
    });

    // Readiness check: wait up to 5s for the port to accept TCP connections
    let ready_addr = format!("127.0.0.1:{}", port);
    for _ in 0..50 {
        if std::net::TcpStream::connect(&ready_addr).is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    TestServer {
        _dir: Some(dir),
        port,
        handle: Some(handle),
        cancel,
        coordinator,
    }
}

/// Wait for a coordinator's chord successor list to reach `min_count` entries.
#[allow(dead_code)]
async fn wait_for_successors(
    coordinator: &Coordinator,
    min_count: usize,
    timeout: Duration,
) -> bool {
    let start = Instant::now();
    loop {
        let sl = coordinator.chord.successor_list.read().await;
        if sl.len() >= min_count {
            return true;
        }
        drop(sl);
        if start.elapsed() >= timeout {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn add_successor(coordinator: &Coordinator, name: &str, addr: &str) {
    let mut sl = coordinator.chord.successor_list.write().await;
    if !sl.contains(&name.to_string()) {
        sl.push(name.to_string());
        coordinator
            .chord
            .successor_addresses
            .write()
            .await
            .push(addr.to_string());
    }
}

// ── Tests ──────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_single_node_insert_and_search() {
    let server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("test", ivf_config(4))
        .await
        .unwrap();
    client
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let results = client
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty(), "search should return results");
    assert_eq!(results[0].id, 1, "nearest vector should be id=1");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_single_node_delete() {
    let server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("test", ivf_config(4))
        .await
        .unwrap();
    client
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();
    let results = client
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty());
    client
        .delete("test", &[1], 1001, ConsistencyLevel::One)
        .await
        .unwrap();
    let results = client
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(results.is_empty(), "deleted vector should not be found");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lww_timestamp_ordering() {
    let server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("test", ivf_config(4))
        .await
        .unwrap();
    client
        .insert(
            "test",
            1,
            vec![0.9, 0.9, 0.9, 0.9],
            None,
            2000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();
    client
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();
    let results = client
        .search("test", vec![0.5, 0.5, 0.5, 0.5], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_two_node_replication() {
    let server1 = start_server("node1", &[]).await;
    let addr1 = format!("http://127.0.0.1:{}", server1.port);
    let addr1_val = format!("127.0.0.1:{}", server1.port);

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord_id2 = hash_to_chord_id(b"node2");
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        chord_id2,
        "127.0.0.1:0",
    ));
    let peer_pool2 = Arc::new(PeerPool::new());
    let coord2 = Coordinator::new(
        store2,
        membership2,
        2,
        "node2".to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord2.clone(),
        peer_pool2,
        86400,
    );
    coord2.initialize().await.unwrap();

    chord2.set_successor("node1", &addr1_val);
    chord2
        .successor_list
        .write()
        .await
        .push("node1".to_string());
    chord2.successor_addresses.write().await.push(addr1_val);

    coord2
        .create_collection(
            "test",
            ivf_config(4),
            "node2",
            1000,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    let colls = client1.list_collections().await.unwrap();
    assert!(
        colls.contains(&"test".to_string()),
        "collection should be replicated to server1"
    );

    coord2
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            "node2",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let results = client1
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "replicated data should be visible on server1"
    );

    let results2 = coord2
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !results2.is_empty(),
        "data should be visible on coordinator 2"
    );
    assert_eq!(results2[0].id, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_replicated_across_nodes() {
    let server1 = start_server("node1", &[]).await;
    let addr1 = format!("http://127.0.0.1:{}", server1.port);
    let addr1_val = format!("127.0.0.1:{}", server1.port);

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(b"node2"),
        "127.0.0.1:0",
    ));
    let peer_pool2 = Arc::new(PeerPool::new());
    let coord2 = Coordinator::new(
        store2,
        membership2,
        2,
        "node2".to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord2.clone(),
        peer_pool2,
        86400,
    );
    coord2.initialize().await.unwrap();

    chord2
        .successor_list
        .write()
        .await
        .push("node1".to_string());
    chord2.successor_addresses.write().await.push(addr1_val);

    coord2
        .create_collection(
            "crosscol",
            ivf_config(4),
            "node2",
            1000,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    let names = client1.list_collections().await.unwrap();
    assert!(
        names.contains(&"crosscol".to_string()),
        "collection should be replicated to server1"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_replicated_across_nodes() {
    let server1 = start_server("node1", &[]).await;
    let addr1 = format!("http://127.0.0.1:{}", server1.port);
    let addr1_val = format!("127.0.0.1:{}", server1.port);

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(b"node2"),
        "127.0.0.1:0",
    ));
    let peer_pool2 = Arc::new(PeerPool::new());
    let coord2 = Coordinator::new(
        store2,
        membership2,
        2,
        "node2".to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord2.clone(),
        peer_pool2,
        86400,
    );
    coord2.initialize().await.unwrap();

    chord2
        .successor_list
        .write()
        .await
        .push("node1".to_string());
    chord2
        .successor_addresses
        .write()
        .await
        .push(addr1_val.clone());

    coord2
        .create_collection(
            "test",
            ivf_config(4),
            "node2",
            1000,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();
    coord2
        .insert(
            "test",
            42,
            vec![0.5, 0.6, 0.7, 0.8],
            None,
            1000,
            "node2",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    let results = client1
        .search("test", vec![0.5, 0.6, 0.7, 0.8], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "replicated insert should be searchable on server1"
    );
    assert_eq!(results[0].id, 42);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delete_replicated_across_nodes() {
    let server1 = start_server("node1", &[]).await;
    let addr1 = format!("http://127.0.0.1:{}", server1.port);
    let addr1_val = format!("127.0.0.1:{}", server1.port);

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    client1
        .create_collection("test", ivf_config(4))
        .await
        .unwrap();
    client1
        .insert(
            "test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(b"node2"),
        "127.0.0.1:0",
    ));
    let peer_pool2 = Arc::new(PeerPool::new());
    let coord2 = Coordinator::new(
        store2,
        membership2,
        2,
        "node2".to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord2.clone(),
        peer_pool2,
        86400,
    );
    coord2.initialize().await.unwrap();

    chord2
        .successor_list
        .write()
        .await
        .push("node1".to_string());
    chord2.successor_addresses.write().await.push(addr1_val);

    coord2
        .delete("test", &[1], 1001, "node2", ConsistencyLevel::One, false)
        .await
        .unwrap();

    let results = client1
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        results.is_empty(),
        "deleted vector should not be on server1 after replicated delete"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_fanout_merges_from_remote() {
    let server1 = start_server("node1", &[]).await;
    let addr1 = format!("http://127.0.0.1:{}", server1.port);
    let addr1_val = format!("127.0.0.1:{}", server1.port);

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(b"node2"),
        "127.0.0.1:0",
    ));
    let peer_pool2 = Arc::new(PeerPool::new());
    let coord2 = Coordinator::new(
        store2,
        membership2,
        2,
        "node2".to_string(),
        false,
        3600,
        ConsistencyLevel::Quorum,
        3,
        chord2.clone(),
        peer_pool2,
        86400,
    );
    coord2.initialize().await.unwrap();

    chord2
        .successor_list
        .write()
        .await
        .push("node1".to_string());
    chord2
        .successor_addresses
        .write()
        .await
        .push(addr1_val.clone());

    coord2
        .create_collection(
            "test",
            ivf_config(4),
            "node2",
            1000,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();
    coord2
        .insert(
            "test",
            10,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            "node2",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    client1
        .insert(
            "test",
            20,
            vec![0.9, 0.8, 0.7, 0.6],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let results = client1
        .search(
            "test",
            vec![0.1, 0.2, 0.3, 0.4],
            5,
            SearchParams {
                local_only: false,
                ..SearchParams::default()
            },
        )
        .await
        .unwrap();
    assert!(!results.is_empty(), "fan-out search should return results");
    assert_eq!(
        results[0].id, 10,
        "nearest vector should be the exact match from coord2's replica"
    );
}

// ── 3-node tests (mirror e2e_prod.sh) ──────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_3_node_replication() {
    let s1 = start_server("node1", &[]).await;
    let a1 = format!("http://127.0.0.1:{}", s1.port);
    let a1v = format!("127.0.0.1:{}", s1.port);

    let s2 = start_server("node2", &[a1v.clone()]).await;
    let a2 = format!("http://127.0.0.1:{}", s2.port);
    let a2v = format!("127.0.0.1:{}", s2.port);

    let s3 = start_server("node3", &[a1v.clone(), a2v.clone()]).await;
    let a3 = format!("http://127.0.0.1:{}", s3.port);

    // Cross-populate all 3 nodes so each forwards to the other two
    add_successor(&s1.coordinator, "node2", &a2v).await;
    add_successor(&s1.coordinator, "node3", &a3).await;
    add_successor(&s2.coordinator, "node3", &a3).await;

    // Create collection + insert on node-1 (gets forwarded to node-2, node-3)
    let mut c1 = rekha_client::Client::connect(&a1).await.unwrap();
    c1.create_collection("images", ivf_config(8)).await.unwrap();
    c1.insert("images", 1, vec![0.1; 8], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();

    // Search from all 3 nodes — each should find the data
    for (name, addr) in [("node1", &a1), ("node2", &a2), ("node3", &a3)] {
        let mut client = rekha_client::Client::connect(addr).await.unwrap();
        let results = client
            .search("images", vec![0.1; 8], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty(), "{} should return results", name);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_failover_after_node_stop() {
    let mut s1 = start_server("node1", &[]).await;
    let a1 = format!("http://127.0.0.1:{}", s1.port);
    let a1v = format!("127.0.0.1:{}", s1.port);

    let s2 = start_server("node2", &[a1v.clone()]).await;
    let a2 = format!("http://127.0.0.1:{}", s2.port);
    let a2v = format!("127.0.0.1:{}", s2.port);

    // Cross-populate: node-1 forwards to node-2
    add_successor(&s1.coordinator, "node2", &a2v).await;

    // Insert on node-1 — gets forwarded to node-2
    let mut c1 = rekha_client::Client::connect(&a1).await.unwrap();
    c1.create_collection("images", ivf_config(8)).await.unwrap();
    c1.insert("images", 1, vec![0.1; 8], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();

    // Stop node-1 (failover)
    s1.stop().await;
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Search from node-2 — should still work (data was forwarded)
    let mut c2 = rekha_client::Client::connect(&a2).await.unwrap();
    let results = c2
        .search("images", vec![0.1; 8], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "search should work after node-1 failure"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_node_restart_serves_data() {
    let s1 = start_server("node1", &[]).await;
    let a1 = format!("http://127.0.0.1:{}", s1.port);

    // Insert data
    let mut c1 = rekha_client::Client::connect(&a1).await.unwrap();
    c1.create_collection("images", ivf_config(8)).await.unwrap();
    c1.insert("images", 1, vec![0.1; 8], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();

    // Stop and restart node-1
    let s1_restarted = s1.restart().await;
    let new_addr = format!("http://127.0.0.1:{}", s1_restarted.port);

    let mut c1r = rekha_client::Client::connect(&new_addr).await.unwrap();
    let results = c1r
        .search("images", vec![0.1; 8], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty(), "data should survive restart");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_collection_cross_node() {
    let s1 = start_server("node1", &[]).await;
    let a1 = format!("http://127.0.0.1:{}", s1.port);
    let a1v = format!("127.0.0.1:{}", s1.port);

    let s2 = start_server("node2", &[a1v.clone()]).await;
    let a2 = format!("http://127.0.0.1:{}", s2.port);
    let a2v = format!("127.0.0.1:{}", s2.port);

    let s3 = start_server("node3", &[a1v, a2v.clone()]).await;
    let a3 = format!("http://127.0.0.1:{}", s3.port);

    // Cross-populate so collections are replicated to node-3
    add_successor(&s1.coordinator, "node2", &a2v).await;
    add_successor(&s1.coordinator, "node3", &a3).await;
    add_successor(&s2.coordinator, "node3", &a3).await;

    // Create 'images' (dim=8) on node-1 — forwarded to 2,3
    let mut c1 = rekha_client::Client::connect(&a1).await.unwrap();
    c1.create_collection("images", ivf_config(8)).await.unwrap();
    c1.insert("images", 1, vec![0.1; 8], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();

    // Create 'texts' (dim=4) on node-2 — forwarded to 3
    let mut c2 = rekha_client::Client::connect(&a2).await.unwrap();
    c2.create_collection("texts", ivf_config(4)).await.unwrap();
    c2.insert("texts", 1, vec![0.5; 4], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();

    // Search 'images' from node-3
    let mut c3 = rekha_client::Client::connect(&a3).await.unwrap();
    let r_images = c3
        .search("images", vec![0.1; 8], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !r_images.is_empty(),
        "images collection searchable from node-3"
    );

    // Search 'texts' from node-3
    let r_texts = c3
        .search("texts", vec![0.5; 4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(
        !r_texts.is_empty(),
        "texts collection searchable from node-3"
    );
}
