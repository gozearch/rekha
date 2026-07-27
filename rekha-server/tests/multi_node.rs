use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use rekha_cluster::chord::hash_to_chord_id;
use rekha_cluster::Membership;
use rekha_coordinator::{Coordinator, PeerPool};
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, SearchParams};
use rekha_server::RekhaService;
use rekha_storage::RekhaStore;
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

async fn start_server(name: &str) -> (TempDir, u16, tokio::sync::oneshot::Sender<()>) {
    let dir = TempDir::new().unwrap();
    let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
    let membership = Arc::new(RwLock::new(Membership::new(name, 5000)));
    let node_id: u64 = {
        let mut h = std::hash::DefaultHasher::new();
        name.hash(&mut h);
        h.finish()
    };
    let chord = Arc::new(rekha_cluster::chord::ChordNode::new(
        hash_to_chord_id(format!("{}:{}", name, "127.0.0.1:0").as_bytes()),
        "127.0.0.1:0",
    ));
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

    let service = RekhaService::new(coordinator);
    let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = tcp_listener.local_addr().unwrap().port();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    tokio::spawn(async move {
        tokio::select! {
            _ = tonic::transport::Server::builder()
                .add_service(rekha_proto::proto::rekha_server::RekhaServer::new(service))
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(tcp_listener)) => {}
            _ = shutdown_rx => {}
        }
    });

    // Readiness check: wait up to 5s for the port to accept TCP connections
    let ready_addr = format!("127.0.0.1:{}", port);
    for _ in 0..50 {
        if std::net::TcpStream::connect(&ready_addr).is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    (dir, port, shutdown_tx)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_single_node_insert_and_search() {
    let (_dir, port, shutdown) = start_server("node1").await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client.create_collection("test", ivf_config(4)).await.unwrap();
    client
        .insert("test", 1, vec![0.1, 0.2, 0.3, 0.4], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();
    let results = client
        .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
        .await
        .unwrap();
    assert!(!results.is_empty(), "search should return results");
    assert_eq!(results[0].id, 1, "nearest vector should be id=1");
    let _ = shutdown.send(());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_single_node_delete() {
    let (_dir, port, shutdown) = start_server("node1").await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client.create_collection("test", ivf_config(4)).await.unwrap();
    client
        .insert("test", 1, vec![0.1, 0.2, 0.3, 0.4], None, 1000, ConsistencyLevel::One)
        .await
        .unwrap();
    let results = client.search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default()).await.unwrap();
    assert!(!results.is_empty());
    client.delete("test", &[1], 1001, ConsistencyLevel::One).await.unwrap();
    let results = client.search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default()).await.unwrap();
    assert!(results.is_empty(), "deleted vector should not be found");
    let _ = shutdown.send(());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lww_timestamp_ordering() {
    let (_dir, port, shutdown) = start_server("node1").await;
    let addr = format!("http://127.0.0.1:{}", port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client.create_collection("test", ivf_config(4)).await.unwrap();
    client.insert("test", 1, vec![0.9, 0.9, 0.9, 0.9], None, 2000, ConsistencyLevel::One).await.unwrap();
    client.insert("test", 1, vec![0.1, 0.2, 0.3, 0.4], None, 1000, ConsistencyLevel::One).await.unwrap();
    let results = client.search("test", vec![0.5, 0.5, 0.5, 0.5], 5, SearchParams::default()).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 1);
    let _ = shutdown.send(());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_two_node_replication() {
    let (_dir1, port1, shutdown1) = start_server("node1").await;
    let addr1 = format!("http://127.0.0.1:{}", port1);

    let mut client1 = rekha_client::Client::connect(&addr1).await.unwrap();
    client1.create_collection("test", ivf_config(4)).await.unwrap();

    let dir2 = TempDir::new().unwrap();
    let store2 = Arc::new(RekhaStore::open(dir2.path().to_str().unwrap()).unwrap());
    let membership2 = Arc::new(RwLock::new(Membership::new("node2", 5000)));
    let chord_id2 = hash_to_chord_id(b"node2");
    let chord2 = Arc::new(rekha_cluster::chord::ChordNode::new(
        chord_id2,
        &format!("127.0.0.1:{}", port1),
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
    coord2.create_collection("test", ivf_config(4), "coord2", 1000, ConsistencyLevel::Quorum, false).await.unwrap();

    chord2.set_successor("node1", &format!("127.0.0.1:{}", port1));
    chord2.successor_list.write().await.push("node1".to_string());
    chord2.successor_addresses.write().await.push(format!("127.0.0.1:{}", port1));

    coord2.insert("test", 1, vec![0.1, 0.2, 0.3, 0.4], None, 1000, "node2", ConsistencyLevel::One, false).await.unwrap();

    let results = client1.search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default()).await.unwrap();
    assert!(!results.is_empty(), "replicated data should be visible on node 1");

    let results2 = coord2.search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default()).await.unwrap();
    assert!(!results2.is_empty(), "data should be visible on node 2");
    assert_eq!(results2[0].id, 1);

    let _ = shutdown1.send(());
}
