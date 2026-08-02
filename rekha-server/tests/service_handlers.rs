use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use rekha_cluster::chord::hash_to_chord_id;
use rekha_cluster::Membership;
use rekha_coordinator::{Coordinator, PeerPool};
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig};
use rekha_proto::proto::{self, rekha_client::RekhaClient as ProtoRekhaClient};
use rekha_server::RekhaService;
use rekha_storage::RekhaStore;
use tempfile::TempDir;
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

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
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.handle.take() {
            handle.abort();
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

async fn start_server(name: &str, _seed_addrs: &[String]) -> TestServer {
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

async fn raw_client(port: u16) -> ProtoRekhaClient<Channel> {
    let addr = format!("http://127.0.0.1:{}", port);
    ProtoRekhaClient::connect(addr).await.unwrap()
}

// ── Collection management handlers ──────────────────────────────────────────────

#[tokio::test]
async fn test_drop_collection_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("test_drop", ivf_config(4))
        .await
        .unwrap();

    let result = client.drop_collection("test_drop").await;
    assert!(result.is_ok());
    let dropped = result.unwrap();
    assert!(dropped);

    let exists = client.collection_exists("test_drop").await.unwrap();
    assert!(!exists);

    server.stop().await;
}

#[tokio::test]
async fn test_collection_exists_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();

    let exists_before = client.collection_exists("nonexistent").await.unwrap();
    assert!(!exists_before);

    client
        .create_collection("test_exists", ivf_config(4))
        .await
        .unwrap();

    let exists_after = client.collection_exists("test_exists").await.unwrap();
    assert!(exists_after);

    server.stop().await;
}

#[tokio::test]
async fn test_list_collections_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("col_a", ivf_config(4))
        .await
        .unwrap();
    client
        .create_collection("col_b", ivf_config(4))
        .await
        .unwrap();

    let cols = client.list_collections().await.unwrap();
    assert!(cols.iter().any(|c| c == "default"));
    assert!(cols.iter().any(|c| c == "col_a"));
    assert!(cols.iter().any(|c| c == "col_b"));

    server.stop().await;
}

// ── Point data handlers ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_fetch_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("test_fetch", ivf_config(4))
        .await
        .unwrap();

    client
        .insert(
            "test_fetch",
            42,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000u64,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let fetched = client.fetch("test_fetch", &[42], true).await.unwrap();
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].id, 42);

    server.stop().await;
}

#[tokio::test]
async fn test_health_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    let ok = client.health().await.unwrap();
    assert!(ok);

    server.stop().await;
}

// ── Streaming RPCs ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search_stream_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut raw = raw_client(server.port).await;
    let coord = server.coordinator.clone();

    coord
        .create_collection(
            "stream_test",
            ivf_config(4),
            "node1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "stream_test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            "node1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();
    coord
        .insert(
            "stream_test",
            2,
            vec![0.9, 0.8, 0.7, 0.6],
            None,
            1000,
            "node1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();
    coord
        .insert(
            "stream_test",
            3,
            vec![0.5, 0.5, 0.5, 0.5],
            None,
            1000,
            "node1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let req = tonic::Request::new(proto::SearchRequest {
        collection_name: "stream_test".to_string(),
        query_vector: vec![0.1, 0.2, 0.3, 0.4],
        top_k: 5,
        local_only: true,
        consistency: proto::ConsistencyLevel::Quorum as i32,
        params: Some(proto::SearchParams {
            ef_search: 0,
            nprobe: 4,
            include_payloads: false,
        }),
    });

    let resp = raw.search_stream(req).await.unwrap().into_inner();
    let mut stream = resp;
    let mut count = 0;
    while let Some(_item) = stream.next().await {
        count += 1;
    }
    assert!(
        count >= 1,
        "expected at least 1 streamed result, got {}",
        count
    );

    server.stop().await;
}

#[tokio::test]
async fn test_insert_batch_handler() {
    let mut server = start_server("node1", &[]).await;

    let mut raw = raw_client(server.port).await;
    let coord = server.coordinator.clone();

    coord
        .create_collection(
            "batch_test",
            ivf_config(4),
            "node1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    let (tx, rx) = mpsc::channel(4);
    tokio::spawn(async move {
        for i in 1..=5u64 {
            let req = proto::InsertRequest {
                id: i,
                vector: vec![0.1 * i as f32, 0.2, 0.3, 0.4],
                payload: None,
                collection_name: "batch_test".to_string(),
                is_replication: false,
                timestamp: 1000 + i,
                consistency: proto::ConsistencyLevel::One as i32,
                origin_node_id: "test".to_string(),
            };
            tx.send(req).await.unwrap();
        }
    });

    let req = tonic::Request::new(tokio_stream::wrappers::ReceiverStream::new(rx));
    let resp = raw.insert_batch(req).await.unwrap().into_inner();
    assert_eq!(resp.inserted_count, 5);
    assert!(resp.errors.is_empty());

    server.stop().await;
}

#[tokio::test]
async fn test_import_handler() {
    let mut server = start_server("node1", &[]).await;

    let mut raw = raw_client(server.port).await;
    let coord = server.coordinator.clone();

    coord
        .create_collection(
            "import_test",
            ivf_config(4),
            "node1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    let requests = vec![
        proto::InsertRequest {
            id: 10,
            vector: vec![0.1, 0.2, 0.3, 0.4],
            payload: None,
            collection_name: "import_test".to_string(),
            is_replication: false,
            timestamp: 1000,
            consistency: proto::ConsistencyLevel::One as i32,
            origin_node_id: "test".to_string(),
        },
        proto::InsertRequest {
            id: 11,
            vector: vec![0.5, 0.6, 0.7, 0.8],
            payload: None,
            collection_name: "import_test".to_string(),
            is_replication: false,
            timestamp: 1001,
            consistency: proto::ConsistencyLevel::One as i32,
            origin_node_id: "test".to_string(),
        },
    ];

    let chunk = proto::ImportChunk { requests };
    let (tx, rx) = mpsc::channel(1);
    tx.send(chunk).await.unwrap();
    drop(tx);

    let req = tonic::Request::new(tokio_stream::wrappers::ReceiverStream::new(rx));
    let resp = raw.import(req).await.unwrap().into_inner();
    assert_eq!(resp.inserted_count, 2);

    server.stop().await;
}

#[tokio::test]
async fn test_import_stream_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("import_stream_test", ivf_config(4))
        .await
        .unwrap();

    let stream = tokio_stream::iter(vec![
        vec![
            proto::InsertRequest {
                id: 100,
                vector: vec![0.1; 4],
                payload: None,
                collection_name: "import_stream_test".to_string(),
                is_replication: false,
                timestamp: 1000,
                consistency: proto::ConsistencyLevel::One as i32,
                origin_node_id: "test".to_string(),
            },
            proto::InsertRequest {
                id: 101,
                vector: vec![0.2; 4],
                payload: None,
                collection_name: "import_stream_test".to_string(),
                is_replication: false,
                timestamp: 1001,
                consistency: proto::ConsistencyLevel::One as i32,
                origin_node_id: "test".to_string(),
            },
        ],
        vec![proto::InsertRequest {
            id: 102,
            vector: vec![0.3; 4],
            payload: None,
            collection_name: "import_stream_test".to_string(),
            is_replication: false,
            timestamp: 1002,
            consistency: proto::ConsistencyLevel::One as i32,
            origin_node_id: "test".to_string(),
        }],
    ]);

    let resp = client.import_stream(stream).await.unwrap();
    assert_eq!(resp.inserted_count, 3);

    server.stop().await;
}

#[tokio::test]
async fn test_export_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("export_test", ivf_config(4))
        .await
        .unwrap();

    client
        .insert(
            "export_test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let exported = client
        .export("export_test", 0, 100, true, true)
        .await
        .unwrap();
    assert_eq!(exported.len(), 1);

    server.stop().await;
}

#[tokio::test]
async fn test_export_stream_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    client
        .create_collection("export_stream_test", ivf_config(4))
        .await
        .unwrap();

    client
        .insert(
            "export_stream_test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();
    client
        .insert(
            "export_stream_test",
            2,
            vec![0.9, 0.8, 0.7, 0.6],
            None,
            1001,
            ConsistencyLevel::One,
        )
        .await
        .unwrap();

    let mut stream = client
        .export_stream("export_stream_test", 0, 100, true, true)
        .await
        .unwrap();

    let mut count = 0;
    while let Some(result) = stream.next().await {
        let ev = result.unwrap();
        count += 1;
        assert!(ev.id == 1 || ev.id == 2);
    }
    assert_eq!(count, 2);

    server.stop().await;
}

// ── Transfer / repair handlers ─────────────────────────────────────────────────

#[tokio::test]
async fn test_transfer_shard_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let coord = server.coordinator.clone();
    coord
        .create_collection(
            "transfer_test",
            ivf_config(4),
            "node1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    for i in 1..=5u64 {
        coord
            .insert(
                "transfer_test",
                i,
                vec![0.1 * i as f32; 4],
                None,
                1000 + i as i64,
                "node1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
    }

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    let chunks = client
        .transfer_shard("transfer_test", "self")
        .await
        .unwrap();

    assert!(
        !chunks.is_empty(),
        "expected at least one transfer shard chunk"
    );

    server.stop().await;
}

#[tokio::test]
async fn test_repair_collection_handler() {
    let mut server = start_server("node1", &[]).await;
    let addr = format!("http://127.0.0.1:{}", server.port);

    let coord = server.coordinator.clone();
    coord
        .create_collection(
            "repair_test",
            ivf_config(4),
            "node1",
            0,
            ConsistencyLevel::Quorum,
            false,
        )
        .await
        .unwrap();

    coord
        .insert(
            "repair_test",
            1,
            vec![0.1, 0.2, 0.3, 0.4],
            None,
            1000,
            "node1",
            ConsistencyLevel::One,
            false,
        )
        .await
        .unwrap();

    let mut client = rekha_client::Client::connect(&addr).await.unwrap();
    let progress = client.repair_collection("repair_test").await.unwrap();

    assert!(
        !progress.is_empty(),
        "expected at least one repair progress item"
    );

    server.stop().await;
}

// ── Cluster / Chord handlers (via raw proto client) ───────────────────────────

#[tokio::test]
async fn test_handshake_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::HandshakeRequest {
        node_id: "peer1".to_string(),
        address: format!("127.0.0.1:{}", server.port),
    });

    let resp = raw.handshake(req).await.unwrap().into_inner();
    assert_eq!(resp.cluster_id, "rekha-cluster");

    server.stop().await;
}

#[tokio::test]
async fn test_heartbeat_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::HeartbeatRequest {
        node_id: "heartbeat_node".to_string(),
        address: format!("127.0.0.1:{}", server.port),
        storage_bytes: 1024,
    });

    let resp = raw.heartbeat(req).await.unwrap().into_inner();
    assert!(resp.success);

    server.stop().await;
}

#[tokio::test]
async fn test_find_successor_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let id_bytes = 42u128.to_le_bytes().to_vec();
    let req = tonic::Request::new(proto::FindSuccessorRequest { id: id_bytes });

    let resp = raw.find_successor(req).await.unwrap().into_inner();
    assert!(resp.successor.is_some());

    server.stop().await;
}

#[tokio::test]
async fn test_get_predecessor_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::GetPredecessorRequest {});

    let resp = raw.get_predecessor(req).await.unwrap().into_inner();
    let _ = resp.predecessor;

    server.stop().await;
}

#[tokio::test]
async fn test_notify_chord_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::NotifyChordRequest {
        node: Some(proto::NodeInfo {
            node_id: "neighbor".to_string(),
            address: "127.0.0.1:59999".to_string(),
            partition_id: 0,
            dim_groups: vec![],
            storage_bytes: 0,
            status: "alive".to_string(),
        }),
    });

    let resp = raw.notify_chord(req).await.unwrap().into_inner();
    let _ = resp.success;

    server.stop().await;
}

#[tokio::test]
async fn test_get_successor_list_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::GetSuccessorListRequest {});

    let resp = raw.get_successor_list(req).await.unwrap().into_inner();
    assert!(!resp.successors.is_empty() || resp.successors.is_empty());

    server.stop().await;
}

#[tokio::test]
async fn test_search_dim_range_handler() {
    let mut server = start_server("node1", &[]).await;
    let mut raw = raw_client(server.port).await;

    let req = tonic::Request::new(proto::SearchDimRangeRequest {
        collection_name: "test".to_string(),
        query_vector: vec![0.0; 4],
        top_k: 10,
        dim_start: 0,
        dim_end: 4,
        nprobe: 4,
    });

    let resp = raw.search_dim_range(req).await;
    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), tonic::Code::Unimplemented);

    server.stop().await;
}
