//! Leadership is decided by consensus, and writes must follow it.
//!
//! These tests run a real three-node Raft group over an in-process network,
//! with a real HTTP listener per node, and assert the property that motivated
//! C0a: **a write accepted by the cluster is applied on the node consensus
//! elected, never only on whichever node the client happened to reach.**
//!
//! The transport details are deliberately split: Raft RPCs go through
//! `ChannelNetwork` (in-process), while the public API is real HTTP, because
//! forwarding writes is an HTTP-level concern.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use openraft::error::{RPCError, RemoteError};
use rekha_api::{AppState, ServerConfig, router};
use rekha_cluster::network::channel::{ChannelHub, ChannelMessage, ChannelNetworkFactory};
use rekha_cluster::state_machine::ClusterStateMachine;
use rekha_cluster::{MemoryLogStore, PeerDirectory, RaftTypeConfig, parse_peers};
use rekha_core::config::CollectionConfig;
use rekha_core::types::Distance;
use rekha_engine::{Engine, EngineConfig};
use rekha_storage::{LocalStorage, RedbCatalog};
use tokio::sync::mpsc;
use uuid::Uuid;

const SECRET: &str = "leadership-test-secret";

struct Node {
    id: u64,
    raft: Arc<openraft::Raft<RaftTypeConfig>>,
    engine: Arc<Engine>,
    state: Arc<AppState>,
    base: String,
    _dir: tempfile::TempDir,
}

/// Spawn the Raft RPC handler for one node on the in-process hub.
#[allow(deprecated)]
fn spawn_rpc_handler(node_id: u64, hub: &ChannelHub, raft: Arc<openraft::Raft<RaftTypeConfig>>) {
    let (tx, mut rx) = mpsc::channel(256);
    let hub = hub.clone();
    tokio::spawn(async move {
        hub.register(node_id, tx).await;
        while let Some(msg) = rx.recv().await {
            match msg {
                ChannelMessage::AppendEntries { req, resp_tx } => {
                    let result = raft
                        .append_entries(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                ChannelMessage::Vote { req, resp_tx } => {
                    let result = raft
                        .vote(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                ChannelMessage::InstallSnapshot { req, resp_tx } => {
                    let result = raft
                        .install_snapshot(req)
                        .await
                        .map_err(|e| RPCError::RemoteError(RemoteError::new(node_id, e)));
                    let _ = resp_tx.send(result);
                }
                _ => {}
            }
        }
    });
}

fn engine_in(dir: &std::path::Path) -> Arc<Engine> {
    let catalog = Arc::new(RedbCatalog::open(dir.join("catalog.redb")).unwrap());
    let storage = Arc::new(LocalStorage::new(dir.join("objects")));
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    // Cluster mode: the Raft log is the durable record, so the engine applies
    // committed entries without its own WAL.
    Arc::new(
        Engine::open(
            catalog,
            storage,
            &wal_dir,
            EngineConfig {
                apply_mode: rekha_engine::ApplyMode::Committed,
                ..EngineConfig::default()
            },
        )
        .unwrap(),
    )
}

/// Bind every listener first, so the peer directory is known before any node
/// is constructed. This mirrors startup: topology is configuration, and it must
/// be identical and complete on every node.
async fn listeners(n: usize) -> Vec<(tokio::net::TcpListener, u16)> {
    let mut out = Vec::new();
    for _ in 0..n {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        out.push((listener, port));
    }
    out
}

async fn start_cluster(n: usize) -> Vec<Node> {
    let hub = ChannelHub::new();
    let bound = listeners(n).await;

    // Every node sees the same topology string, which is also the invariant the
    // CLI enforces.
    let peers_spec = bound
        .iter()
        .enumerate()
        .map(|(i, (_, port))| format!("{}@127.0.0.1:{port}", i + 1))
        .collect::<Vec<_>>()
        .join(",");
    let peer_list = parse_peers(&peers_spec).unwrap();
    let directory = PeerDirectory::new(&peer_list);

    let mut nodes = Vec::new();
    for (index, (listener, port)) in bound.into_iter().enumerate() {
        let id = (index + 1) as u64;
        let dir = tempfile::tempdir().unwrap();
        let engine = engine_in(dir.path());

        let mut sm = ClusterStateMachine::new();
        sm.set_catalog(engine.catalog());
        sm.set_engine(engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);

        // A fast but honest configuration: short timeouts keep the test under a
        // couple of seconds without weakening the protocol.
        let config = Arc::new(openraft::Config {
            heartbeat_interval: 50,
            election_timeout_min: 200,
            election_timeout_max: 400,
            ..Default::default()
        });

        let raft = Arc::new(
            openraft::Raft::<RaftTypeConfig>::new(
                id,
                config,
                ChannelNetworkFactory::new(hub.clone()),
                MemoryLogStore::new(),
                sm,
            )
            .await
            .expect("raft"),
        );
        spawn_rpc_handler(id, &hub, raft.clone());

        let state = Arc::new(
            AppState::new(
                engine.clone(),
                Some(raft.clone()),
                ServerConfig {
                    tenant: "default_tenant".into(),
                    database: "default_database".into(),
                    api_key: None,
                    cluster_secret: Some(SECRET.into()),
                    pepper: Vec::new(),
                    allow_open_mode: true,
                },
            )
            .with_peer_directory(directory.clone())
            .with_data_via_log(true),
        );

        let app = router(state.clone());
        let base = format!("http://127.0.0.1:{port}");
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });

        nodes.push(Node {
            id,
            raft,
            engine,
            state,
            base,
            _dir: dir,
        });
    }

    // Bootstrap exactly one node, exactly as `--bootstrap` does.
    let bootstrap = nodes[0].raft.clone();
    bootstrap
        .initialize(BTreeSet::from([1u64]))
        .await
        .expect("initialize");
    wait_for_leader(&nodes, Duration::from_secs(5)).await;

    for node in nodes.iter().skip(1) {
        bootstrap.add_learner(node.id, node.id, true).await.unwrap();
    }
    tokio::time::sleep(Duration::from_millis(300)).await;
    bootstrap
        .change_membership(BTreeSet::from([1u64, 2, 3]), true)
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    nodes
}

async fn wait_for_leader(nodes: &[Node], timeout: Duration) -> u64 {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        for node in nodes {
            let m = node.raft.metrics().borrow().clone();
            if m.state.is_leader() {
                return node.id;
            }
        }
        if tokio::time::Instant::now() > deadline {
            panic!("no leader elected within {timeout:?}");
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

/// Create the same collection on every node, so the test exercises the data
/// write path rather than collection creation.
fn seed_collection(nodes: &[Node], cid: Uuid) {
    for node in nodes {
        let mut config = CollectionConfig::new("docs".into(), 4, Distance::L2);
        config.id = cid;
        node.engine.create_collection(&config).unwrap();
    }
}

fn add_body() -> serde_json::Value {
    serde_json::json!({
        "ids": ["a"],
        "embeddings": [[1.0, 0.0, 0.0, 0.0]],
        "metadatas": [{ "k": "v" }],
    })
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_write_to_a_follower_reaches_the_leader_and_replicates() {
    let nodes = start_cluster(3).await;
    let leader = wait_for_leader(&nodes, Duration::from_secs(5)).await;
    let cid = Uuid::new_v4();
    seed_collection(&nodes, cid);

    let follower = nodes.iter().find(|n| n.id != leader).unwrap();
    assert_ne!(follower.id, leader);
    assert_eq!(
        follower.state.peers.len(),
        3,
        "the follower must know every node's address"
    );
    assert!(
        follower.state.peers.url(leader).is_some(),
        "the follower must be able to resolve the leader's address"
    );

    let resp = client()
        .post(format!("{}/api/v2/collections/{cid}/add", follower.base))
        .json(&add_body())
        .send()
        .await
        .expect("request");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "follower must accept the write by forwarding it: {}",
        resp.text().await.unwrap_or_default()
    );

    // The leader applied it.
    let leader_node = nodes.iter().find(|n| n.id == leader).unwrap();
    assert_eq!(
        leader_node.engine.count(&cid).unwrap(),
        1,
        "the leader must hold the acknowledged write"
    );

    // And it replicates. The follower does apply this write, but it applies it
    // from the log, not from the client: the request it received was forwarded.
    // The property that a refused write reaches no store at all is covered by
    // `a_write_without_quorum_is_refused_and_not_applied`.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while follower.engine.count(&cid).unwrap() != 1 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the follower never converged from the log"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

/// A leader that has lost its quorum must not acknowledge writes. Before C0a
/// the write was applied and acknowledged locally, then lost at the next
/// election.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn a_write_without_quorum_is_refused_and_not_applied() {
    let nodes = start_cluster(3).await;
    let leader = wait_for_leader(&nodes, Duration::from_secs(5)).await;
    let cid = Uuid::new_v4();
    seed_collection(&nodes, cid);

    // Isolate the leader: two of three voters are gone, so it cannot commit.
    for node in nodes.iter().filter(|n| n.id != leader) {
        node.raft.shutdown().await.unwrap();
    }

    let leader_node = nodes.iter().find(|n| n.id == leader).unwrap();
    let resp = client()
        .post(format!("{}/api/v2/collections/{cid}/add", leader_node.base))
        .json(&add_body())
        .send()
        .await;

    match resp {
        Ok(resp) => {
            assert!(
                !resp.status().is_success(),
                "a leader without quorum must not acknowledge: {}",
                resp.status()
            );
            assert_eq!(
                resp.status(),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
                "expected 503 with a retry hint"
            );
        }
        Err(e) => panic!("request should have been answered with 503, not {e}"),
    }

    assert_eq!(
        leader_node.engine.count(&cid).unwrap(),
        0,
        "a refused write must not reach the store"
    );
}

/// After the leader dies, writes must reach whoever is elected next.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn writes_follow_the_newly_elected_leader() {
    let nodes = start_cluster(3).await;
    let first = wait_for_leader(&nodes, Duration::from_secs(5)).await;
    let cid = Uuid::new_v4();
    seed_collection(&nodes, cid);

    let old_leader = nodes.iter().find(|n| n.id == first).unwrap();
    old_leader.raft.shutdown().await.unwrap();

    let survivors: Vec<&Node> = nodes.iter().filter(|n| n.id != first).collect();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut new_leader = None;
    while tokio::time::Instant::now() < deadline {
        if let Some(node) = survivors
            .iter()
            .find(|n| n.raft.metrics().borrow().state.is_leader())
        {
            new_leader = Some(node.id);
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    let new_leader = new_leader.expect("a new leader must be elected from the surviving majority");
    assert_ne!(new_leader, first);

    let target = survivors
        .iter()
        .find(|n| n.id != new_leader)
        .expect("a follower survives");

    let resp = client()
        .post(format!("{}/api/v2/collections/{cid}/add", target.base))
        .json(&add_body())
        .send()
        .await
        .expect("request");

    assert!(
        resp.status().is_success(),
        "write must be routed to the new leader: {}",
        resp.text().await.unwrap_or_default()
    );

    let elected = survivors.iter().find(|n| n.id == new_leader).unwrap();
    assert_eq!(
        elected.engine.count(&cid).unwrap(),
        1,
        "the newly elected leader must hold the write"
    );
}
