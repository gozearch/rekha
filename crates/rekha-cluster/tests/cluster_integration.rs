use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use openraft::Config;
use openraft::error::{RPCError, RemoteError};
use openraft::storage::RaftLogStorage;
use tokio::sync::mpsc;

use rekha_cluster::RaftTypeConfig;
use rekha_cluster::log_store::MemoryLogStore;
use rekha_cluster::network::channel::{ChannelHub, ChannelMessage, ChannelNetworkFactory};
use rekha_cluster::raft_types::ClusterOperation;
use rekha_cluster::state_machine::ClusterStateMachine;

async fn create_node(node_id: u64, hub: ChannelHub) -> openraft::Raft<RaftTypeConfig> {
    let log_store = MemoryLogStore::new();
    let sm = ClusterStateMachine::new();
    let network = ChannelNetworkFactory::new(hub);
    let config = Arc::new(Config::default());
    openraft::Raft::<RaftTypeConfig>::new(node_id, config, network, log_store, sm)
        .await
        .expect("Raft initialization failed")
}

#[allow(deprecated)]
async fn spawn_handler(node_id: u64, hub: &ChannelHub, raft: openraft::Raft<RaftTypeConfig>) {
    let (tx, mut rx) = mpsc::channel(64);
    hub.register(node_id, tx).await;
    tokio::spawn(async move {
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
                ChannelMessage::FullSnapshot {
                    vote,
                    meta,
                    data,
                    resp_tx,
                } => {
                    let snapshot = openraft::Snapshot {
                        meta,
                        snapshot: Box::new(std::io::Cursor::new(data)),
                    };
                    let result = raft
                        .install_full_snapshot(vote, snapshot)
                        .await
                        .map_err(|e| {
                            openraft::error::StreamingError::Unreachable(
                                openraft::error::Unreachable::new(&e),
                            )
                        });
                    let _ = resp_tx.send(result);
                }
            }
        }
    });
}

fn voter_ids(metrics: &openraft::RaftMetrics<u64, u64>) -> BTreeSet<u64> {
    metrics.membership_config.voter_ids().collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn single_node_bootstrap() {
    let hub = ChannelHub::new();
    let raft = create_node(1, hub.clone()).await;
    spawn_handler(1, &hub, raft.clone()).await;

    let mut members = BTreeSet::new();
    members.insert(1u64);
    raft.initialize(members).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let m = raft.metrics().borrow().clone();
    assert_eq!(m.id, 1);
    assert!(m.state.is_leader());

    let op = ClusterOperation::AddNode {
        node_id: 1,
        addr: "127.0.0.1:8001".into(),
    };
    raft.client_write(op).await.unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;

    let m = raft.metrics().borrow().clone();
    assert!(m.state.is_leader());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_node_cluster() {
    let hub = ChannelHub::new();
    let raft1 = create_node(1, hub.clone()).await;
    let raft2 = create_node(2, hub.clone()).await;

    spawn_handler(1, &hub, raft1.clone()).await;
    spawn_handler(2, &hub, raft2.clone()).await;

    let mut members = BTreeSet::new();
    members.insert(1u64);
    raft1.initialize(members).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    raft1.add_learner(2, 2u64, true).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let new_members: BTreeSet<u64> = [1u64, 2].into();
    raft1.change_membership(new_members, true).await.unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;

    let m1 = raft1.metrics().borrow().clone();
    assert!(m1.state.is_leader());

    let cid = uuid::Uuid::new_v4();
    let op = ClusterOperation::AddCollection {
        collection_id: cid,
        name: "test".into(),
        dimension: 128,
        distance: "l2".into(),
        tenant: "default".into(),
        database: "default".into(),
        owner_nodes: vec![1, 2],
    };
    raft1.client_write(op).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    let m1 = raft1.metrics().borrow().clone();
    let m2 = raft2.metrics().borrow().clone();
    assert!(m1.state.is_leader());

    let voters1 = voter_ids(&m1);
    let voters2 = voter_ids(&m2);
    assert!(voters1.contains(&1));
    assert!(voters1.contains(&2));
    assert!(voters2.contains(&1));
    assert!(voters2.contains(&2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[allow(deprecated)]
async fn three_node_cluster() {
    let hub = ChannelHub::new();
    let raft1 = create_node(1, hub.clone()).await;
    let raft2 = create_node(2, hub.clone()).await;
    let raft3 = create_node(3, hub.clone()).await;

    spawn_handler(1, &hub, raft1.clone()).await;
    spawn_handler(2, &hub, raft2.clone()).await;
    spawn_handler(3, &hub, raft3.clone()).await;

    let mut members = BTreeSet::new();
    members.insert(1u64);
    raft1.initialize(members).await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;

    raft1.add_learner(2, 2u64, true).await.unwrap();
    raft1.add_learner(3, 3u64, true).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;

    let new_members: BTreeSet<u64> = [1u64, 2, 3].into();
    raft1.change_membership(new_members, true).await.unwrap();
    tokio::time::sleep(Duration::from_millis(150)).await;

    let m = raft1.metrics().borrow().clone();
    assert!(m.state.is_leader());

    for i in 0..5 {
        let op = ClusterOperation::AddNode {
            node_id: 100 + i,
            addr: format!("127.0.0.1:{}", 9000 + i),
        };
        raft1.client_write(op).await.unwrap();
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let m1 = raft1.metrics().borrow().clone();
    let m2 = raft2.metrics().borrow().clone();
    let m3 = raft3.metrics().borrow().clone();

    assert!(m1.state.is_leader());

    for m in [&m2, &m3] {
        let voters = voter_ids(m);
        assert!(voters.contains(&1));
        assert!(voters.contains(&2));
        assert!(voters.contains(&3));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn persistent_raft_survives_restart() {
    let dir = tempfile::tempdir().unwrap();
    let log_path = dir.path().join("raft.redb");

    // First run: create node, bootstrap, write
    {
        let log_store = rekha_cluster::RedbLogStore::open(&log_path).unwrap();
        let sm = ClusterStateMachine::new();
        let hub = ChannelHub::new();
        let network = ChannelNetworkFactory::new(hub.clone());
        let config = Arc::new(Config::default());
        let raft = openraft::Raft::<RaftTypeConfig>::new(1, config, network, log_store, sm)
            .await
            .unwrap();

        let (tx, mut rx) = mpsc::channel(64);
        hub.register(1, tx).await;
        let raft_clone = raft.clone();
        let handle = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match msg {
                    ChannelMessage::AppendEntries { req, resp_tx } => {
                        let result = raft_clone.append_entries(req).await.map_err(|e| {
                            RPCError::RemoteError(openraft::error::RemoteError::new(1, e))
                        });
                        let _ = resp_tx.send(result);
                    }
                    ChannelMessage::Vote { req, resp_tx } => {
                        let result = raft_clone.vote(req).await.map_err(|e| {
                            RPCError::RemoteError(openraft::error::RemoteError::new(1, e))
                        });
                        let _ = resp_tx.send(result);
                    }
                    ChannelMessage::InstallSnapshot { req, resp_tx } => {
                        #[allow(deprecated)]
                        let result = raft_clone.install_snapshot(req).await.map_err(|e| {
                            RPCError::RemoteError(openraft::error::RemoteError::new(1, e))
                        });
                        let _ = resp_tx.send(result);
                    }
                    ChannelMessage::FullSnapshot {
                        vote,
                        meta,
                        data,
                        resp_tx,
                    } => {
                        let snapshot = openraft::Snapshot {
                            meta,
                            snapshot: Box::new(std::io::Cursor::new(data)),
                        };
                        let result = raft_clone
                            .install_full_snapshot(vote, snapshot)
                            .await
                            .map_err(|e| {
                                openraft::error::StreamingError::Unreachable(
                                    openraft::error::Unreachable::new(&e),
                                )
                            });
                        let _ = resp_tx.send(result);
                    }
                }
            }
        });

        let mut members = BTreeSet::new();
        members.insert(1u64);
        raft.initialize(members).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        let op = ClusterOperation::AddNode {
            node_id: 1,
            addr: "test".into(),
        };
        raft.client_write(op).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Explicitly drop raft + handler to release the redb lock
        drop(raft);
        drop(hub);
        handle.abort();
        let _ = handle.await;
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Second run: reopen from persistent log — verify entries survived
    {
        let mut log_store = rekha_cluster::RedbLogStore::open(&log_path).unwrap();
        let state = log_store.get_log_state().await.unwrap();
        // The log should have entries from the first run
        assert!(state.last_log_id.is_some());
        let last = state.last_log_id.unwrap();
        assert!(last.index >= 1);
    }
}
