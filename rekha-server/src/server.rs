use rekha_partition::PartitionManager;
use rekha_storage::RocksVectorStore;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::server::ServerTlsConfig;
use tonic::transport::{Identity, Server};
use tracing::{info, warn};

use crate::config::ServerConfig;
use crate::coordinator::{Coordinator, SystemRaftHandle, SYSTEM_PARTITION_ID};
use crate::proto::rekha_server::RekhaServer as RekhaGrpcServer;
use crate::proto::{AppendEntriesRequest, HeartbeatRequest, RaftVoteRequest};
use crate::service::RekhaService;
use tonic::transport::Channel;

async fn connect_peer(addr: &str, timeout: Duration) -> Option<Channel> {
    let uri = format!("http://{addr}");
    match tonic::transport::Channel::from_shared(uri) {
        Ok(e) => e.connect_timeout(timeout).connect().await.ok(),
        Err(_) => None,
    }
}

pub struct ServerInstance {
    config: ServerConfig,
    coordinator: Arc<Coordinator>,
}

impl ServerInstance {
    pub async fn from_config_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ServerConfig::from_file(path)?;
        Self::from_config(config).await
    }

    pub async fn from_config(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let _ = tracing_subscriber::fmt().with_target(false).try_init();

        info!(
            "Starting Rekha server: node={}, bind={}",
            config.cluster.node_id, config.cluster.bind_addr
        );

        let store = Arc::new(
            RocksVectorStore::open(&config.cluster.data_dir)
                .map_err(|e| format!("failed to open storage: {e}"))?,
        );
        info!("Storage opened at {}", config.cluster.data_dir);

        let partition_manager = Arc::new(RwLock::new(PartitionManager::new(
            HashMap::new(),
            config.partition.num_dim_groups,
            config.partition.dim_group_size * config.partition.num_dim_groups as usize,
        )));

        let coordinator = Arc::new(Coordinator::new(
            config.clone(),
            store.clone(),
            partition_manager,
        ));

        // Create default collection.
        coordinator.create_default_collection().await?;

        // Create Raft nodes for the default collection.
        let raft_log_store = coordinator.raft_log_store_for("default");
        let num_shards = config.partition.num_vector_shards;
        let node_id = &config.cluster.node_id;
        let bind_port = config
            .cluster
            .bind_addr
            .split(':')
            .nth(1)
            .unwrap_or("50051");
        let peers: Vec<String> = config
            .cluster
            .seed_nodes
            .iter()
            .filter(|s| {
                // Filter self: seed starts with node_id (Docker convention).
                if s.starts_with(node_id) {
                    return false;
                }
                // Filter self: seed has same port as bind_addr (local dev).
                // This filters out 127.0.0.1:50051 when bind_addr is 0.0.0.0:50051.
                let seed_port = s.split(':').nth(1).unwrap_or("50051");
                let seed_host = s.split(':').next().unwrap_or("");
                if seed_port == bind_port && (seed_host == "127.0.0.1" || seed_host == "localhost")
                {
                    return false;
                }
                true
            })
            .cloned()
            .collect();

        let index_handle: Arc<dyn rekha_core::IndexBufferHandle> = {
            struct DefaultHandle {
                name: String,
                collections: Arc<dashmap::DashMap<String, crate::coordinator::CollectionState>>,
            }
            impl rekha_core::IndexBufferHandle for DefaultHandle {
                fn buffer_insert(&self, id: u64, vector: Vec<f32>) {
                    if let Some(state) = self.collections.get(&self.name) {
                        if let Ok(idx) = state.index.try_read() {
                            if let Some(ref idx) = *idx {
                                idx.buffer_insert(id, vector);
                            }
                        }
                    }
                }
                fn buffer_delete(&self, ids: &[u64]) {
                    if let Some(state) = self.collections.get(&self.name) {
                        if let Ok(idx) = state.index.try_read() {
                            if let Some(ref idx) = *idx {
                                idx.buffer_delete(ids);
                            }
                        }
                    }
                }
            }
            Arc::new(DefaultHandle {
                name: "default".into(),
                collections: coordinator.collections.clone(),
            })
        };

        for shard in 0..num_shards {
            let state = rekha_raft::ReplicatedState::new(shard);
            let raft_node = Arc::new(rekha_raft::RaftNode::with_store(
                config.cluster.node_id.clone(),
                shard,
                peers.clone(),
                state,
                Some(raft_log_store.clone()),
                Some(index_handle.clone()),
            ));
            if peers.is_empty() {
                let _ = raft_node.start_election().await;
            }
            coordinator.register_raft_node("default", shard, raft_node);
        }
        info!("Created {} Raft nodes for default collection", num_shards);

        // Create the system Raft group for collection metadata.
        let system_raft_log_store = coordinator.raft_log_store_for("__system__");
        let system_state = rekha_raft::ReplicatedState::new(SYSTEM_PARTITION_ID);
        let system_handle = Arc::new(SystemRaftHandle {
            coordinator: Arc::as_ptr(&coordinator),
        });
        let system_raft_node = Arc::new(rekha_raft::RaftNode::with_store(
            config.cluster.node_id.clone(),
            SYSTEM_PARTITION_ID,
            peers.clone(),
            system_state,
            Some(system_raft_log_store),
            Some(system_handle as Arc<dyn rekha_core::IndexBufferHandle>),
        ));
        if peers.is_empty() {
            let _ = system_raft_node.start_election().await;
        }

        // Register system Raft node on the coordinator.
        coordinator.register_system_raft_node(system_raft_node);
        info!("System Raft group created");

        coordinator.initialize_all().await;

        Ok(Self {
            config,
            coordinator,
        })
    }

    fn spawn_heartbeat_loop(&self) {
        let coordinator = self.coordinator.clone();
        let heartbeat_ms = self.config.raft.heartbeat_interval_ms;
        let node_id = self.config.cluster.node_id.clone();
        let seed_nodes = self.config.cluster.seed_nodes.clone();
        let bind_addr = self.config.cluster.bind_addr.clone();

        let my_addr = seed_nodes
            .iter()
            .find(|s| s.starts_with(&node_id))
            .cloned()
            .unwrap_or_else(|| bind_addr.clone());

        tokio::spawn(async move {
            let mut health_tick = 0u64;
            loop {
                tokio::time::sleep(Duration::from_millis(heartbeat_ms)).await;

                for seed in &seed_nodes {
                    if seed.starts_with(&node_id) {
                        continue;
                    }
                    let endpoint = format!("http://{}", seed);
                    match tonic::transport::Channel::from_shared(endpoint) {
                        Ok(ch) => match ch.connect().await {
                            Ok(ch) => {
                                let mut client = crate::proto::rekha_client::RekhaClient::new(ch);
                                let (raft_term, commit_idx) = if let Some(sys_node) =
                                    coordinator.system_raft_node()
                                {
                                    (sys_node.current_term().await, sys_node.commit_index().await)
                                } else if let Some(raft_node) = coordinator.raft_node("default", 0)
                                {
                                    (
                                        raft_node.current_term().await,
                                        raft_node.commit_index().await,
                                    )
                                } else {
                                    (0, 0)
                                };
                                let req = tonic::Request::new(HeartbeatRequest {
                                    node_id: node_id.clone(),
                                    address: my_addr.clone(),
                                    raft_term,
                                    commit_index: commit_idx,
                                    storage_bytes: 0,
                                });
                                match client.heartbeat(req).await {
                                    Ok(resp) => {
                                        let _resp = resp.into_inner();
                                    }
                                    Err(e) => {
                                        warn!("Heartbeat to {} failed: {}", seed, e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("Cannot connect to seed node {}: {}", seed, e);
                            }
                        },
                        Err(e) => {
                            warn!("Invalid seed URI {}: {}", seed, e);
                        }
                    }
                }

                health_tick += 1;
                if health_tick >= 10 {
                    health_tick = 0;
                    coordinator.check_peer_health().await;
                }
            }
        });
    }

    fn spawn_raft_timers(&self) {
        let coordinator = self.coordinator.clone();
        let election_check_ms = std::cmp::min(self.config.raft.election_timeout_min_ms, 100);
        let node_id = self.config.cluster.node_id.clone();

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(election_check_ms)).await;

                let nodes = coordinator.all_raft_nodes();
                for (col_name, pid, node) in nodes {
                    if node.check_election_timeout().await {
                        info!(
                            "Election triggered for '{}' partition {} on node {}",
                            col_name,
                            pid,
                            node.node_id()
                        );

                        let term = node.current_term().await;
                        let last_log_idx = node.last_log_index().await;
                        let last_log_term = node.last_log_term().await;
                        let peers = node.peers().to_vec();
                        let candidate_id = node_id.clone();
                        let n = node.clone();
                        let cname = col_name.clone();

                        if peers.is_empty() {
                            continue;
                        }

                        tokio::spawn(async move {
                            let mut votes = 1u64;
                            let group_size = peers.len() as u64 + 1;
                            let majority = group_size / 2 + 1;

                            for peer_addr in &peers {
                                let Some(ch) =
                                    connect_peer(peer_addr, Duration::from_secs(2)).await
                                else {
                                    continue;
                                };
                                let mut client = crate::proto::rekha_client::RekhaClient::new(ch);
                                let req = tonic::Request::new(RaftVoteRequest {
                                    term,
                                    candidate_id: candidate_id.clone(),
                                    last_log_index: last_log_idx,
                                    last_log_term,
                                    partition_id: pid,
                                    collection_name: cname.clone(),
                                });
                                if let Ok(resp) = client.raft_request_vote(req).await {
                                    if resp.into_inner().vote_granted {
                                        votes += 1;
                                    }
                                }
                            }

                            if votes >= majority {
                                n.become_leader().await;
                            } else {
                                info!(
                                    "Election lost for '{}' partition {} (got {votes}/{majority} votes)",
                                    cname, pid
                                );
                            }
                        });
                    }
                }
            }
        });
    }

    fn spawn_raft_heartbeat_loop(&self) {
        let coordinator = self.coordinator.clone();
        let heartbeat_ms = self.config.raft.heartbeat_interval_ms;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(heartbeat_ms)).await;

                let nodes = coordinator.all_raft_nodes();

                for (col_name, pid, node) in &nodes {
                    if !node.is_leader().await {
                        continue;
                    }

                    let term = node.current_term().await;
                    let commit_idx = node.commit_index().await;
                    let leader_id = node.node_id().to_string();
                    let peers = node.peers().to_vec();

                    for peer_addr in &peers {
                        let (entries, prev_log_index, prev_log_term) =
                            node.entries_for_peer(peer_addr).await;

                        let Some(ch) = connect_peer(peer_addr, Duration::from_secs(2)).await else {
                            continue;
                        };
                        let mut client = crate::proto::rekha_client::RekhaClient::new(ch);

                        // Convert entries to proto
                        let proto_entries = entries
                            .iter()
                            .map(|e| crate::proto::RaftEntry {
                                term: e.term,
                                index: e.index,
                                command: Some(crate::proto::RaftCommand {
                                    cmd: Some(crate::service::raft_command_to_proto(&e.command)),
                                }),
                            })
                            .collect();

                        let req = tonic::Request::new(AppendEntriesRequest {
                            partition_id: *pid,
                            leader_term: term,
                            leader_id: leader_id.clone(),
                            prev_log_index,
                            prev_log_term,
                            entries: proto_entries,
                            leader_commit: commit_idx,
                            collection_name: col_name.clone(),
                        });

                        match client.raft_append_entries(req).await {
                            Ok(resp) => {
                                let ack = resp.into_inner();
                                if ack.success {
                                    // Record replication up to the last entry sent
                                    if let Some(last) = entries.last() {
                                        node.record_replication(peer_addr, last.index).await;
                                    }
                                } else if ack.current_term > term {
                                    // Leader is stale → step down
                                    info!(
                                        "Stale leader for '{}' partition {} (peer has term {} > {}), stepping down",
                                        col_name, pid, ack.current_term, term
                                    );
                                    // We don't step down automatically here;
                                    // the election timer will handle it.
                                } else {
                                    // Log mismatch — retry with older entries
                                    node.retry_replication(peer_addr).await;
                                }
                            }
                            Err(_) => {
                                // Peer unreachable — will retry on next tick
                            }
                        }
                    }

                    // After processing all peers, check if we can advance commit_index
                    node.advance_commit().await;
                }
            }
        });
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.config.cluster.bind_addr.parse()?;
        let service = RekhaService::new(self.coordinator.clone());

        self.spawn_heartbeat_loop();
        self.spawn_raft_timers();
        self.spawn_raft_heartbeat_loop();

        let shutdown = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for ctrl-c");
            info!("Shutdown signal received, draining connections...");
        };

        if self.config.tls.enabled {
            let cert_path = self
                .config
                .tls
                .cert_path
                .as_ref()
                .ok_or_else(|| "TLS enabled but cert_path not configured".to_string())?;
            let key_path = self
                .config
                .tls
                .key_path
                .as_ref()
                .ok_or_else(|| "TLS enabled but key_path not configured".to_string())?;

            let cert = std::fs::read(cert_path).map_err(|e| format!("failed to read cert: {e}"))?;
            let key = std::fs::read(key_path).map_err(|e| format!("failed to read key: {e}"))?;
            let identity = Identity::from_pem(cert, key);

            let mut tls_config = ServerTlsConfig::new().identity(identity);
            if let Some(ca_path) = &self.config.tls.ca_cert_path {
                let ca_cert =
                    std::fs::read(ca_path).map_err(|e| format!("failed to read CA cert: {e}"))?;
                tls_config =
                    tls_config.client_ca_root(tonic::transport::Certificate::from_pem(ca_cert));
            }

            info!("gRPC server listening on {addr} with TLS");
            Server::builder()
                .tls_config(tls_config)?
                .add_service(RekhaGrpcServer::new(service))
                .serve_with_shutdown(addr, shutdown)
                .await?;
        } else {
            info!("gRPC server listening on {addr} (plaintext)");
            Server::builder()
                .add_service(RekhaGrpcServer::new(service))
                .serve_with_shutdown(addr, shutdown)
                .await?;
        }

        info!("Server shut down gracefully");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir() -> String {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("/tmp/rekha_server_test_{}", n)
    }

    #[tokio::test]
    async fn test_server_from_config() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_default_collection_created() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        assert!(server.coordinator.collection_exists("default").await);
    }

    #[tokio::test]
    async fn test_raft_nodes_created_for_default() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        let node = server.coordinator.raft_node("default", 0);
        assert!(node.is_some());
        assert_eq!(node.unwrap().node_id(), "test-node");
    }

    #[tokio::test]
    async fn test_system_raft_node_created() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        let sys_node = server.coordinator.system_raft_node();
        assert!(sys_node.is_some());
        assert_eq!(sys_node.unwrap().partition_id(), SYSTEM_PARTITION_ID);
    }

    #[tokio::test]
    async fn test_raft_nodes_contain_system_node() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        let all = server.coordinator.all_raft_nodes();
        assert!(all.iter().any(|(name, _, _)| name == "__system__"));
        assert!(all.iter().any(|(name, _, _)| name == "default"));
    }

    #[tokio::test]
    async fn test_from_config_invalid_path() {
        let config = ServerConfig::dev_default("test-node", "/nonexistent_dir_xyz/data");
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_err());
    }
}
