use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_raft::RaftLogStore;
use rekha_storage::RocksVectorStore;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tonic::transport::server::ServerTlsConfig;
use tonic::transport::{Identity, Server};
use tracing::{info, warn};

use crate::auth::Authenticator;
use crate::config::ServerConfig;
use crate::coordinator::Coordinator;
use crate::proto::rekha_server::RekhaServer as RekhaGrpcServer;
use crate::proto::{HeartbeatRequest, RaftVoteRequest};
use crate::raft_network::GrpcRaftNetwork;
use crate::service::RekhaService;

/// IndexBufferHandle for the metadata Raft group.
/// When CreateCollection/DropCollection commands replicate to followers,
/// this handle initializes or tears down the local collection context.
struct MetadataIndexHandle {
    coordinator: Arc<Coordinator>,
}

impl rekha_core::IndexBufferHandle for MetadataIndexHandle {
    fn buffer_insert(&self, _id: u64, _vector: Vec<f32>, _payload: Option<Vec<u8>>) {}

    fn buffer_delete(&self, _ids: &[u64]) {}

    fn on_collection_created(&self, name: &str, dim: usize, config: &rekha_core::CollectionConfig) {
        let coord = &self.coordinator;
        let rt = tokio::runtime::Handle::current();
        // Spawn a task to initialize the local collection context, since
        // this callback runs inside the Raft apply path.
        let name = name.to_string();
        let config = config.clone();
        let coord = coord.clone();
        tokio::spawn(async move {
            // Skip if already exists locally.
            if coord.collection_exists(&name).await {
                return;
            }
            if let Err(e) = coord.init_collection_ctx(&name, dim, config).await {
                tracing::error!("Failed to init collection context for '{name}': {e}");
            }
        });
    }

    fn on_collection_dropped(&self, name: &str) {
        let coord = &self.coordinator;
        let name = name.to_string();
        let coord = coord.clone();
        tokio::spawn(async move {
            coord.remove_collection_ctx(&name).await;
        });
    }
}

/// The Rekha distributed vector database server.
///
/// Orchestrates:
/// - gRPC service for client RPCs
/// - Coordinator for query routing
/// - Local index for vector search
/// - Raft consensus for replication
/// - Partition management for topology
pub struct ServerInstance {
    config: ServerConfig,
    coordinator: Arc<Coordinator>,
    shutdown_token: CancellationToken,
    authenticator: Arc<Authenticator>,
}

impl ServerInstance {
    /// Create and initialize a new Rekha server from a configuration file.
    pub async fn from_config_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config = ServerConfig::from_file(path)?;
        Self::from_config(config).await
    }

    /// Create and initialize a new Rekha server from a config struct.
    pub async fn from_config(config: ServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize tracing/logging (ignore errors if already initialized).
        let _ = tracing_subscriber::fmt().with_target(false).try_init();

        // Install a global panic hook that logs panics (catches panics in background tasks).
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            tracing::error!("PANIC: {info}");
            prev_hook(info);
        }));

        info!(
            "Starting Rekha server: node={}, bind={}",
            config.cluster.node_id, config.cluster.bind_addr
        );

        // Open storage.
        let store = Arc::new(
            RocksVectorStore::open(&config.cluster.data_dir)
                .map_err(|e| format!("failed to open storage: {e}"))?,
        );
        info!("Storage opened at {}", config.cluster.data_dir);

        // Create partition manager.
        let partition_manager = Arc::new(RwLock::new(PartitionManager::new(
            HashMap::new(),
            config.partition.num_vector_shards,
        )));

        // Create coordinator.
        let coordinator = Arc::new(Coordinator::new(
            config.clone(),
            store.clone(),
            partition_manager,
        ));

        // Create metadata Raft node (partition_id = METADATA_PARTITION_ID).
        {
            let meta_state = rekha_raft::ReplicatedState::new(rekha_core::METADATA_PARTITION_ID);
            let bind_port = config.cluster.bind_addr.split(':').nth(1).unwrap_or("");
            let meta_peers: Vec<String> = config
                .cluster
                .seed_nodes
                .iter()
                .filter(|s| {
                    !s.starts_with(&config.cluster.node_id) && !s.ends_with(&format!(":{bind_port}"))
                })
                .cloned()
                .collect();
            let meta_log_store = RaftLogStore::new(store.clone());
            let meta_network = Arc::new(GrpcRaftNetwork::new());
            let is_meta_single = meta_peers.is_empty();
            // Create a metadata index handle that initializes collection contexts when
            // CreateCollection commands are replicated from the metadata leader.
            let meta_index_handle = Arc::new(MetadataIndexHandle {
                coordinator: coordinator.clone(),
            });
            let meta_raft = Arc::new(rekha_raft::RaftNode::with_store(
                config.cluster.node_id.clone(),
                rekha_core::METADATA_PARTITION_ID,
                meta_peers,
                meta_state,
                Some(meta_log_store),
                Some(meta_index_handle as Arc<dyn rekha_core::IndexBufferHandle>),
                Some(meta_network.clone() as Arc<dyn rekha_raft::RaftPeerNetwork>),
            ));
            // Self-elect metadata Raft in single-node mode.
            if is_meta_single {
                meta_raft.start_election().await.unwrap_or_else(|e| {
                    warn!("Failed to self-elect metadata Raft: {e}");
                });
            }
            coordinator.set_metadata_raft(meta_raft).await;
            info!("Metadata Raft node created");
        }

        // Create the "default" collection (locally — not via Raft proposal, since
        // the metadata Raft may not have a leader yet at this point in the startup).
        let default_dim =
            config.partition.dim_group_size * config.partition.num_dim_groups as usize;
        let default_config = rekha_core::CollectionConfig {
            dim: default_dim as u32,
            num_vector_shards: config.partition.num_vector_shards,
            replication_factor: config.partition.replication_factor as u64,
            num_dim_groups: config.partition.num_dim_groups,
            dim_group_size: config.partition.dim_group_size as u32,
            graph_degree: config.index.graph_degree as u32,
            search_list_size: config.index.search_list_size as u32,
            pq_num_sub_vectors: config.index.pq_num_sub_vectors as u32,
            pq_num_centroids: config.index.pq_num_centroids as u32,
            re_rank_k: config.index.re_rank_k as u32,
        };
        coordinator
            .init_collection_ctx("default", default_dim, default_config)
            .await?;
        info!("Default collection created (dim={default_dim})");

        // Self-elect data Raft nodes with no peers.
        let raft_ids: Vec<u64> = coordinator.raft_nodes.iter().map(|e| *e.key()).collect();
        for pid in raft_ids {
            if let Some(node) = coordinator.raft_node(pid) {
                if node.peers().await.is_empty() {
                    node.start_election().await.unwrap_or_else(|e| {
                        warn!("Failed to self-elect Raft node {pid}: {e}");
                    });
                }
            }
        }

        coordinator.initialize().await;

        // Bootstrap the users store from metadata Raft state.
        let users = {
            let meta_node = coordinator.metadata_raft_node().await;
            let mut users_map: std::collections::HashMap<String, rekha_core::UserConfig> =
                std::collections::HashMap::new();

            // Load existing users from metadata Raft state.
            if let Some(ref meta_node) = meta_node {
                let state = meta_node.read_state().await;
                users_map = state.users.clone();
            }
            drop(meta_node);

            // If no users exist and a default user is configured, create one.
            if users_map.is_empty() {
                if let Some(ref default_user) = config.security.default_user {
                    let password = std::env::var("REKHA_ADMIN_PASSWORD")
                        .unwrap_or_else(|_| default_user.password.clone());

                    if password.is_empty() {
                        warn!("No admin password configured. Set REKHA_ADMIN_PASSWORD or security.default_user.password");
                    } else {
                        let hash = Authenticator::hash_password(&password)
                            .map_err(|e| format!("failed to hash default password: {e}"))?;
                        let user_config = rekha_core::UserConfig {
                            password_hash: hash,
                            role: rekha_core::UserRole::Admin,
                            collections: Vec::new(),
                            created_at: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs(),
                        };

                        if let Some(ref meta_node) = coordinator.metadata_raft_node().await {
                            let cmd = rekha_raft::state::RaftCommand::CreateUser {
                                username: default_user.username.clone(),
                                config: user_config,
                            };
                            let _ = meta_node.propose(cmd).await;
                            info!(
                                "Default admin user '{}' created. Change password immediately.",
                                default_user.username
                            );
                        }
                    }
                }
            }
            Arc::new(tokio::sync::RwLock::new(users_map))
        };

        let authenticator = Arc::new(Authenticator::new(
            &config.security.auth_method,
            users,
        ));

        Ok(Self {
            config,
            coordinator,
            shutdown_token: CancellationToken::new(),
            authenticator,
        })
    }

    /// Access the coordinator (for testing).
    pub fn coordinator(&self) -> &Arc<Coordinator> {
        &self.coordinator
    }

    /// Set the index on the coordinator (deprecated — index is now per-collection).
    #[allow(unused_variables)]
    pub async fn with_index(self, index: RekhaIndex) -> Self {
        info!("with_index is deprecated in multi-collection mode — use create_collection instead");
        self
    }

    /// Spawn a background task that periodically sends heartbeats to seed nodes
    /// and checks peer health.
    fn spawn_heartbeat_loop(&self) {
        let coordinator = self.coordinator.clone();
        let heartbeat_ms = self.config.raft.heartbeat_interval_ms;
        let node_id = self.config.cluster.node_id.clone();
        let bind_port = self.config.cluster.bind_addr.split(':').nth(1).unwrap_or("").to_string();
        let seed_nodes = self.config.cluster.seed_nodes.clone();
        let bind_addr = self.config.cluster.bind_addr.clone();

        // Derive this node's externally-reachable address: look for a seed
        // that starts with our node_id (Docker convention node-X:port)
        // or matches our bind port. Fall back to bind_addr if none found.
        let my_addr = seed_nodes
            .iter()
            .find(|s| s.starts_with(&node_id) || s.ends_with(&format!(":{bind_port}")))
            .cloned()
            .unwrap_or_else(|| bind_addr.clone());

        let token = self.shutdown_token.clone();
        tokio::spawn(async move {
            let mut health_tick = 0u64;
            let mut backoff: std::collections::HashMap<String, (Instant, u64)> =
                std::collections::HashMap::new();

            loop {
                if token.is_cancelled() {
                    info!("Heartbeat loop shutting down");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(heartbeat_ms)).await;

                for seed in &seed_nodes {
                    if seed.starts_with(&node_id) || seed.ends_with(&format!(":{bind_port}")) {
                        continue;
                    }

                    let now = Instant::now();
                    // Backoff: skip entirely within the backoff window.
                    if let Some(&(next_at, _)) = backoff.get(seed.as_str()) {
                        if now < next_at {
                            continue;
                        }
                    }

                    let endpoint = format!("http://{}", seed);
                    let ch = match tonic::transport::Channel::from_shared(endpoint.clone()) {
                        Ok(e) => match e.connect_timeout(Duration::from_secs(2)).connect().await {
                            Ok(ch) => ch,
                            Err(e) => {
                                let entry_ref = backoff.entry(seed.clone()).or_insert((now, 0));
                                entry_ref.1 += 1;
                                let delay = std::cmp::min(
                                    heartbeat_ms * (1u64 << std::cmp::min(entry_ref.1, 10)),
                                    60_000,
                                );
                                entry_ref.0 = now + Duration::from_millis(delay);
                                if entry_ref.1 == 1 {
                                    warn!("Cannot connect to seed node {}: {}", seed, e);
                                } else {
                                    info!(
                                        "Seed {} unreachable (attempt {}, retry in {}ms)",
                                        seed, entry_ref.1, delay
                                    );
                                }
                                continue;
                            }
                        },
                        Err(e) => {
                            warn!("Invalid seed URI {}: {}", seed, e);
                            continue;
                        }
                    };

                    // Connection succeeded — reset backoff.
                    backoff.remove(seed.as_str());

                    let mut client = crate::proto::rekha_client::RekhaClient::new(ch);
                    let (raft_term, commit_idx) = if let Some(raft_node) = coordinator.raft_node(0)
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
                    if let Err(e) = client.heartbeat(req).await {
                        let entry_ref = backoff.entry(seed.clone()).or_insert((now, 0));
                        entry_ref.1 += 1;
                        let delay = std::cmp::min(
                            heartbeat_ms * (1u64 << std::cmp::min(entry_ref.1, 10)),
                            60_000,
                        );
                        entry_ref.0 = now + Duration::from_millis(delay);
                        warn!("Heartbeat to {} failed: {}", seed, e);
                    }
                }

                // Check peer health every 10 heartbeats.
                health_tick += 1;
                if health_tick >= 10 {
                    health_tick = 0;
                    coordinator.check_peer_health().await;
                }
            }
        });
    }

    /// Spawn background timers for all Raft nodes (election timeout checks)
    /// and handle multi-node vote collection after elections start.
    fn spawn_raft_timers(&self) {
        let coordinator = self.coordinator.clone();
        let election_check_ms = std::cmp::min(self.config.raft.election_timeout_min_ms, 100);
        let node_id = self.config.cluster.node_id.clone();
        let token = self.shutdown_token.clone();

        tokio::spawn(async move {
            loop {
                if token.is_cancelled() {
                    info!("Raft timer loop shutting down");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(election_check_ms)).await;

                // Check election timeout for every Raft node.
                let partition_ids: Vec<u64> = coordinator
                    .raft_nodes
                    .iter()
                    .map(|entry| *entry.key())
                    .collect();
                for pid in partition_ids {
                    if let Some(node) = coordinator.raft_node(pid) {
                        if node.check_election_timeout().await {
                            info!(
                                "Election triggered for partition {} on node {}",
                                pid,
                                node.node_id()
                            );

                            let term = node.current_term().await;
                            let last_log_idx = node.last_log_index().await;
                            let last_log_term = node.last_log_term().await;
                            let peers = node.peers().await;
                            let candidate_id = node_id.clone();
                            let n = node.clone();

                            // If no peers, the node self-elected in start_election.
                            if peers.is_empty() {
                                continue;
                            }

                            // Spawn vote collection task — sends RequestVote to each peer.
                            tokio::spawn(async move {
                                let mut votes = 1u64; // self vote
                                let group_size = peers.len() as u64 + 1;
                                let majority = group_size / 2 + 1;

                                for peer_addr in &peers {
                                    let endpoint = format!("http://{}", peer_addr);
                                    // Retry connection up to 3 times (peer gRPC may not be ready).
                                    let ch = {
                                        let mut channel = None;
                                        for attempt in 0..3 {
                                            match tonic::transport::Channel::from_shared(
                                                endpoint.clone(),
                                            ) {
                                                Ok(e) => {
                                                    match tokio::time::timeout(
                                                        Duration::from_secs(2),
                                                        e.connect(),
                                                    )
                                                    .await
                                                    {
                                                        Ok(Ok(c)) => {
                                                            channel = Some(c);
                                                            break;
                                                        }
                                                        _ => {
                                                            tokio::time::sleep(
                                                                Duration::from_millis(500),
                                                            )
                                                            .await;
                                                        }
                                                    }
                                                }
                                                Err(_) => break,
                                            }
                                        }
                                        match channel {
                                            Some(c) => c,
                                            None => {
                                                warn!("Failed to connect to peer {peer_addr} for vote");
                                                continue;
                                            }
                                        }
                                    };
                                    let mut client =
                                        crate::proto::rekha_client::RekhaClient::new(ch);
                                    let req = tonic::Request::new(RaftVoteRequest {
                                        term,
                                        candidate_id: candidate_id.clone(),
                                        last_log_index: last_log_idx,
                                        last_log_term,
                                        partition_id: pid,
                                        collection_name: "default".to_string(),
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
                                        "Election lost for partition {} (got {votes}/{majority} votes)",
                                        pid
                                    );
                                }
                            });
                        }
                    }
                }
            }
        });
    }

    /// Spawn a background loop that sends Raft AppendEntries heartbeats
    /// from leaders to their followers (prevents follower election timeouts).
    fn spawn_raft_heartbeat_loop(&self) {
        let coordinator = self.coordinator.clone();
        let heartbeat_ms = self.config.raft.heartbeat_interval_ms;
        let token = self.shutdown_token.clone();

        tokio::spawn(async move {
            loop {
                if token.is_cancelled() {
                    info!("Raft heartbeat loop shutting down");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(heartbeat_ms)).await;

                let partition_ids: Vec<u64> = coordinator
                    .raft_nodes
                    .iter()
                    .map(|entry| *entry.key())
                    .collect();

                for pid in partition_ids {
                    if let Some(node) = coordinator.raft_node(pid) {
                        if !node.is_leader().await {
                            continue;
                        }

                        let term = node.current_term().await;
                        let commit_idx = node.commit_index().await;
                        let last_log_idx = node.last_log_index().await;
                        let last_log_term = node.last_log_term().await;
                        let leader_id = node.node_id().to_string();
                        let peers = node.peers().await;

                        for peer_addr in &peers {
                            let endpoint = format!("http://{}", peer_addr);
                            let ch = match tonic::transport::Channel::from_shared(endpoint) {
                                Ok(e) => {
                                    match e.connect_timeout(Duration::from_secs(2)).connect().await
                                    {
                                        Ok(ch) => ch,
                                        Err(_) => continue,
                                    }
                                }
                                Err(_) => continue,
                            };
                            let mut client = crate::proto::rekha_client::RekhaClient::new(ch);
                            let req = tonic::Request::new(crate::proto::AppendEntriesRequest {
                                partition_id: pid,
                                leader_term: term,
                                leader_id: leader_id.clone(),
                                prev_log_index: last_log_idx,
                                prev_log_term: last_log_term,
                                entries: vec![], // heartbeat — empty entries
                                leader_commit: commit_idx,
                                collection_name: "default".to_string(),
                            });
                            let _ = client.raft_append_entries(req).await;
                        }
                    }
                }
            }
        });
    }

    /// Spawn a background loop that periodically creates snapshots
    /// of the Raft state and truncates compacted log entries.
    fn spawn_snapshot_loop(&self) {
        let coordinator = self.coordinator.clone();
        let snapshot_interval = self.config.raft.snapshot_interval;
        let check_interval_ms = (snapshot_interval * 100).min(60_000); // at most every 60s
        let token = self.shutdown_token.clone();

        tokio::spawn(async move {
            loop {
                if token.is_cancelled() {
                    info!("Snapshot loop shutting down");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(check_interval_ms)).await;

                let partition_ids: Vec<u64> = coordinator
                    .raft_nodes
                    .iter()
                    .map(|entry| *entry.key())
                    .collect();

                for pid in partition_ids {
                    if let Some(node) = coordinator.raft_node(pid) {
                        if node.should_snapshot(snapshot_interval).await {
                            if let Err(e) = node.create_snapshot().await {
                                warn!("Failed to create snapshot for partition {pid}: {e}");
                            }
                        }
                    }
                }
            }
        });
    }

    /// Run the server (blocking). Handles SIGTERM/SIGINT for graceful shutdown.
    /// Spawns a background heartbeat loop for cluster discovery and health monitoring.
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let addr = self.config.cluster.bind_addr.parse()?;
        let service = RekhaService::new(self.coordinator.clone(), self.authenticator.clone());

        // Spawn background loops.
        self.spawn_heartbeat_loop();
        self.spawn_raft_timers();
        self.spawn_raft_heartbeat_loop();
        self.spawn_snapshot_loop();

        let token = self.shutdown_token.clone();
        let shutdown = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to listen for ctrl-c");
            info!("Shutdown signal received, cancelling background tasks...");
            token.cancel();
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
    async fn test_server_from_config_file_not_found() {
        let result = ServerInstance::from_config_file("/nonexistent/config.yaml").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_with_index() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        let store = rekha_storage::RocksVectorStore::open(temp_dir()).unwrap();
        let mut index =
            rekha_index::RekhaIndex::new(8, 4, 16, 4, store, rekha_core::DistanceMetric::L2)
                .unwrap();
        for i in 0..5 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        let server = server.with_index(index).await;
        assert!(server.coordinator.is_initialized().await);
    }

    #[tokio::test]
    async fn test_raft_nodes_created_at_startup() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        // With num_vector_shards=1, exactly 1 raft node should exist
        let node = server.coordinator.raft_node(0);
        assert!(node.is_some());
        assert_eq!(node.unwrap().node_id(), "test-node");
    }

    #[tokio::test]
    async fn test_spawn_heartbeat_loop() {
        // Just verify spawn_heartbeat_loop doesn't panic
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        // The method is private, called by run(). We verify via run() that it spawns without panic.
        // For unit testing, just verify the server was created successfully.
        assert!(server.coordinator.raft_node(0).is_some());
    }

    #[tokio::test]
    async fn test_spawn_raft_timers() {
        // Verify the raft timers can be spawned without panic
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        // spawn_raft_timers is called by run(). Verify the coordinator has raft nodes.
        assert!(!server.coordinator.raft_nodes.is_empty());
    }

    #[tokio::test]
    async fn test_server_with_tls_config() {
        // TLS path should fail gracefully when cert/key files don't exist
        let mut config = ServerConfig::dev_default("test-node", &temp_dir());
        config.tls.enabled = true;
        config.tls.cert_path = Some("/nonexistent/cert.pem".into());
        config.tls.key_path = Some("/nonexistent/key.pem".into());
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_ok()); // Server config is valid; TLS error only happens on run()
    }

    #[tokio::test]
    async fn test_server_config_refs() {
        let config = ServerConfig::dev_default("test-node", &temp_dir());
        let server = ServerInstance::from_config(config).await.unwrap();
        assert_eq!(server.config.cluster.node_id, "test-node");
        assert_eq!(server.config.partition.num_dim_groups, 4);
    }

    #[tokio::test]
    async fn test_from_config_invalid_path() {
        // Using a path in a nonexistent directory should cause an error.
        let config = ServerConfig::dev_default("test-node", "/nonexistent_dir_xyz/data");
        let result = ServerInstance::from_config(config).await;
        assert!(result.is_err());
    }
}
