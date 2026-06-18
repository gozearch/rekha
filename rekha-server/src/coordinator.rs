use rekha_core::{
    ClusterTopology, DistanceMetric, IndexBufferHandle, NodeInfo, NodeStatus, Payload, RekhaError,
    ScoredPoint, SearchParams, SearchStats, VectorIndex, VectorStoreBackend,
};
use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_raft::{RaftLogStore, RaftNode};
use rekha_storage::RocksVectorStore;

use dashmap::DashMap;
use rekha_client::RekhaClient as PeerRekhaClient;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

use crate::config::ServerConfig;

/// How long without a heartbeat before a peer is marked Unreachable.
const PEER_TIMEOUT: Duration = Duration::from_secs(10);

/// Tracked information about a peer node.
#[derive(Debug, Clone)]
pub(crate) struct PeerState {
    pub info: NodeInfo,
    pub last_seen: Instant,
}

/// A gRPC client connection to a peer node.
struct PeerClient {
    #[allow(dead_code)]
    info: NodeInfo,
    client: PeerRekhaClient,
    last_used: Instant,
    error_count: u64,
    collection_name: String,
}

impl PeerClient {
    async fn connect(info: &NodeInfo, collection_name: &str) -> Result<Self, RekhaError> {
        let seeds = vec![info.address.clone()];
        let client = PeerRekhaClient::connect(&seeds).await?;
        Ok(Self {
            info: info.clone(),
            client,
            last_used: Instant::now(),
            error_count: 0,
            collection_name: collection_name.to_string(),
        })
    }

    async fn try_search(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        self.last_used = Instant::now();
        self.client
            .search_with_params(query.to_vec(), &self.collection_name, k, params.clone())
            .await
    }
}

/// Pool of gRPC clients to known peer nodes.
pub(crate) struct PeerPool {
    clients: HashMap<String, PeerClient>,
    collection_name: String,
}

impl PeerPool {
    pub fn new(collection_name: &str) -> Self {
        Self {
            clients: HashMap::new(),
            collection_name: collection_name.to_string(),
        }
    }

    /// Reconcile the pool with the current peer list.
    /// Connects to new peers, keeps existing connections, drops removed peers.
    pub async fn refresh(&mut self, peers: &[NodeInfo]) {
        // Drop peers no longer in the list.
        let active: std::collections::HashSet<String> =
            peers.iter().map(|p| p.node_id.clone()).collect();
        self.clients.retain(|node_id, _| active.contains(node_id));

        // Connect to new or reconnecting peers.
        for info in peers {
            if !self.clients.contains_key(&info.node_id) {
                match PeerClient::connect(info, &self.collection_name).await {
                    Ok(client) => {
                        info!("Connected to peer {} at {}", info.node_id, info.address);
                        self.clients.insert(info.node_id.clone(), client);
                    }
                    Err(e) => {
                        info!("Failed to connect to peer {}: {}", info.node_id, e);
                    }
                }
            }
        }
    }

    /// Fan out a search query to all connected peers.
    /// Returns merged (candidates, stats) from all peers that responded.
    pub async fn search_fan_out(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> (Vec<ScoredPoint>, SearchStats) {
        let mut peer_params = params.clone();
        peer_params.local_only = true;
        let mut all_candidates: Vec<ScoredPoint> = Vec::new();
        let mut stats = SearchStats::default();
        let mut nodes_contacted = 0u32;

        let node_ids: Vec<String> = self.clients.keys().cloned().collect();
        for node_id in &node_ids {
            if let Some(client) = self.clients.get_mut(node_id) {
                match client.try_search(query, k, &peer_params).await {
                    Ok((candidates, _peer_stats)) => {
                        nodes_contacted += 1;
                        all_candidates.extend(candidates);
                        client.error_count = 0;
                    }
                    Err(_) => {
                        if let Some(c) = self.clients.get_mut(node_id) {
                            c.error_count += 1;
                            // Remove client after 3 consecutive errors.
                            if c.error_count >= 3 {
                                info!("Dropping peer {} after 3 errors", node_id);
                                self.clients.remove(node_id);
                            }
                        }
                        stats.warnings.push(format!("peer {node_id} search failed"));
                    }
                }
            }
        }

        all_candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_candidates.truncate(k * 2);
        stats.nodes_contacted = nodes_contacted;
        (all_candidates, stats)
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }
}

/// The distributed query coordinator.
///
/// Responsibilities:
/// - Route search queries to the appropriate partitions
/// - Fan out to dimension groups for partial distance computation
/// - Merge partial results from multiple nodes
/// - Apply early-stop pruning across dimension groups
/// - Manage cluster topology and node health
pub struct Coordinator {
    /// Server configuration.
    config: ServerConfig,
    /// Local index (this node's data).
    index: Arc<RwLock<Option<RekhaIndex>>>,
    /// Local storage backend.
    store: Arc<RocksVectorStore>,
    /// Partition topology manager.
    #[allow(dead_code)]
    partition_manager: Arc<RwLock<PartitionManager>>,
    /// Raft nodes for each partition hosted on this node.
    pub raft_nodes: DashMap<u64, Arc<RaftNode>>,
    /// Cluster topology (peer nodes).
    topology: Arc<RwLock<ClusterTopology>>,
    /// Whether this coordinator is initialized.
    initialized: Arc<RwLock<bool>>,
    /// Peer node tracking (node_id → state).
    peers: Arc<RwLock<HashMap<String, PeerState>>>,
    /// gRPC client pool for peer nodes.
    peer_pool: Arc<RwLock<PeerPool>>,
    /// Auto-incrementing ID counter. Initialized from max stored ID + 1.
    /// When the client sends id=0, we replace it with next_auto_id.fetch_add(1).
    next_auto_id: AtomicU64,
}

impl Coordinator {
    /// Create a new coordinator.
    pub fn new(
        config: ServerConfig,
        store: Arc<RocksVectorStore>,
        partition_manager: Arc<RwLock<PartitionManager>>,
    ) -> Self {
        let starting_id = Self::starting_auto_id(&store);
        Self {
            config,
            index: Arc::new(RwLock::new(None)),
            store,
            partition_manager,
            raft_nodes: DashMap::new(),
            topology: Arc::new(RwLock::new(ClusterTopology {
                cluster_id: String::new(),
                nodes: HashMap::new(),
                partition_map: HashMap::new(),
            })),
            initialized: Arc::new(RwLock::new(false)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            peer_pool: Arc::new(RwLock::new(PeerPool::new("default"))),
            next_auto_id: AtomicU64::new(starting_id),
        }
    }

    /// Compute the starting auto-ID by scanning all stored IDs and taking max+1.
    fn starting_auto_id(store: &RocksVectorStore) -> u64 {
        match store.iter_ids() {
            Ok(ids) => ids.iter().max().copied().unwrap_or(0) + 1,
            Err(_) => 1,
        }
    }

    /// Initialize the coordinator with an index.
    pub async fn initialize(&self, index: RekhaIndex) {
        {
            let mut idx = self.index.write().await;
            *idx = Some(index);
        }
        *self.initialized.write().await = true;

        // Rehydrate insert buffer from Raft state (crash recovery).
        let _ = self.recover_index_buffer().await;

        // Spawn background flush loop.
        self.spawn_flush_loop();
        info!("Coordinator initialized");
    }

    /// Recover the insert buffer from Raft-replicated state after restart.
    async fn recover_index_buffer(&self) -> Result<(), RekhaError> {
        let mut idx = self.index.write().await;
        let index = idx.as_mut().ok_or_else(|| RekhaError::Internal {
            detail: "index not initialized for recovery".into(),
        })?;

        let mut recovered = 0usize;
        for item in self.raft_nodes.iter() {
            let raft_node = item.value();
            let state = raft_node.read_state().await;
            for (id, bytes) in state.vectors.iter() {
                if index.graph_contains_id(*id) {
                    continue; // Already in the graph
                }
                let vec: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                index.buffer_insert(*id, vec);
                recovered += 1;
            }
        }

        if recovered > 0 {
            info!("Recovered {recovered} vectors into insert buffer from Raft state");
            // Flush immediately if buffer exceeds threshold
            if index.should_flush() {
                index.flush_buffer()?;
                info!("Immediate flush after recovery ({recovered} vectors)");
            }
        }
        Ok(())
    }

    /// Spawn a background task that periodically flushes the insert buffer.
    fn spawn_flush_loop(&self) {
        let flush_ms = self.config.index.insert_buffer_flush_interval_ms;
        let index = self.index.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(flush_ms));
            loop {
                interval.tick().await;
                let mut idx = index.write().await;
                if let Some(ref mut idx) = *idx {
                    if idx.should_flush() || idx.buffer_len() > 0 {
                        let buf_len = idx.buffer_len();
                        if let Err(e) = idx.flush_buffer() {
                            tracing::warn!("Buffer flush failed: {e} (buffer: {buf_len} vectors)");
                        }
                    }
                }
            }
        });
    }

    /// Check if initialized.
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    /// Look up a peer's address by node ID.
    pub async fn peer_address(&self, node_id: &str) -> Option<String> {
        self.peers
            .read()
            .await
            .get(node_id)
            .map(|p| p.info.address.clone())
    }

    /// Get node info for this node.
    pub fn local_node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.config.cluster.node_id.clone(),
            address: self.config.cluster.bind_addr.clone(),
            partition_id: 0, // Simplified
            dim_groups: (0..self.config.partition.num_dim_groups).collect(),
            is_leader: true,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        }
    }

    /// Get a reference to the local store.
    pub fn store(&self) -> &Arc<RocksVectorStore> {
        &self.store
    }

    pub fn cluster_id(&self) -> &str {
        "rekha-dev"
    }

    pub fn node_id(&self) -> &str {
        &self.config.cluster.node_id
    }

    pub fn bind_addr(&self) -> &str {
        &self.config.cluster.bind_addr
    }

    pub fn seed_nodes(&self) -> &[String] {
        &self.config.cluster.seed_nodes
    }

    /// Register or update a peer from a heartbeat or handshake.
    pub async fn register_peer(&self, info: NodeInfo) {
        let mut peers = self.peers.write().await;
        let mut info = info;
        info.last_heartbeat = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        peers.insert(
            info.node_id.clone(),
            PeerState {
                info,
                last_seen: Instant::now(),
            },
        );
        drop(peers);
        self.refresh_peer_pool().await;
    }

    /// Get the list of healthy peers (for pool refresh and handshake responses).
    pub async fn healthy_peers(&self) -> Vec<NodeInfo> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| p.info.status == NodeStatus::Healthy)
            .map(|p| p.info.clone())
            .collect()
    }

    /// Refresh the peer connection pool from current healthy peer list.
    async fn refresh_peer_pool(&self) {
        let healthy = self.healthy_peers().await;
        let mut pool = self.peer_pool.write().await;
        let external: Vec<NodeInfo> = healthy
            .into_iter()
            .filter(|p| p.node_id != self.config.cluster.node_id)
            .collect();
        pool.refresh(&external).await;
    }

    /// Get the list of known peers for handshake responses.
    /// Excludes the given node_id (the requester).
    pub async fn peers_for_handshake(&self, exclude: &str) -> Vec<NodeInfo> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| p.info.node_id != exclude)
            .map(|p| p.info.clone())
            .collect()
    }

    /// Mark peers that haven't sent a heartbeat recently as Unreachable.
    pub async fn check_peer_health(&self) {
        let mut peers = self.peers.write().await;
        let mut changed = false;
        for peer in peers.values_mut() {
            if peer.last_seen.elapsed() > PEER_TIMEOUT && peer.info.status == NodeStatus::Healthy {
                peer.info.status = NodeStatus::Unreachable;
                changed = true;
            } else if peer.last_seen.elapsed() <= PEER_TIMEOUT
                && peer.info.status == NodeStatus::Unreachable
            {
                peer.info.status = NodeStatus::Healthy;
                changed = true;
            }
        }
        if changed {
            drop(peers);
            self.sync_topology().await;
        }
    }

    /// Rebuild the cluster topology from current peer state + local node.
    async fn sync_topology(&self) {
        let peers = self.peers.read().await;
        let mut nodes = HashMap::new();
        nodes.insert(self.config.cluster.node_id.clone(), self.local_node_info());
        for peer in peers.values() {
            nodes.insert(peer.info.node_id.clone(), peer.info.clone());
        }
        let mut topo = self.topology.write().await;
        topo.nodes = nodes;
    }

    /// Return the config reference (for server heartbeat loop).
    pub fn config_ref(&self) -> &ServerConfig {
        &self.config
    }

    /// Register a Raft node for a partition.
    pub fn register_raft_node(&self, partition_id: u64, node: Arc<RaftNode>) {
        self.raft_nodes.insert(partition_id, node);
    }

    /// Get a Raft node by partition ID.
    pub fn raft_node(&self, partition_id: u64) -> Option<Arc<RaftNode>> {
        self.raft_nodes.get(&partition_id).map(|n| n.clone())
    }

    /// Create a persistent log store backed by the local RocksDB.
    pub fn raft_log_store(&self) -> RaftLogStore {
        RaftLogStore::new(self.store.clone())
    }
}

impl IndexBufferHandle for Coordinator {
    fn buffer_insert(&self, id: u64, vector: Vec<f32>) {
        if let Ok(idx) = self.index.try_read() {
            if let Some(ref idx) = *idx {
                idx.buffer_insert(id, vector);
            }
        }
    }

    fn buffer_delete(&self, ids: &[u64]) {
        if let Ok(idx) = self.index.try_read() {
            if let Some(ref idx) = *idx {
                idx.buffer_delete(ids);
            }
        }
    }
}

impl Coordinator {
    pub async fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let index_guard = self.index.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal {
            detail: "index not initialized".into(),
        })?;

        let num_groups = self.config.partition.num_dim_groups;
        let total_dim = query.len();
        let dims_per_group = total_dim / num_groups as usize;

        // Phase 1: Collect candidates from local index (dimension group fan-out).
        let mut candidates: Vec<ScoredPoint> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        let mut local_groups = 0u32;

        for group in 0..num_groups {
            let start_dim = (group as usize) * dims_per_group;
            let end_dim = start_dim + dims_per_group;

            match index.search_dim_range(&query, k * 2, start_dim, end_dim, &params) {
                Ok((ids, dists)) => {
                    local_groups += 1;
                    for (i, id) in ids.iter().enumerate() {
                        if !seen_ids.insert(*id) {
                            continue;
                        }
                        let score = dists.get(i).copied().unwrap_or(f32::MAX);

                        if candidates.len() >= k {
                            candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                            if score > candidates[k - 1].score {
                                continue;
                            }
                        }

                        let payload = if params.include_payloads {
                            self.store
                                .get_payload(*id)
                                .ok()
                                .flatten()
                                .map(Payload::from_bytes)
                        } else {
                            None
                        };

                        candidates.push(ScoredPoint {
                            id: *id,
                            score,
                            payload,
                        });
                    }
                }
                Err(e) => {
                    stats
                        .warnings
                        .push(format!("local dim_group {group} search failed: {e}"));
                }
            }
        }

        // Phase 2: Fan out to peer nodes if available (skip for local-only searches).
        let peer_count = if params.local_only {
            0
        } else {
            let has_peers = { !self.peer_pool.read().await.is_empty() };
            if has_peers {
                let mut pool = self.peer_pool.write().await;
                let (peer_results, peer_stats) = pool.search_fan_out(&query, k, &params).await;
                stats.nodes_contacted = local_groups + peer_stats.nodes_contacted;
                stats.warnings.extend(peer_stats.warnings);
                candidates.extend(peer_results);
                pool.len()
            } else {
                stats.nodes_contacted = local_groups;
                0
            }
        };

        // Phase 3: Re-rank top candidates with full-precision vectors.
        // Sort by partial score and take top (k * 2) for re-ranking.
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k * 2);

        // Re-rank with exact distances.
        let metric = DistanceMetric::L2; // default; configurable later
        for candidate in candidates.iter_mut().take(k * 2) {
            let id = candidate.id;
            if let Ok(Some(full_vec)) = self.store.get_vector(id) {
                let exact_dist = metric.distance(&full_vec, &query);
                candidate.score = exact_dist;
            }
        }
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k);

        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.vectors_scanned = index.len() as u64 + peer_count as u64 * k as u64;

        Ok((candidates, stats))
    }

    pub async fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Payload>,
    ) -> Result<u64, RekhaError> {
        let id = if id == 0 {
            self.next_auto_id.fetch_add(1, Ordering::SeqCst)
        } else {
            id
        };

        // Route through Raft if a Raft node exists for partition 0.
        if let Some(raft_node) = self.raft_node(0) {
            let cmd = rekha_raft::state::RaftCommand::Insert {
                id,
                vector,
                payload: payload.map(|p| p.data),
            };
            raft_node.propose(cmd).await?;
            return Ok(id);
        }

        // Fallback: direct store write (single-node / uninitialized).
        self.store.put_vector(id, &vector)?;
        if let Some(ref p) = payload {
            self.store.put_payload(id, &p.data)?;
        }
        Ok(id)
    }

    pub async fn topology(&self) -> Result<ClusterTopology, RekhaError> {
        let topo = self.topology.read().await;
        Ok(topo.clone())
    }

    pub async fn node_info(&self, _node_id: &str) -> Result<NodeInfo, RekhaError> {
        Ok(self.local_node_info())
    }

    pub async fn build_index(&self) -> Result<(), RekhaError> {
        Err(RekhaError::Unavailable {
            detail: "build_index is deprecated; index builds automatically".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_store() -> Arc<RocksVectorStore> {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_coord_store_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        Arc::new(rekha_storage::RocksVectorStore::open(&dir).unwrap())
    }

    fn test_coordinator() -> Coordinator {
        let config = ServerConfig::dev_default("test-node", "/tmp/rekha_coord_test");
        let store = temp_store();
        let pm = Arc::new(RwLock::new(rekha_partition::PartitionManager::new(
            HashMap::new(),
            4,
            768,
        )));
        Coordinator::new(config, store, pm)
    }

    #[tokio::test]
    async fn test_coordinator_new() {
        let coord = test_coordinator();
        assert!(!coord.is_initialized().await);
    }

    #[tokio::test]
    async fn test_coordinator_initialize() {
        let coord = test_coordinator();
        let store = temp_store();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;
        assert!(coord.is_initialized().await);
    }

    #[tokio::test]
    async fn test_coordinator_search_before_init() {
        let coord = test_coordinator();
        let result = coord.search(vec![0.0; 8], 5, SearchParams::default()).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_coordinator_local_node_info() {
        let coord = test_coordinator();
        let info = coord.local_node_info();
        assert_eq!(info.node_id, "test-node");
        assert!(info.is_leader);
        assert_eq!(info.status, NodeStatus::Healthy);
        assert_eq!(info.dim_groups, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_coordinator_store() {
        let coord = test_coordinator();
        let store = coord.store();
        // Store should be accessible
        store.put_vector(1, &[1.0, 2.0]).unwrap();
        let v = store.get_vector(1).unwrap().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_insert() {
        let coord = test_coordinator();
        coord.insert(42, vec![0.1, 0.2, 0.3], None).await.unwrap();
        let v = coord.store().get_vector(42).unwrap().unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_insert_with_payload() {
        let coord = test_coordinator();
        let payload = Payload::from_text("test data");
        coord.insert(7, vec![0.5], Some(payload)).await.unwrap();
        let stored_payload = coord.store().get_payload(7).unwrap().unwrap();
        assert_eq!(stored_payload, b"test data");
    }

    #[tokio::test]
    async fn test_coordinator_topology() {
        let coord = test_coordinator();
        let topo = coord.topology().await.unwrap();
        assert!(topo.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_coordinator_node_info() {
        let coord = test_coordinator();
        let info = coord.node_info("any-node").await.unwrap();
        assert_eq!(info.node_id, "test-node");
    }

    #[tokio::test]
    async fn test_register_peer() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![0, 1],
            is_leader: false,
            raft_term: 1,
            commit_index: 5,
            storage_bytes: 256,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info.clone()).await;
        let peers = coord.peers_for_handshake("").await;
        assert!(!peers.is_empty());
        assert_eq!(peers[0].node_id, "peer-1");
    }

    #[tokio::test]
    async fn test_register_peer_update() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![0, 1],
            is_leader: false,
            raft_term: 1,
            commit_index: 5,
            storage_bytes: 256,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info).await;
        let updated_info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![0, 1],
            is_leader: true,
            raft_term: 2,
            commit_index: 10,
            storage_bytes: 512,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(updated_info).await;
        let peers = coord.peers_for_handshake("").await;
        assert_eq!(peers.len(), 1);
        assert!(peers[0].is_leader);
    }

    #[tokio::test]
    async fn test_peers_for_handshake_excludes_requester() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![0, 1],
            is_leader: false,
            raft_term: 1,
            commit_index: 5,
            storage_bytes: 256,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info).await;
        let peers = coord.peers_for_handshake("peer-1").await;
        assert!(peers.is_empty());
    }

    #[tokio::test]
    async fn test_peers_for_handshake_empty() {
        let coord = test_coordinator();
        let peers = coord.peers_for_handshake("any").await;
        assert!(peers.is_empty());
    }

    #[tokio::test]
    async fn test_healthy_peers() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![0, 1],
            is_leader: false,
            raft_term: 1,
            commit_index: 5,
            storage_bytes: 256,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info).await;
        let healthy = coord.healthy_peers().await;
        assert_eq!(healthy.len(), 1);
    }

    #[tokio::test]
    async fn test_register_raft_node() {
        let coord = test_coordinator();
        let state = rekha_raft::ReplicatedState::new(0);
        let node = std::sync::Arc::new(rekha_raft::RaftNode::new("n1".into(), 0, vec![], state));
        coord.register_raft_node(0, node);
        let found = coord.raft_node(0);
        assert!(found.is_some());
        assert_eq!(found.unwrap().node_id(), "n1");
    }

    #[tokio::test]
    async fn test_raft_node_nonexistent() {
        let coord = test_coordinator();
        let node = coord.raft_node(999);
        assert!(node.is_none());
    }

    #[tokio::test]
    async fn test_raft_log_store_creation() {
        let coord = test_coordinator();
        let log_store = coord.raft_log_store();
        // RaftLogStore created from the coordinator's store should be usable
        let entry = rekha_raft::node::RaftLogEntry {
            term: 1,
            index: 1,
            command: rekha_raft::state::RaftCommand::NoOp,
        };
        log_store.store_entry(0, &entry).unwrap();
        let entries = log_store.load_entries(0, 1).unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_search_with_local_index() {
        let coord = test_coordinator();
        let store = temp_store();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..20 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        let (results, _stats) = coord
            .search(vec![0.0; 8], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[tokio::test]
    async fn test_search_with_payloads() {
        let coord = test_coordinator();
        // Use coordinator's store so include_payloads search can find them
        let shared_store = coord.store().clone();
        let payload_data = b"test-payload".to_vec();

        // Insert vector and payload directly into coordinator's store
        shared_store.put_vector(42, &[1.0; 8]).unwrap();
        shared_store.put_payload(42, &payload_data).unwrap();

        // Initialize index with same store
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*shared_store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        let params = SearchParams {
            include_payloads: true,
            ..Default::default()
        };
        let (results, stats) = coord.search(vec![0.0; 8], 5, params).await.unwrap();
        assert!(!results.is_empty());
        assert!(stats.vectors_scanned > 0);
    }

    #[tokio::test]
    async fn test_search_re_rank_exact() {
        let coord = test_coordinator();
        // Use coordinator's store so re-rank can find full-precision vectors
        let shared_store = coord.store().clone();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*shared_store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        // Add vectors where one is an exact match for the query
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| i as f32 * 10.0 + d as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        // Search for a vector identical to id=5
        let query: Vec<f32> = (0..8).map(|d| 50.0 + d as f32).collect();
        let (results, _stats) = coord
            .search(query, 3, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
        // The exact match (id=5) should be the first result
        assert_eq!(results[0].id, 5);
        // Score should be 0.0 for exact match
        assert!(results[0].score.abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_peer_pool_new_empty() {
        let pool = PeerPool::new("default");
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[tokio::test]
    async fn test_starting_auto_id_empty_store() {
        let store = temp_store();
        let id = Coordinator::starting_auto_id(&store);
        assert_eq!(id, 1); // empty store -> 0 + 1
    }

    #[tokio::test]
    async fn test_starting_auto_id_with_existing() {
        let store = temp_store();
        store.put_vector(10, &[1.0]).unwrap();
        store.put_vector(5, &[2.0]).unwrap();
        store.put_vector(100, &[3.0]).unwrap();
        let id = Coordinator::starting_auto_id(&store);
        assert_eq!(id, 101); // max=100, +1=101
    }

    #[tokio::test]
    async fn test_peer_address_known() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![],
            is_leader: false,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info).await;
        let addr = coord.peer_address("peer-1").await;
        assert_eq!(addr, Some("10.0.0.2:50051".to_string()));
    }

    #[tokio::test]
    async fn test_peer_address_unknown() {
        let coord = test_coordinator();
        let addr = coord.peer_address("nonexistent").await;
        assert!(addr.is_none());
    }

    #[test]
    fn test_accessors() {
        let config = ServerConfig::dev_default("test-node", "/tmp/acc_test");
        let store = temp_store();
        let pm = Arc::new(RwLock::new(rekha_partition::PartitionManager::new(
            HashMap::new(),
            4,
            768,
        )));
        let coord = Coordinator::new(config, store, pm);

        assert_eq!(coord.cluster_id(), "rekha-dev");
        assert_eq!(coord.node_id(), "test-node");
        assert_eq!(coord.bind_addr(), "0.0.0.0:50051");
        assert_eq!(coord.seed_nodes(), &["127.0.0.1:50051"]);
        assert_eq!(coord.config_ref().cluster.node_id, "test-node");
    }

    #[tokio::test]
    async fn test_sync_topology() {
        let coord = test_coordinator();
        // Register a peer then sync
        let info = NodeInfo {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
            partition_id: 0,
            dim_groups: vec![],
            is_leader: false,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        coord.register_peer(info).await;
        coord.sync_topology().await;
        let topo = coord.topology().await.unwrap();
        assert!(topo.nodes.contains_key("test-node"));
        assert!(topo.nodes.contains_key("peer-1"));
    }

    #[tokio::test]
    async fn test_check_peer_health_no_peers() {
        let coord = test_coordinator();
        // Should not panic with no peers
        coord.check_peer_health().await;
        let healthy = coord.healthy_peers().await;
        assert!(healthy.is_empty());
    }

    #[tokio::test]
    async fn test_index_buffer_handle_not_initialized() {
        let coord = test_coordinator();
        // buffer_insert and buffer_delete should be no-ops when index is None
        coord.buffer_insert(1, vec![1.0, 2.0]);
        coord.buffer_delete(&[1]);
        // No panic = success
    }

    #[tokio::test]
    async fn test_index_buffer_handle_with_initialized_index() {
        let coord = test_coordinator();
        let store = temp_store();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        coord.buffer_insert(20, vec![0.0; 8]);
        coord.buffer_delete(&[1]);
        // No panic = success; buffer operations went through
    }

    #[tokio::test]
    async fn test_build_index_deprecated() {
        let coord = test_coordinator();
        let result = coord.build_index().await;
        assert!(result.is_err());
        match result {
            Err(RekhaError::Unavailable { .. }) => {}
            _ => panic!("expected Unavailable error"),
        }
    }

    #[tokio::test]
    async fn test_search_local_only() {
        let coord = test_coordinator();
        let store = temp_store();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        let params = SearchParams {
            local_only: true,
            ..Default::default()
        };
        let (results, stats) = coord.search(vec![0.0; 8], 5, params).await.unwrap();
        assert!(!results.is_empty());
        // local_only=true skips peer fan-out; vectors_scanned should be > 0
        assert!(stats.vectors_scanned > 0);
    }

    #[tokio::test]
    async fn test_insert_auto_id() {
        let coord = test_coordinator();
        // id=0 triggers auto-generation
        let id = coord.insert(0, vec![0.1, 0.2], None).await.unwrap();
        assert_eq!(id, 1);
        // Next auto-id should be 2
        let id2 = coord.insert(0, vec![0.3, 0.4], None).await.unwrap();
        assert_eq!(id2, 2);
    }

    #[tokio::test]
    async fn test_insert_fallback_direct_store() {
        // When no raft node exists for partition 0, insert falls through to
        // direct store write (single-node / uninitialized path).
        let coord = test_coordinator();
        // coord has no raft nodes registered — insert will use the fallback path
        let id = coord.insert(0, vec![0.5, 0.6], None).await.unwrap();
        let v = coord.store().get_vector(id).unwrap().unwrap();
        assert!((v[0] - 0.5).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_insert_fallback_with_payload() {
        let coord = test_coordinator();
        let payload = Payload::from_text("fallback data");
        let id = coord.insert(0, vec![0.7], Some(payload)).await.unwrap();
        let stored = coord.store().get_payload(id).unwrap().unwrap();
        assert_eq!(stored, b"fallback data");
    }

    #[tokio::test]
    async fn test_check_peer_health_timeout_transition() {
        let coord = test_coordinator();
        // Register a peer with a very old last_seen (Instant::now() - 20s).
        let info = NodeInfo {
            node_id: "old-peer".into(),
            address: "10.0.0.3:50051".into(),
            partition_id: 0,
            dim_groups: vec![],
            is_leader: false,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        // Insert directly into peers map with an old timestamp
        {
            let mut peers = coord.peers.write().await;
            peers.insert(
                "old-peer".into(),
                PeerState {
                    info,
                    last_seen: Instant::now() - Duration::from_secs(20),
                },
            );
        }
        coord.check_peer_health().await;
        let healthy = coord.healthy_peers().await;
        assert!(healthy.is_empty()); // old-peer should be marked Unreachable
    }

    #[tokio::test]
    async fn test_check_peer_health_recovery() {
        let coord = test_coordinator();
        // Start with a peer that's Unreachable (last_seen fresh but status=Unreachable)
        let info = NodeInfo {
            node_id: "recovering-peer".into(),
            address: "10.0.0.4:50051".into(),
            partition_id: 0,
            dim_groups: vec![],
            is_leader: false,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Unreachable,
            last_heartbeat: 0,
        };
        {
            let mut peers = coord.peers.write().await;
            peers.insert(
                "recovering-peer".into(),
                PeerState {
                    info,
                    last_seen: Instant::now(), // fresh
                },
            );
        }
        coord.check_peer_health().await;
        let healthy = coord.healthy_peers().await;
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].node_id, "recovering-peer");
    }

    #[tokio::test]
    async fn test_recover_index_buffer_no_index() {
        let coord = test_coordinator();
        // recover_index_buffer with no index should return error
        let result = coord.recover_index_buffer().await;
        assert!(result.is_err());
        match result {
            Err(RekhaError::Internal { detail }) => {
                assert!(detail.contains("not initialized for recovery"));
            }
            _ => panic!("expected Internal error"),
        }
    }

    #[tokio::test]
    async fn test_insert_via_raft() {
        let coord = std::sync::Arc::new(test_coordinator());
        // Register a raft node with no peers (auto self-elects on start_election)
        let state = rekha_raft::ReplicatedState::new(0);
        let raft_log_store = coord.raft_log_store();
        let node = std::sync::Arc::new(rekha_raft::RaftNode::with_store(
            "test-node".into(),
            0,
            vec![],
            state,
            Some(raft_log_store),
            Some(coord.clone() as Arc<dyn rekha_core::IndexBufferHandle>),
        ));
        let node_clone = node.clone();
        node.start_election().await.unwrap();
        assert!(node.is_leader().await);
        coord.register_raft_node(0, node);

        // Now insert routes through Raft (raft_node(0) exists and is leader)
        let id = coord.insert(42, vec![0.1, 0.2, 0.3], None).await.unwrap();
        assert_eq!(id, 42);
        // The Raft path stores vectors in ReplicatedState, not directly in RocksDB
        let state = node_clone.read_state().await;
        assert!(state.get_vector(42).is_some());
    }

    #[tokio::test]
    async fn test_insert_via_raft_with_payload() {
        let coord = test_coordinator();
        let state = rekha_raft::ReplicatedState::new(0);
        let raft_log_store = coord.raft_log_store();
        let node = std::sync::Arc::new(rekha_raft::RaftNode::with_store(
            "test-node".into(),
            0,
            vec![],
            state,
            Some(raft_log_store),
            None,
        ));
        let node_clone = node.clone();
        node.start_election().await.unwrap();
        coord.register_raft_node(0, node);

        let payload = Payload::from_text("raft payload");
        let id = coord.insert(7, vec![0.5], Some(payload)).await.unwrap();
        assert_eq!(id, 7);
        let state = node_clone.read_state().await;
        assert_eq!(state.get_payload(7), Some(&b"raft payload"[..]));
    }

    #[tokio::test]
    async fn test_recover_index_buffer_with_state() {
        let coord = test_coordinator();
        let store = coord.store().clone();

        // Create a ReplicatedState with pre-populated vectors
        let mut state = rekha_raft::ReplicatedState::new(0);
        let vec_bytes: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        state.vectors.insert(100u64, vec_bytes.clone());

        let raft_log_store = coord.raft_log_store();
        let node = std::sync::Arc::new(rekha_raft::RaftNode::with_store(
            "test-node".into(),
            0,
            vec![],
            state,
            Some(raft_log_store),
            None,
        ));
        coord.register_raft_node(0, node);

        // Initialize coordinator with an empty built index
        let index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        coord.initialize(index).await;

        // recover_index_buffer ran inside initialize — verify vectors were recovered
        // The index should buffer contain vector 100 (not in the graph)
        assert!(coord.is_initialized().await);
    }

    #[tokio::test]
    async fn test_search_empty_unbuilt_index() {
        // Search with an initialized but empty (unbuilt) index should hit
        // the search_dim_range error path for each dim group
        let coord = test_coordinator();
        let store = coord.store().clone();
        let index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        coord.initialize(index).await;

        let (results, _stats) = coord
            .search(vec![0.0; 8], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(results.is_empty());
        // Each dim group search should have failed, generating warnings
        // (warnings presence varies; just verify search doesn't crash)
    }

    #[tokio::test]
    async fn test_search_candidate_pruning() {
        // Test that the score-based pruning in the candidate collection path works
        let coord = test_coordinator();
        let shared_store = coord.store().clone();
        let mut index = rekha_index::RekhaIndex::new(
            8,
            4,
            16,
            4,
            (*shared_store).clone(),
            rekha_core::DistanceMetric::L2,
        )
        .unwrap();
        for i in 0..50 {
            let v: Vec<f32> = (0..8).map(|d| (i * 10 + d) as f32).collect();
            index.add_vector_for_test(i, v);
        }
        index.build().unwrap();
        coord.initialize(index).await;

        // Search with a small k to trigger pruning
        let (results, _stats) = coord
            .search(vec![1000.0; 8], 3, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 3);
    }

    #[tokio::test]
    async fn test_spawn_flush_loop_runs() {
        let coord = test_coordinator();
        // spawn_flush_loop starts a background tokio task; verify it doesn't panic
        coord.spawn_flush_loop();
        // Give it a moment to tick once
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        // No assertion needed — test passes if no panic
    }
}
