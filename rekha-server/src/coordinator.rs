use rekha_core::{
    ClusterTopology, CollectionMetadata, DistanceMetric, IndexBufferHandle, NodeInfo, NodeStatus,
    Payload, RekhaError, ScoredPoint, SearchParams, SearchStats, VectorIndex, VectorStoreBackend,
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

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.clients.len()
    }
}

/// Per-collection state: index, storage, Raft nodes, and auto-ID.
pub(crate) struct CollectionCtx {
    #[allow(dead_code)]
    pub metadata: CollectionMetadata,
    pub store: RocksVectorStore,
    pub index: Arc<RwLock<RekhaIndex>>,
    pub shard_raft_nodes: DashMap<u64, Arc<RaftNode>>,
    pub peer_pool: RwLock<PeerPool>,
    pub next_auto_id: AtomicU64,
    #[allow(dead_code)]
    pub dim: usize,
}

/// An IndexBufferHandle that routes to a single collection's index.
/// Used as the callback for per-collection data Raft nodes.
pub(crate) struct PerCollectionIndexHandle {
    #[allow(dead_code)]
    pub collection_name: String,
    pub store: RocksVectorStore,
    pub index: Arc<RwLock<RekhaIndex>>,
}

impl IndexBufferHandle for PerCollectionIndexHandle {
    fn buffer_insert(&self, id: u64, vector: Vec<f32>, payload: Option<Vec<u8>>) {
        let _ = self.store.put_vector(id, &vector);
        if let Some(ref p) = payload {
            let _ = self.store.put_payload(id, p);
        }
        if let Ok(idx) = self.index.try_read() {
            idx.buffer_insert(id, vector);
        }
    }

    fn buffer_delete(&self, ids: &[u64]) {
        let _ = self.store.delete(ids);
        if let Ok(idx) = self.index.try_read() {
            idx.buffer_delete(ids);
        }
    }
}

/// The distributed query coordinator.
///
/// Responsibilities:
/// - Route search queries to the appropriate collections and partitions
/// - Fan out to dimension groups for partial distance computation
/// - Merge partial results from multiple nodes
/// - Apply early-stop pruning across dimension groups
/// - Manage cluster topology and node health
pub struct Coordinator {
    /// Server configuration.
    config: ServerConfig,
    /// Local storage backend (raw — for metadata and namespace-derived stores).
    store: Arc<RocksVectorStore>,
    /// Metadata Raft node (partition_id = METADATA_PARTITION_ID).
    metadata_raft_node: RwLock<Option<Arc<RaftNode>>>,
    /// Per-collection contexts.
    collections: RwLock<HashMap<String, Arc<CollectionCtx>>>,
    /// Partition topology manager.
    #[allow(dead_code)]
    partition_manager: Arc<RwLock<PartitionManager>>,
    /// All Raft nodes across all collections (flat map for background timers).
    pub raft_nodes: DashMap<u64, Arc<RaftNode>>,
    /// Cluster topology (peer nodes).
    topology: Arc<RwLock<ClusterTopology>>,
    /// Whether this coordinator is initialized.
    initialized: Arc<RwLock<bool>>,
    /// Peer node tracking (node_id → state).
    peers: Arc<RwLock<HashMap<String, PeerState>>>,
}

impl Coordinator {
    /// Create a new coordinator.
    pub fn new(
        config: ServerConfig,
        store: Arc<RocksVectorStore>,
        partition_manager: Arc<RwLock<PartitionManager>>,
    ) -> Self {
        Self {
            config,
            store,
            metadata_raft_node: RwLock::new(None),
            collections: RwLock::new(HashMap::new()),
            partition_manager,
            raft_nodes: DashMap::new(),
            topology: Arc::new(RwLock::new(ClusterTopology {
                cluster_id: String::new(),
                nodes: HashMap::new(),
                partition_map: HashMap::new(),
            })),
            initialized: Arc::new(RwLock::new(false)),
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Initialize the coordinator. Called after metadata Raft node and default collection are set up.
    pub async fn initialize(&self) {
        *self.initialized.write().await = true;
        info!("Coordinator initialized");
    }

    /// Set the metadata Raft node (partition_id = METADATA_PARTITION_ID).
    pub async fn set_metadata_raft(&self, node: Arc<RaftNode>) {
        *self.metadata_raft_node.write().await = Some(node);
    }

    /// Get the metadata Raft node.
    pub async fn metadata_raft_node(&self) -> Option<Arc<RaftNode>> {
        self.metadata_raft_node.read().await.clone()
    }

    /// Create a new collection with the given name and dimension.
    /// Proposes to the metadata Raft group, then initializes the local context.
    pub async fn create_collection(&self, name: &str, dim: usize) -> Result<(), RekhaError> {
        // Validate collection doesn't already exist locally.
        {
            let cols = self.collections.read().await;
            if cols.contains_key(name) {
                return Err(RekhaError::CollectionAlreadyExists(name.to_string()));
            }
        }

        // Build a CollectionConfig using server defaults.
        let config = rekha_core::CollectionConfig {
            dim: dim as u32,
            num_vector_shards: self.config.partition.num_vector_shards,
            replication_factor: self.config.partition.replication_factor as u64,
            num_dim_groups: self.config.partition.num_dim_groups,
            dim_group_size: self.config.partition.dim_group_size as u32,
            graph_degree: self.config.index.graph_degree as u32,
            search_list_size: self.config.index.search_list_size as u32,
            pq_num_sub_vectors: self.config.index.pq_num_sub_vectors as u32,
            pq_num_centroids: self.config.index.pq_num_centroids as u32,
            re_rank_k: self.config.index.re_rank_k as u32,
        };

        // Propose to the metadata Raft group.
        let meta_node = self.metadata_raft_node.read().await.clone();
        if let Some(ref meta_node) = meta_node {
            let cmd = rekha_raft::state::RaftCommand::CreateCollection {
                name: name.to_string(),
                dim,
                config: config.clone(),
            };
            meta_node.propose(cmd).await?;
        }

        // Initialize the local collection context.
        self.init_collection_ctx(name, dim, config).await?;
        Ok(())
    }

    /// Drop a collection and all its data.
    pub async fn drop_collection(&self, name: &str) -> Result<(), RekhaError> {
        if name == "default" {
            return Err(RekhaError::InvalidArgument(
                "cannot drop the default collection".into(),
            ));
        }

        // Propose to the metadata Raft group.
        let meta_node = self.metadata_raft_node.read().await.clone();
        if let Some(ref meta_node) = meta_node {
            let cmd = rekha_raft::state::RaftCommand::DropCollection {
                name: name.to_string(),
            };
            meta_node.propose(cmd).await?;
        }

        // Remove the local context and clean up data.
        let ctx = {
            let mut cols = self.collections.write().await;
            cols.remove(name)
        };

        if let Some(ctx) = ctx {
            // Delete all data in the collection's namespace from RocksDB.
            if let Err(e) = ctx.store.delete_all_in_namespace() {
                tracing::warn!("Failed to delete collection data from store: {e}");
            }
            // Remove raft nodes from the global registry.
            for entry in ctx.shard_raft_nodes.iter() {
                let pid = *entry.key();
                self.raft_nodes.remove(&pid);
            }
            info!("Dropped collection: {name}");
        }

        Ok(())
    }

    /// List all collections from the metadata Raft state.
    pub async fn list_collections(&self) -> Result<Vec<CollectionMetadata>, RekhaError> {
        let meta_node = self.metadata_raft_node.read().await.clone();
        if let Some(ref meta_node) = meta_node {
            let state = meta_node.read_state().await;
            let mut result: Vec<CollectionMetadata> = state.collections.values().cloned().collect();
            result.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    /// Check if a collection exists locally.
    pub async fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().await.contains_key(name)
    }

    /// Initialize a local collection context (index + store + Raft nodes).
    pub(crate) async fn init_collection_ctx(
        &self,
        name: &str,
        dim: usize,
        config: rekha_core::CollectionConfig,
    ) -> Result<Arc<CollectionCtx>, RekhaError> {
        // Create a namespaced store for this collection.
        let coll_store = RocksVectorStore::from_db(self.store.db().clone(), Some(name.to_string()));

        // Build the Vamana + PQ index.
        let index = RekhaIndex::new(
            dim,
            config.pq_num_sub_vectors as usize,
            config.pq_num_centroids as usize,
            config.graph_degree as usize,
            coll_store.clone(),
            rekha_core::DistanceMetric::L2,
        )?;
        let index = Arc::new(RwLock::new(index));

        // Create the per-collection index handle for Raft callbacks.
        let index_handle = Arc::new(PerCollectionIndexHandle {
            collection_name: name.to_string(),
            store: coll_store.clone(),
            index: index.clone(),
        });

        // Build the peer pool for this collection.
        let peer_pool = RwLock::new(PeerPool::new(name));

        // Compute starting auto-ID from existing data in this namespace.
        let starting_id = match coll_store.iter_ids() {
            Ok(ids) => ids.iter().max().copied().unwrap_or(0) + 1,
            Err(_) => 1,
        };

        // Create per-shard data Raft nodes for this collection.
        let raft_log_store = RaftLogStore::with_namespace(self.store.clone(), name.to_string());
        let raft_network = Arc::new(crate::raft_network::GrpcRaftNetwork::new());
        let shard_raft_nodes: DashMap<u64, Arc<RaftNode>> = DashMap::new();

        let num_shards = self.config.partition.num_vector_shards;
        for shard in 0..num_shards {
            let state = rekha_raft::ReplicatedState::new(shard);
            let node_id = &self.config.cluster.node_id;
            let peers: Vec<String> = self
                .config
                .cluster
                .seed_nodes
                .iter()
                .filter(|s| !s.starts_with(node_id))
                .cloned()
                .collect();
            let is_single_node = peers.is_empty();
            let raft_node = Arc::new(rekha_raft::RaftNode::with_store(
                self.config.cluster.node_id.clone(),
                shard,
                peers,
                state,
                Some(raft_log_store.clone()),
                Some(index_handle.clone() as Arc<dyn IndexBufferHandle>),
                Some(raft_network.clone() as Arc<dyn rekha_raft::RaftPeerNetwork>),
            ));
            shard_raft_nodes.insert(shard, raft_node.clone());
            self.raft_nodes.insert(shard, raft_node.clone());
            // Auto-elect in single-node mode (no peers).
            if is_single_node {
                if let Err(e) = raft_node.start_election().await {
                    tracing::warn!("Failed to self-elect raft node for shard {shard}: {e}");
                }
            }
        }

        let metadata = CollectionMetadata {
            name: name.to_string(),
            dim,
            config,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let ctx = Arc::new(CollectionCtx {
            metadata,
            store: coll_store,
            index,
            shard_raft_nodes,
            peer_pool,
            next_auto_id: AtomicU64::new(starting_id),
            dim,
        });

        // Register in the collections map.
        self.collections
            .write()
            .await
            .insert(name.to_string(), ctx.clone());

        // Spawn per-collection flush loop.
        Self::spawn_collection_flush_loop(ctx.clone());

        info!("Initialized collection: {name} (dim={dim})");
        Ok(ctx)
    }

    /// Get a collection context by name.
    pub(crate) async fn get_collection(
        &self,
        name: &str,
    ) -> Result<Arc<CollectionCtx>, RekhaError> {
        self.collections
            .read()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| RekhaError::CollectionNotFound(name.to_string()))
    }

    /// Spawn a per-collection background task that periodically flushes the insert buffer.
    fn spawn_collection_flush_loop(ctx: Arc<CollectionCtx>) {
        let index = ctx.index.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(1000)).await;
                let mut idx = index.write().await;
                if idx.should_flush() || idx.buffer_len() > 0 {
                    let buf_len = idx.buffer_len();
                    if let Err(e) = idx.flush_buffer() {
                        tracing::warn!("Buffer flush failed: {e} (buffer: {buf_len} vectors)");
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

    /// Refresh the peer connection pools for all collections.
    async fn refresh_peer_pool(&self) {
        let healthy = self.healthy_peers().await;
        let external: Vec<NodeInfo> = healthy
            .into_iter()
            .filter(|p| p.node_id != self.config.cluster.node_id)
            .collect();
        let cols = self.collections.read().await;
        for ctx in cols.values() {
            let mut pool = ctx.peer_pool.write().await;
            pool.refresh(&external).await;
        }
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

impl Coordinator {
    pub async fn search(
        &self,
        collection_name: &str,
        query: Vec<f32>,
        k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let ctx = self.get_collection(collection_name).await?;
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let index = ctx.index.read().await;

        // Phase 1: Local full-precision approximate search.
        let mut candidates: Vec<ScoredPoint> = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        match index.search(&query, k * 2, &params) {
            Ok((ids, dists)) => {
                stats.vectors_scanned += ids.len() as u64;
                for (i, id) in ids.iter().enumerate() {
                    if !seen_ids.insert(*id) {
                        continue;
                    }
                    candidates.push(ScoredPoint {
                        id: *id,
                        score: dists.get(i).copied().unwrap_or(f32::MAX),
                        payload: if params.include_payloads {
                            ctx.store
                                .get_payload(*id)
                                .ok()
                                .flatten()
                                .map(Payload::from_bytes)
                        } else {
                            None
                        },
                    });
                }
            }
            Err(e) => {
                stats.warnings.push(format!("local search failed: {e}"));
            }
        }
        drop(index);

        // Phase 2: Fan out to peer nodes (skip for local-only searches).
        if !params.local_only {
            let has_peers = { !ctx.peer_pool.read().await.is_empty() };
            if has_peers {
                let mut pool = ctx.peer_pool.write().await;
                let mut peer_params = params.clone();
                peer_params.local_only = true;
                let (peer_results, peer_stats) = pool.search_fan_out(&query, k, &peer_params).await;
                stats.nodes_contacted = 1 + peer_stats.nodes_contacted;
                stats.vectors_scanned += peer_stats.vectors_scanned;
                stats.warnings.extend(peer_stats.warnings);
                for r in peer_results {
                    if seen_ids.insert(r.id) {
                        candidates.push(r);
                    }
                }
            } else {
                stats.nodes_contacted = 1;
            }
        } else {
            stats.nodes_contacted = 1;
        }

        // Phase 3: Re-rank with exact distances.
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k * 2);

        let metric = DistanceMetric::L2;
        for candidate in candidates.iter_mut().take(k * 2) {
            let id = candidate.id;
            if let Ok(Some(full_vec)) = ctx.store.get_vector(id) {
                let exact_dist = metric.distance(&full_vec, &query);
                candidate.score = exact_dist;
            }
        }
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k);

        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok((candidates, stats))
    }

    pub async fn insert(
        &self,
        collection_name: &str,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Payload>,
    ) -> Result<u64, RekhaError> {
        let ctx = self.get_collection(collection_name).await?;

        let id = if id == 0 {
            ctx.next_auto_id.fetch_add(1, Ordering::SeqCst)
        } else {
            id
        };

        // Route through this collection's data Raft node (partition 0 if single-shard).
        let shard = id % self.config.partition.num_vector_shards;
        if let Some(raft_node) = ctx.shard_raft_nodes.get(&shard) {
            let cmd = rekha_raft::state::RaftCommand::Insert {
                collection_name: collection_name.to_string(),
                id,
                vector,
                payload: payload.as_ref().map(|p| p.data.clone()),
            };
            raft_node.propose(cmd).await?;
            return Ok(id);
        }

        Err(RekhaError::Unavailable {
            detail: format!("no raft node for shard {shard} in collection {collection_name}"),
        })
    }

    pub async fn delete_ids(
        &self,
        collection_name: &str,
        ids: Vec<u64>,
    ) -> Result<u64, RekhaError> {
        let ctx = self.get_collection(collection_name).await?;

        // For single-shard, route to partition 0.
        if let Some(raft_node) = ctx.shard_raft_nodes.get(&0) {
            let cmd = rekha_raft::state::RaftCommand::Delete {
                collection_name: collection_name.to_string(),
                ids,
            };
            raft_node.propose(cmd).await?;
            return Ok(0);
        }

        Err(RekhaError::Unavailable {
            detail: format!("no raft node for collection {collection_name}"),
        })
    }

    /// Transfer a shard's data to a target node.
    /// Reads all vector IDs from the store, batches them, and sends via client.
    pub async fn transfer_shard(
        &self,
        target_addr: &str,
        shard_id: u64,
    ) -> Result<u64, RekhaError> {
        use rekha_client::RekhaClient;

        let client = RekhaClient::connect(&[target_addr.to_string()]).await?;
        let ids = self.store.iter_ids()?;
        let mut transferred = 0u64;

        // Batch transfer: collect and send in groups.
        let mut batch = Vec::new();
        for id in ids {
            if id % self.config.partition.num_vector_shards != shard_id {
                continue; // Not our shard
            }
            if let Ok(Some(vector)) = self.store.get_vector(id) {
                let payload = self.store.get_payload(id).ok().flatten();
                batch.push((id, vector, payload));
                if batch.len() >= 100 {
                    for (bid, bvec, bpayload) in batch.drain(..) {
                        let _ = client.insert(bid, bvec, "default", bpayload).await;
                        transferred += 1;
                    }
                }
            }
        }
        // Drain remaining.
        for (id, vector, payload) in batch {
            let _ = client.insert(id, vector, "default", payload).await;
            transferred += 1;
        }

        Ok(transferred)
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

    fn temp_dir() -> String {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        format!("/tmp/rekha_test_{}", n)
    }

    async fn init_coordinator() -> Coordinator {
        let dir = temp_dir();
        let mut config = ServerConfig::dev_default("test-node", &dir);
        config.cluster.seed_nodes.clear(); // single-node test
        let store = temp_store();
        let pm = Arc::new(RwLock::new(rekha_partition::PartitionManager::new(
            HashMap::new(),
            1,
        )));
        let coord = Coordinator::new(config, store, pm);
        let meta_state = rekha_raft::ReplicatedState::new(rekha_core::METADATA_PARTITION_ID);
        let meta_raft = Arc::new(rekha_raft::RaftNode::new(
            "test-node".into(),
            rekha_core::METADATA_PARTITION_ID,
            vec![],
            meta_state,
        ));
        coord.set_metadata_raft(meta_raft).await;
        if let Some(meta) = coord.metadata_raft_node().await {
            meta.start_election().await.unwrap();
        }
        coord.create_collection("default", 256).await.unwrap();
        // Self-elect all data Raft nodes for single-node tests.
        let raft_ids: Vec<u64> = coord.raft_nodes.iter().map(|e| *e.key()).collect();
        for pid in raft_ids {
            if let Some(node) = coord.raft_node(pid) {
                if let Err(e) = node.start_election().await {
                    eprintln!("WARN: failed to self-elect raft node {pid}: {e}");
                }
            }
        }
        coord.initialize().await;
        coord
    }

    fn test_coordinator() -> Coordinator {
        let config = ServerConfig::dev_default("test-node", "/tmp/rekha_coord_test");
        let store = temp_store();
        let pm = Arc::new(RwLock::new(rekha_partition::PartitionManager::new(
            HashMap::new(),
            1,
        )));
        Coordinator::new(config, store, pm)
    }

    #[tokio::test]
    async fn test_new() {
        assert!(!test_coordinator().is_initialized().await);
    }

    #[tokio::test]
    async fn test_initialize() {
        let coord = init_coordinator().await;
        assert!(coord.is_initialized().await);
        assert!(coord.collection_exists("default").await);
    }

    #[tokio::test]
    async fn test_create_list_drop() {
        let coord = init_coordinator().await;
        assert_eq!(coord.list_collections().await.unwrap().len(), 1);
        coord.create_collection("imgs", 256).await.unwrap();
        assert_eq!(coord.list_collections().await.unwrap().len(), 2);
        coord.drop_collection("imgs").await.unwrap();
        assert_eq!(coord.list_collections().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_no_drop_default() {
        assert!(init_coordinator()
            .await
            .drop_collection("default")
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_no_dup_collection() {
        assert!(init_coordinator()
            .await
            .create_collection("default", 256)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_insert_and_fetch() {
        let coord = init_coordinator().await;
        coord
            .insert("default", 42, vec![0.1, 0.2], None)
            .await
            .unwrap();
        let v = coord
            .get_collection("default")
            .await
            .unwrap()
            .store
            .get_vector(42)
            .unwrap()
            .unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_insert_with_payload() {
        let coord = init_coordinator().await;
        let p = Payload::from_text("hello");
        coord
            .insert("default", 7, vec![0.5], Some(p))
            .await
            .unwrap();
        let stored = coord
            .get_collection("default")
            .await
            .unwrap()
            .store
            .get_payload(7)
            .unwrap()
            .unwrap();
        assert_eq!(stored, b"hello");
    }

    #[tokio::test]
    async fn test_search_needs_init() {
        assert!(test_coordinator()
            .search("default", vec![0.0; 256], 5, SearchParams::default())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_insert_auto_id() {
        let coord = init_coordinator().await;
        assert_eq!(
            coord
                .insert("default", 0, vec![0.1, 0.2], None)
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            coord
                .insert("default", 0, vec![0.3, 0.4], None)
                .await
                .unwrap(),
            2
        );
    }

    #[tokio::test]
    async fn test_namespace_isolation() {
        let coord = init_coordinator().await;
        coord.create_collection("a", 256).await.unwrap();
        coord.create_collection("b", 256).await.unwrap();
        coord.insert("a", 1, vec![0.1; 256], None).await.unwrap();
        coord.insert("b", 1, vec![0.9; 256], None).await.unwrap();
        let va = coord
            .get_collection("a")
            .await
            .unwrap()
            .store
            .get_vector(1)
            .unwrap()
            .unwrap();
        let vb = coord
            .get_collection("b")
            .await
            .unwrap()
            .store
            .get_vector(1)
            .unwrap()
            .unwrap();
        assert!((va[0] - 0.1).abs() < 1e-6);
        assert!((vb[0] - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_search_local() {
        let coord = init_coordinator().await;
        let ctx = coord.get_collection("default").await.unwrap();
        let mut idx = ctx.index.write().await;
        for i in 0..20 {
            let v: Vec<f32> = (0..256).map(|d| (i * 256 + d) as f32).collect();
            idx.add_vector_for_test(i, v);
        }
        idx.build().unwrap();
        drop(idx);
        let (r, _) = coord
            .search("default", vec![0.0; 256], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(!r.is_empty() && r.len() <= 5);
    }

    #[tokio::test]
    async fn test_peer_mgmt() {
        let coord = test_coordinator();
        let info = NodeInfo {
            node_id: "n1".into(),
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
        assert_eq!(coord.peers_for_handshake("").await.len(), 1);
        assert!(coord.peers_for_handshake("n1").await.is_empty());
        assert_eq!(
            coord.peer_address("n1").await,
            Some("10.0.0.2:50051".into())
        );
    }

    #[tokio::test]
    async fn test_health() {
        test_coordinator().check_peer_health().await;
    }

    #[test]
    fn test_accessors() {
        let config = ServerConfig::dev_default("t", "/tmp/t");
        let store = temp_store();
        let coord = Coordinator::new(
            config,
            store,
            Arc::new(RwLock::new(PartitionManager::new(HashMap::new(), 1))),
        );
        assert_eq!(coord.node_id(), "t");
    }

    #[tokio::test]
    async fn test_sync_topology() {
        let coord = test_coordinator();
        coord
            .register_peer(NodeInfo {
                node_id: "p1".into(),
                address: "10.0.0.2:50051".into(),
                partition_id: 0,
                dim_groups: vec![],
                is_leader: false,
                raft_term: 1,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            })
            .await;
        coord.sync_topology().await;
        assert!(coord.topology().await.unwrap().nodes.contains_key("p1"));
    }
}
