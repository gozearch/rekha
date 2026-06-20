use rekha_core::{
    ClusterTopology, CollectionConfig, CollectionMeta, DistanceMetric, IndexBufferHandle, NodeInfo,
    NodeStatus, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats, VectorIndex,
    VectorStoreBackend,
};
use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_raft::{RaftLogStore, RaftNode};
use rekha_storage::RocksVectorStore;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::RwLock as SyncRwLock;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::ServerConfig;
use crate::peer::{PeerPool, PeerState};

const PEER_TIMEOUT: Duration = Duration::from_secs(10);

/// Per-collection state held by the coordinator.
pub struct CollectionState {
    pub config: CollectionConfig,
    pub meta: CollectionMeta,
    pub store: Arc<RocksVectorStore>,
    pub index: Arc<RwLock<Option<RekhaIndex>>>,
    pub raft_nodes: DashMap<u64, Arc<RaftNode>>,
    pub raft_log_store: RaftLogStore,
    pub next_auto_id: AtomicU64,
}

/// Thin handle that routes IndexBufferHandle calls to a specific collection.
struct PerCollectionHandle {
    collection_name: String,
    collections: Arc<DashMap<String, CollectionState>>,
}

impl IndexBufferHandle for PerCollectionHandle {
    fn buffer_insert(&self, id: u64, vector: Vec<f32>) {
        if let Some(state) = self.collections.get(&self.collection_name) {
            if let Ok(idx) = state.index.try_read() {
                if let Some(ref idx) = *idx {
                    idx.buffer_insert(id, vector);
                }
            }
        }
    }

    fn buffer_delete(&self, ids: &[u64]) {
        if let Some(state) = self.collections.get(&self.collection_name) {
            if let Ok(idx) = state.index.try_read() {
                if let Some(ref idx) = *idx {
                    idx.buffer_delete(ids);
                }
            }
        }
    }
}

/// Handle for the system Raft group. Materializes collections on commit.
pub(crate) struct SystemRaftHandle {
    pub(crate) coordinator: *const Coordinator,
}

unsafe impl Send for SystemRaftHandle {}
unsafe impl Sync for SystemRaftHandle {}

impl IndexBufferHandle for SystemRaftHandle {
    fn buffer_insert(&self, _id: u64, _vector: Vec<f32>) {}

    fn buffer_delete(&self, _ids: &[u64]) {}

    fn notify_create_collection(&self, name: &str, config: &CollectionConfig) {
        let coord = unsafe { &*self.coordinator };
        let name = name.to_string();
        let config = config.clone();
        tokio::spawn(async move {
            if coord.collections.contains_key(&name) {
                return;
            }
            let _ = coord.create_collection(&name, config).await;
        });
    }

    fn notify_drop_collection(&self, name: &str) {
        let coord = unsafe { &*self.coordinator };
        let name = name.to_string();
        tokio::spawn(async move {
            let _ = coord.drop_collection(&name).await;
        });
    }
}

/// The distributed query coordinator, now collection-aware.
/// System Raft partition ID used for collection metadata replication.
pub const SYSTEM_PARTITION_ID: u64 = u64::MAX;

pub struct Coordinator {
    pub config: ServerConfig,
    pub store: Arc<RocksVectorStore>,
    pub collections: Arc<DashMap<String, CollectionState>>,
    pub partition_manager: Arc<RwLock<PartitionManager>>,
    topology: Arc<RwLock<ClusterTopology>>,
    initialized: Arc<RwLock<bool>>,
    peers: Arc<RwLock<HashMap<String, PeerState>>>,
    peer_pool: Arc<RwLock<PeerPool>>,
    system_raft_node: SyncRwLock<Option<Arc<RaftNode>>>,
}

impl Coordinator {
    pub fn new(
        config: ServerConfig,
        store: Arc<RocksVectorStore>,
        partition_manager: Arc<RwLock<PartitionManager>>,
    ) -> Self {
        Self {
            config,
            store,
            collections: Arc::new(DashMap::new()),
            partition_manager,
            topology: Arc::new(RwLock::new(ClusterTopology {
                cluster_id: String::new(),
                nodes: HashMap::new(),
                partition_map: HashMap::new(),
            })),
            initialized: Arc::new(RwLock::new(false)),
            peers: Arc::new(RwLock::new(HashMap::new())),
            peer_pool: Arc::new(RwLock::new(PeerPool::new("default"))),
            system_raft_node: SyncRwLock::new(None),
        }
    }

    /// Create a "default" collection for backward compat.
    /// Used when server starts with no explicit collections.
    /// Build a CollectionState from config + meta (shared by create and initialize).
    fn build_collection_state(
        &self,
        name: &str,
        config: &CollectionConfig,
        meta: CollectionMeta,
    ) -> Result<CollectionState, RekhaError> {
        let namespaced_store = self.namespaced_store(name);
        let dim = (config.dim_group_size as usize) * (config.num_dim_groups as usize);
        let index = RekhaIndex::new(
            dim,
            config.pq_num_sub_vectors as usize,
            config.pq_num_centroids as usize,
            config.graph_degree as usize,
            (*namespaced_store).clone(),
            config.distance_metric,
        )?;
        Ok(CollectionState {
            config: config.clone(),
            meta,
            store: namespaced_store,
            index: Arc::new(RwLock::new(Some(index))),
            raft_nodes: DashMap::new(),
            raft_log_store: RaftLogStore::with_namespace(self.store.clone(), name.into()),
            next_auto_id: AtomicU64::new(1),
        })
    }

    pub async fn create_default_collection(&self) -> Result<(), RekhaError> {
        if self.collections.contains_key("default") {
            return Ok(());
        }
        let meta = self.load_or_create_meta("default").await;
        let state = self.build_collection_state("default", &meta.config, meta.clone())?;
        self.collections.insert("default".into(), state);
        self.spawn_flush_loop("default");
        info!("Created default collection");
        Ok(())
    }

    /// Create a named collection with given config.
    pub async fn create_collection(
        &self,
        name: &str,
        config: CollectionConfig,
    ) -> Result<(), RekhaError> {
        if self.collections.contains_key(name) {
            return Err(RekhaError::InvalidArgument(format!(
                "collection '{name}' already exists"
            )));
        }

        let meta = CollectionMeta {
            name: name.to_string(),
            config: config.clone(),
            vector_count: 0,
            index_ready: false,
            created_at_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.store.store_collection_meta(&meta)?;

        let state = self.build_collection_state(name, &config, meta)?;

        let handle = Arc::new(PerCollectionHandle {
            collection_name: name.to_string(),
            collections: self.collections.clone(),
        });

        let node_id = &self.config.cluster.node_id;
        let peers: Vec<String> = self
            .config
            .cluster
            .seed_nodes
            .iter()
            .filter(|s| !s.starts_with(node_id))
            .cloned()
            .collect();

        for shard in 0..config.num_vector_shards {
            let raft_state = rekha_raft::ReplicatedState::new(shard);
            let raft_node = Arc::new(RaftNode::with_store(
                node_id.clone(),
                shard,
                peers.clone(),
                raft_state,
                Some(state.raft_log_store.clone()),
                Some(handle.clone() as Arc<dyn IndexBufferHandle>),
            ));
            if peers.is_empty() {
                let _ = raft_node.start_election().await;
            }
            state.raft_nodes.insert(shard, raft_node);
        }
        info!(
            "Created {} Raft nodes for collection '{}'",
            config.num_vector_shards, name
        );

        self.collections.insert(name.to_string(), state);
        self.spawn_flush_loop(name);
        info!("Collection '{}' created successfully", name);
        Ok(())
    }

    /// Drop a collection and all its data.
    pub async fn drop_collection(&self, name: &str) -> Result<(), RekhaError> {
        let state = self
            .collections
            .remove(name)
            .ok_or_else(|| RekhaError::NotFound(format!("collection '{name}' not found")))?;

        if let Err(e) = state.1.store.delete_all_in_namespace() {
            warn!("Failed to delete all data in collection '{name}': {e}");
        }
        if let Err(e) = self.store.delete_collection_meta(name) {
            warn!("Failed to delete collection meta '{name}': {e}");
        }
        info!("Collection '{}' dropped", name);
        Ok(())
    }

    /// List all collections.
    pub async fn list_collections(&self) -> Vec<CollectionMeta> {
        self.store.list_collections().unwrap_or_else(|e| {
            warn!("Failed to list collections: {e}");
            Vec::new()
        })
    }

    /// Check if a collection exists by name.
    pub async fn collection_exists(&self, name: &str) -> bool {
        self.collections.contains_key(name)
            || self
                .store
                .load_collection_meta(name)
                .ok()
                .flatten()
                .is_some()
    }

    /// Get collection config.
    pub fn collection_config(&self, name: &str) -> Option<CollectionConfig> {
        self.collections.get(name).map(|s| s.config.clone())
    }

    /// Get a collection's namespaced store.
    pub fn collection_store(&self, name: &str) -> Option<Arc<RocksVectorStore>> {
        self.collections.get(name).map(|s| s.store.clone())
    }

    fn namespaced_store(&self, name: &str) -> Arc<RocksVectorStore> {
        let db = self.store.db().clone();
        Arc::new(RocksVectorStore::from_db(db, Some(name.to_string())))
    }

    async fn load_or_create_meta(&self, name: &str) -> CollectionMeta {
        if let Ok(Some(meta)) = self.store.load_collection_meta(name) {
            return meta;
        }
        let dim_val =
            self.config.partition.dim_group_size * self.config.partition.num_dim_groups as usize;
        let mut pq_m = std::cmp::min(self.config.index.pq_num_sub_vectors as u32, dim_val as u32);
        while pq_m > 1 && !(dim_val as u32).is_multiple_of(pq_m) {
            pq_m -= 1;
        }
        let config = CollectionConfig {
            dim: dim_val as u32,
            num_vector_shards: self.config.partition.num_vector_shards,
            replication_factor: self.config.partition.replication_factor as u64,
            num_dim_groups: self.config.partition.num_dim_groups,
            dim_group_size: self.config.partition.dim_group_size as u32,
            graph_degree: self.config.index.graph_degree as u32,
            search_list_size: self.config.index.search_list_size as u32,
            pq_num_sub_vectors: pq_m,
            pq_num_centroids: self.config.index.pq_num_centroids as u32,
            re_rank_k: std::cmp::min(self.config.index.re_rank_k as u32, dim_val as u32),
            distance_metric: DistanceMetric::L2,
        };
        let meta = CollectionMeta {
            name: name.to_string(),
            config,
            vector_count: 0,
            index_ready: false,
            created_at_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let _ = self.store.store_collection_meta(&meta);
        meta
    }

    /// Initialize all collections from stored metadata and recover state.
    pub async fn initialize_all(&self) {
        let collections = self.list_collections().await;
        for meta in &collections {
            if self.collections.contains_key(&meta.name) {
                continue;
            }
            let name = meta.name.clone();
            let state = match self.build_collection_state(&name, &meta.config, meta.clone()) {
                Ok(s) => s,
                Err(e) => {
                    warn!("Failed to create index for collection '{name}': {e}");
                    let store = self.namespaced_store(&name);
                    CollectionState {
                        config: meta.config.clone(),
                        meta: meta.clone(),
                        store,
                        index: Arc::new(RwLock::new(None)),
                        raft_nodes: DashMap::new(),
                        raft_log_store: RaftLogStore::with_namespace(
                            self.store.clone(),
                            name.clone(),
                        ),
                        next_auto_id: AtomicU64::new(1),
                    }
                }
            };

            self.collections.insert(name.clone(), state);
            self.spawn_flush_loop(&name);
            info!("Recovered collection '{}' from storage", name);
        }
        *self.initialized.write().await = true;
        info!(
            "Coordinator initialized with {} collections",
            collections.len()
        );
    }

    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }

    pub async fn peer_address(&self, node_id: &str) -> Option<String> {
        self.peers
            .read()
            .await
            .get(node_id)
            .map(|p| p.info.address.clone())
    }

    pub fn local_node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.config.cluster.node_id.clone(),
            address: self.config.cluster.bind_addr.clone(),
            partition_id: 0,
            dim_groups: (0..self.config.partition.num_dim_groups).collect(),
            is_leader: true,
            raft_term: 1,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        }
    }

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

    pub async fn register_peer(&self, info: NodeInfo) {
        if info.node_id.is_empty() || info.address.is_empty() {
            return;
        }
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

    pub async fn healthy_peers(&self) -> Vec<NodeInfo> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| p.info.status == NodeStatus::Healthy)
            .map(|p| p.info.clone())
            .collect()
    }

    async fn refresh_peer_pool(&self) {
        let healthy = self.healthy_peers().await;
        let mut pool = self.peer_pool.write().await;
        let external: Vec<NodeInfo> = healthy
            .into_iter()
            .filter(|p| p.node_id != self.config.cluster.node_id)
            .collect();
        pool.refresh(&external).await;
    }

    pub async fn peers_for_handshake(&self, exclude: &str) -> Vec<NodeInfo> {
        let peers = self.peers.read().await;
        peers
            .values()
            .filter(|p| p.info.node_id != exclude)
            .map(|p| p.info.clone())
            .collect()
    }

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

    pub fn config_ref(&self) -> &ServerConfig {
        &self.config
    }

    /// Register a Raft node for a specific collection + partition.
    pub fn register_raft_node(
        &self,
        collection_name: &str,
        partition_id: u64,
        node: Arc<RaftNode>,
    ) {
        if let Some(state) = self.collections.get(collection_name) {
            state.raft_nodes.insert(partition_id, node);
        }
    }

    /// Get a Raft node for a specific collection + partition.
    pub fn raft_node(&self, collection_name: &str, partition_id: u64) -> Option<Arc<RaftNode>> {
        // Check per-collection raft nodes.
        if let Some(node) = self
            .collections
            .get(collection_name)
            .and_then(|state| state.raft_nodes.get(&partition_id).map(|n| n.clone()))
        {
            return Some(node);
        }
        // Check system Raft group.
        if collection_name == "__system__" || partition_id == SYSTEM_PARTITION_ID {
            if let Ok(guard) = self.system_raft_node.read() {
                if let Some(ref sys) = *guard {
                    return Some(sys.clone());
                }
            }
        }
        None
    }

    /// Create a RaftLogStore for a given collection.
    pub fn raft_log_store_for(&self, collection_name: &str) -> RaftLogStore {
        RaftLogStore::with_namespace(self.store.clone(), collection_name.into())
    }

    fn spawn_flush_loop(&self, collection_name: &str) {
        let flush_ms = self.config.index.insert_buffer_flush_interval_ms;
        let name = collection_name.to_string();
        let collections = self.collections.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(flush_ms));
            loop {
                interval.tick().await;
                if let Some(state) = collections.get(&name) {
                    let mut idx = state.index.write().await;
                    if let Some(ref mut idx) = *idx {
                        if idx.should_flush() || idx.buffer_len() > 0 {
                            let buf_len = idx.buffer_len();
                            if let Err(e) = idx.flush_buffer() {
                                warn!(
                                    "Buffer flush failed for '{}': {e} (buffer: {buf_len} vectors)",
                                    name
                                );
                            }
                        }
                    }
                }
            }
        });
    }

    /// Search within a specific collection.
    pub async fn search_for_collection(
        &self,
        collection_name: &str,
        query: Vec<f32>,
        k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let (index_arc, store) = {
            let state = self.collections.get(collection_name).ok_or_else(|| {
                RekhaError::NotFound(format!("collection '{collection_name}' not found"))
            })?;
            (state.index.clone(), state.store.clone())
        };
        let index_guard = index_arc.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal {
            detail: format!("index for collection '{collection_name}' not initialized"),
        })?;

        let num_groups = self.config.partition.num_dim_groups;
        let total_dim = query.len();
        let dims_per_group = total_dim / num_groups as usize;

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
                            store
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

        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k * 2);

        let metric = DistanceMetric::L2;
        for candidate in candidates.iter_mut().take(k * 2) {
            let id = candidate.id;
            if let Ok(Some(full_vec)) = store.get_vector(id) {
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

    /// Insert into a specific collection.
    pub async fn insert_into_collection(
        &self,
        collection_name: &str,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Payload>,
    ) -> Result<u64, RekhaError> {
        let state = self.collections.get(collection_name).ok_or_else(|| {
            RekhaError::NotFound(format!("collection '{collection_name}' not found"))
        })?;

        let expected_dim = state.config.dim as usize;
        if vector.len() != expected_dim {
            return Err(RekhaError::InvalidDimension {
                expected: expected_dim,
                actual: vector.len(),
            });
        }

        let id = if id == 0 {
            state.next_auto_id.fetch_add(1, Ordering::SeqCst)
        } else {
            id
        };

        let store = state.store.clone();
        let index = state.index.clone();
        if let Some(raft_node) = state.raft_nodes.get(&0) {
            if raft_node.is_leader().await {
                let cmd = rekha_raft::state::RaftCommand::Insert {
                    id,
                    vector,
                    payload: payload.map(|p| p.data),
                };
                raft_node.propose(cmd).await?;
                return Ok(id);
            }
        }
        drop(state);

        store.put_vector(id, &vector)?;
        if let Some(ref p) = payload {
            store.put_payload(id, &p.data)?;
        }
        // Also buffer into index for immediate searchability
        let mut idx_guard = index.write().await;
        if let Some(ref mut idx) = *idx_guard {
            idx.buffer_insert(id, vector);
        }
        Ok(id)
    }

    /// Delete from a specific collection.
    pub async fn delete_from_collection(
        &self,
        collection_name: &str,
        ids: &[u64],
    ) -> Result<u64, RekhaError> {
        let state = self.collections.get(collection_name).ok_or_else(|| {
            RekhaError::NotFound(format!("collection '{collection_name}' not found"))
        })?;
        state.store.delete(ids)
    }

    /// Fetch from a specific collection.
    pub async fn fetch_from_collection(
        &self,
        collection_name: &str,
        ids: &[u64],
    ) -> Result<Vec<(u64, Option<Vec<f32>>, Option<Vec<u8>>)>, RekhaError> {
        let state = self.collections.get(collection_name).ok_or_else(|| {
            RekhaError::NotFound(format!("collection '{collection_name}' not found"))
        })?;
        let mut results = Vec::new();
        for id in ids {
            let vec = state.store.get_vector(*id)?;
            let payload = state.store.get_payload(*id)?;
            results.push((*id, vec, payload));
        }
        Ok(results)
    }

    pub async fn topology(&self) -> Result<ClusterTopology, RekhaError> {
        let topo = self.topology.read().await;
        Ok(topo.clone())
    }

    pub async fn node_info(&self, _node_id: &str) -> Result<NodeInfo, RekhaError> {
        Ok(self.local_node_info())
    }
}

impl Coordinator {
    /// Returns all Raft nodes across all collections plus the system node.
    /// Returns (collection_name_or_system, partition_id, Arc<RaftNode>).
    pub fn all_raft_nodes(&self) -> Vec<(String, u64, Arc<RaftNode>)> {
        let mut nodes = Vec::new();
        for entry in self.collections.iter() {
            let col_name = entry.key().clone();
            for n in entry.raft_nodes.iter() {
                nodes.push((col_name.clone(), *n.key(), n.value().clone()));
            }
        }
        if let Ok(guard) = self.system_raft_node.read() {
            if let Some(ref sys) = *guard {
                nodes.push(("__system__".into(), SYSTEM_PARTITION_ID, sys.clone()));
            }
        }
        nodes
    }

    /// Register the system Raft node for collection metadata replication.
    pub fn register_system_raft_node(&self, node: Arc<RaftNode>) {
        if let Ok(mut guard) = self.system_raft_node.write() {
            *guard = Some(node);
        }
    }

    /// Get the system Raft node.
    pub fn system_raft_node(&self) -> Option<Arc<RaftNode>> {
        self.system_raft_node.read().ok().and_then(|g| g.clone())
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
    async fn test_create_default_collection() {
        let coord = test_coordinator();
        coord.create_default_collection().await.unwrap();
        assert!(coord.collections.contains_key("default"));
        assert!(coord.collection_exists("default").await);
    }

    #[tokio::test]
    async fn test_create_collection_and_exists() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "test_col",
                CollectionConfig {
                    dim: 64,
                    num_vector_shards: 1,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(coord.collection_exists("test_col").await);
        assert!(!coord.collection_exists("nonexistent").await);
    }

    #[tokio::test]
    async fn test_create_duplicate_collection() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "dup",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let result = coord
            .create_collection(
                "dup",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_drop_collection() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "to_drop",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(coord.collection_exists("to_drop").await);
        coord.drop_collection("to_drop").await.unwrap();
        assert!(!coord.collection_exists("to_drop").await);
    }

    #[tokio::test]
    async fn test_list_collections() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "list_a",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        coord
            .create_collection(
                "list_b",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let list = coord.list_collections().await;
        assert!(list.iter().any(|m| m.name == "list_a"));
        assert!(list.iter().any(|m| m.name == "list_b"));
    }

    #[tokio::test]
    async fn test_collection_config() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "cfg_col",
                CollectionConfig {
                    dim: 128,
                    num_vector_shards: 3,
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let cfg = coord.collection_config("cfg_col").unwrap();
        assert_eq!(cfg.dim, 128);
        assert_eq!(cfg.num_vector_shards, 3);
        assert_eq!(cfg.distance_metric, DistanceMetric::Cosine);
    }

    #[tokio::test]
    async fn test_insert_into_collection() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "ins_col",
                CollectionConfig {
                    dim: 8,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let id = coord
            .insert_into_collection("ins_col", 42, vec![0.1; 8], None)
            .await
            .unwrap();
        assert_eq!(id, 42);
        let store = coord.collection_store("ins_col").unwrap();
        let v = store.get_vector(42).unwrap().unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_insert_auto_id() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "auto_col",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let id1 = coord
            .insert_into_collection("auto_col", 0, vec![0.1; 64], None)
            .await
            .unwrap();
        let id2 = coord
            .insert_into_collection("auto_col", 0, vec![0.2; 64], None)
            .await
            .unwrap();
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[tokio::test]
    async fn test_insert_with_payload() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "pay_col",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let payload = Payload::from_text("test data");
        coord
            .insert_into_collection("pay_col", 7, vec![0.5; 64], Some(payload))
            .await
            .unwrap();
        let store = coord.collection_store("pay_col").unwrap();
        let stored = store.get_payload(7).unwrap().unwrap();
        assert_eq!(stored, b"test data");
    }

    #[tokio::test]
    async fn test_search_across_collections() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "srch_a",
                CollectionConfig {
                    dim: 64,
                    num_vector_shards: 1,
                    num_dim_groups: 8,
                    dim_group_size: 8,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let store_a = coord.collection_store("srch_a").unwrap();
        let mut index_a =
            RekhaIndex::new(64, 4, 16, 4, (*store_a).clone(), DistanceMetric::L2).unwrap();
        for i in 0..5 {
            let v: Vec<f32> = (0..64).map(|d| (i * 10 + d) as f32).collect();
            index_a.add_vector_for_test(i, v);
            store_a
                .put_vector(i, &(0..64).map(|d| (i * 10 + d) as f32).collect::<Vec<_>>())
                .unwrap();
        }
        index_a.build().unwrap();
        if let Some(state) = coord.collections.get("srch_a") {
            *state.index.write().await = Some(index_a);
        }

        let (results, _stats) = coord
            .search_for_collection("srch_a", vec![0.0; 64], 3, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_search_missing_collection() {
        let coord = test_coordinator();
        let result = coord
            .search_for_collection("nonexistent", vec![0.0; 8], 3, SearchParams::default())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_from_collection() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "del_col",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        coord
            .insert_into_collection("del_col", 1, vec![1.0; 64], None)
            .await
            .unwrap();
        let count = coord.delete_from_collection("del_col", &[1]).await.unwrap();
        assert_eq!(count, 1);
        let store = coord.collection_store("del_col").unwrap();
        assert!(store.get_vector(1).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_fetch_from_collection() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "fetch_col",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        coord
            .insert_into_collection(
                "fetch_col",
                10,
                vec![1.0; 64],
                Some(Payload::from_bytes(b"data".to_vec())),
            )
            .await
            .unwrap();
        let results = coord
            .fetch_from_collection("fetch_col", &[10])
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        let (id, _vec, payload) = &results[0];
        assert_eq!(*id, 10);
        assert_eq!(payload.as_deref(), Some(&b"data"[..]));
    }

    #[tokio::test]
    async fn test_initialize_all_recovery() {
        let coord = test_coordinator();
        coord
            .create_collection(
                "recover_col",
                CollectionConfig {
                    dim: 64,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 16,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(coord.collections.contains_key("recover_col"));

        // Create a new coordinator from the same store to test recovery
        let coord2 = Coordinator::new(
            coord.config.clone(),
            coord.store.clone(),
            coord.partition_manager.clone(),
        );
        coord2.initialize_all().await;
        assert!(coord2.collections.contains_key("recover_col"));
        assert!(coord2.is_initialized().await);
    }

    #[tokio::test]
    async fn test_local_node_info() {
        let coord = test_coordinator();
        let info = coord.local_node_info();
        assert_eq!(info.node_id, "test-node");
        assert!(info.is_leader);
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
    }

    #[tokio::test]
    async fn test_check_peer_health() {
        let coord = test_coordinator();
        coord.check_peer_health().await;
        let healthy = coord.healthy_peers().await;
        assert!(healthy.is_empty());
    }

    #[tokio::test]
    async fn test_accessors() {
        let coord = test_coordinator();
        assert_eq!(coord.cluster_id(), "rekha-dev");
        assert_eq!(coord.node_id(), "test-node");
        assert_eq!(coord.bind_addr(), "0.0.0.0:50051");
    }

    #[tokio::test]
    async fn test_drop_nonexistent_collection() {
        let coord = test_coordinator();
        let result = coord.drop_collection("nope").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_raft_log_store_for() {
        let coord = test_coordinator();
        let log_store = coord.raft_log_store_for("my_col");
        let entry = rekha_raft::node::RaftLogEntry {
            term: 1,
            index: 1,
            command: rekha_raft::state::RaftCommand::NoOp,
        };
        log_store.store_entry(0, &entry).unwrap();
        let entries = log_store.load_entries(0, 1).unwrap();
        assert_eq!(entries.len(), 1);
    }
}
