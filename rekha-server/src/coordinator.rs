use rekha_core::{
    ClusterTopology, CollectionConfig, CollectionMeta, ConsistencyLevel, DistanceMetric,
    NodeInfo, NodeStatus, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats,
    VectorRecord, VectorStoreBackend, now_micros, quorum,
};
use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_storage::RocksVectorStore;

use rekha_client::RekhaClient as PeerRekhaClient;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::info;

use crate::config::ServerConfig;

const PEER_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub(crate) struct PeerState {
    pub info: NodeInfo,
    pub last_seen: Instant,
}

struct PeerClient {
    #[allow(dead_code)]
    info: NodeInfo,
    client: PeerRekhaClient,
    last_used: Instant,
    error_count: u64,
}

impl PeerClient {
    async fn connect(info: &NodeInfo) -> Result<Self, RekhaError> {
        let seeds = vec![info.address.clone()];
        let client = PeerRekhaClient::connect(&seeds).await?;
        Ok(Self {
            info: info.clone(),
            client,
            last_used: Instant::now(),
            error_count: 0,
        })
    }

    async fn try_search(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        collection: &str,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        self.last_used = Instant::now();
        self.client
            .search_with_params(query.to_vec(), collection, k, params.clone(), ConsistencyLevel::One)
            .await
    }

    async fn try_remote_insert(
        &mut self, collection: &str, id: u64, vector: &[f32], payload: &Option<Vec<u8>>, timestamp: u64,
    ) -> Result<(), RekhaError> {
        self.last_used = Instant::now();
        self.client
            .replica_insert(id, vector.to_vec(), collection, payload.clone(), timestamp)
            .await?;
        Ok(())
    }

    async fn try_remote_create_collection(
        &mut self, name: &str, config: &crate::proto::CollectionConfig, timestamp: u64,
    ) -> Result<bool, RekhaError> {
        let client_cfg = rekha_client::proto::CollectionConfig {
            dim: config.dim,
            num_vector_shards: config.num_vector_shards,
            replication_factor: config.replication_factor,
            num_dim_groups: config.num_dim_groups,
            dim_group_size: config.dim_group_size,
            nlist: config.nlist,
            nprobe: config.nprobe,
            pq_num_sub_vectors: config.pq_num_sub_vectors,
            pq_num_centroids: config.pq_num_centroids,
            re_rank_k: config.re_rank_k,
        };
        self.last_used = Instant::now();
        self.client
            .replica_create_collection(name, client_cfg, timestamp)
            .await
    }

    async fn try_remote_drop_collection(&mut self, name: &str, timestamp: u64) -> Result<bool, RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_drop_collection(name, timestamp).await
    }

    async fn try_remote_delete(
        &mut self, collection: &str, ids: &[u64], timestamp: u64,
    ) -> Result<(), RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_delete(ids, collection, timestamp).await?;
        Ok(())
    }
}

pub(crate) struct PeerPool {
    clients: HashMap<String, PeerClient>,
}

impl PeerPool {
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
        }
    }

    pub async fn refresh(&mut self, peers: &[NodeInfo]) {
        let active: std::collections::HashSet<String> =
            peers.iter().map(|p| p.node_id.clone()).collect();
        self.clients.retain(|node_id, _| active.contains(node_id));

        for info in peers {
            if !self.clients.contains_key(&info.node_id) {
                match PeerClient::connect(info).await {
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

    pub async fn search_fan_out(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
        collection: &str,
    ) -> (Vec<ScoredPoint>, SearchStats) {
        let mut peer_params = params.clone();
        peer_params.local_only = true;
        let mut all_candidates: Vec<ScoredPoint> = Vec::new();
        let mut stats = SearchStats::default();

        let node_ids: Vec<String> = self.clients.keys().cloned().collect();
        for node_id in &node_ids {
            if let Some(client) = self.clients.get_mut(node_id) {
                match client.try_search(query, k, &peer_params, collection).await {
                    Ok((candidates, _peer_stats)) => {
                        all_candidates.extend(candidates);
                        client.error_count = 0;
                    }
                    Err(_) => {
                        if let Some(c) = self.clients.get_mut(node_id) {
                            c.error_count += 1;
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
        stats.nodes_contacted = node_ids.len() as u32;
        (all_candidates, stats)
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

pub struct Coordinator {
    config: ServerConfig,
    index: Arc<RwLock<Option<RekhaIndex>>>,
    store: Arc<RocksVectorStore>,
    partition_manager: Arc<RwLock<PartitionManager>>,
    topology: Arc<RwLock<ClusterTopology>>,
    initialized: Arc<RwLock<bool>>,
    peers: Arc<RwLock<HashMap<String, PeerState>>>,
    peer_pool: Arc<RwLock<PeerPool>>,
    next_auto_id: AtomicU64,
}

impl Coordinator {
    pub fn new(
            config: ServerConfig,
            store: Arc<RocksVectorStore>,
            partition_manager: Arc<RwLock<PartitionManager>>,
        ) -> Self {
            let starting_id = Self::starting_auto_id(&store);
            Self {
                config: config.clone(),
                index: Arc::new(RwLock::new(None)),
                store,
                partition_manager,
                topology: Arc::new(RwLock::new(ClusterTopology {
                    cluster_id: String::new(),
                    nodes: HashMap::new(),
                    partition_map: HashMap::new(),
                })),
                initialized: Arc::new(RwLock::new(false)),
                peers: Arc::new(RwLock::new(HashMap::new())),
                peer_pool: Arc::new(RwLock::new(PeerPool::new())),
                next_auto_id: AtomicU64::new(starting_id),
            }
    }

    fn starting_auto_id(store: &RocksVectorStore) -> u64 {
        match store.iter_ids() {
            Ok(ids) => ids.iter().max().copied().unwrap_or(0) + 1,
            Err(_) => 1,
        }
    }

    pub async fn initialize(&self, index: RekhaIndex) {
        {
            let mut idx = self.index.write().await;
            *idx = Some(index);
        }
        *self.initialized.write().await = true;

        let default_key = "collection:default".to_string();
        if self.store.get_metadata(&default_key).unwrap_or(None).is_none() {
            let cfg = CollectionConfig {
                dim: 8, num_vector_shards: 6, replication_factor: 1,
                num_dim_groups: 4, dim_group_size: 2,
                nlist: 128, nprobe: 16,
                pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
            };
            let meta = CollectionMeta {
                config: cfg,
                timestamp: now_micros(),
                is_deleted: false, vector_count: 0,
            };
            if let Ok(json) = serde_json::to_vec(&meta) {
                let _ = self.store.put_metadata(&default_key, &json);
            }
        }

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection("default", 8, 128, 16);
        }
        drop(idx);

        self.spawn_flush_loop();
        self.spawn_maintenance_loops();
        info!("Coordinator initialized");
    }

    fn spawn_flush_loop(&self) {
        let index = self.index.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(1000));
            loop {
                interval.tick().await;
                let idx = index.read().await;
                if let Some(ref index) = *idx {
                    for name in index.collection_names() {
                        if index.should_flush(&name).unwrap_or(false) {
                            if let Err(e) = index.flush_buffer(&name) {
                                tracing::warn!("Flush failed for '{name}': {e}");
                            }
                        }
                    }
                }
            }
        });
    }

    fn spawn_maintenance_loops(&self) {
        let store = self.store.clone();
        let gc_grace = self.config.storage.gc_grace_seconds;
        let hint_ttl = self.config.cluster.max_hint_window_secs;

        // Tombstone GC loop — runs every hour
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600));
            loop {
                interval.tick().await;
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if let Ok(tombstones) = store.scan_tombstones() {
                    let expired: Vec<u64> = tombstones
                        .into_iter()
                        .filter(|(_, ts)| {
                            let tomb_secs = ts / 1_000_000;
                            now_secs.saturating_sub(tomb_secs) >= gc_grace
                        })
                        .map(|(id, _)| id)
                        .collect();
                    if !expired.is_empty() {
                        let count = expired.len();
                        let _ = store.physically_delete_vectors(&expired);
                        info!("GC collected {count} expired tombstones");
                    }
                }
            }
        });

        // Hint expiry loop — runs every 30 minutes
        let store2 = self.store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1800));
            loop {
                interval.tick().await;
                let deleted = store2.delete_expired_hints(hint_ttl).unwrap_or(0);
                if deleted > 0 {
                    info!("Expired {deleted} stale hints");
                }
            }
        });
    }

    /// Resolve proto consistency level to core type, falling back to config default.
    pub(crate) fn resolve_consistency(&self, proto_cl: i32) -> ConsistencyLevel {
        match proto_cl {
            1 => ConsistencyLevel::One,
            2 => ConsistencyLevel::Quorum,
            3 => ConsistencyLevel::All,
            _ => crate::config::parse_consistency(&self.config.cluster.default_write_consistency)
                .unwrap_or(ConsistencyLevel::Quorum),
        }
    }

    fn now_epoch_secs() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    fn maybe_load_timestamp(&self, collection: &str, id: u64) -> u64 {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        ns.get_vector_record(id).ok().flatten().map(|r| r.timestamp).unwrap_or(0)
    }

    pub async fn search(
        &self, collection: &str, query: Vec<f32>, k: usize,
        params: SearchParams, consistency: ConsistencyLevel,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let index_guard = self.index.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal {
            detail: "index not initialized".into(),
        })?;

        if !index.has_collection(collection) {
            return Err(RekhaError::NotFound(collection.into()));
        }

        let mut candidates: Vec<ScoredPoint> = Vec::new();
        let (ids, dists) = index.search(collection, &query, k * 2, &params).map_err(|e| {
            stats.warnings.push(format!("search failed: {e}"));
            e
        })?;
        for (i, id) in ids.iter().enumerate() {
            let score = dists.get(i).copied().unwrap_or(f32::MAX);
            let payload = self.maybe_load_payload(collection, params.include_payloads, *id);
            let timestamp = self.maybe_load_timestamp(collection, *id);
            candidates.push(ScoredPoint { id: *id, score, payload, timestamp });
        }

        if !params.local_only {
            let peer_count = { self.peer_pool.read().await.clients.len() };
            if peer_count > 0 {
                let rf = self.read_collection_config(collection)
                    .map(|c| c.replication_factor as usize)
                    .unwrap_or(1);
                let needed_per_shard = match consistency {
                    ConsistencyLevel::One => 1,
                    ConsistencyLevel::Quorum => quorum(rf),
                    ConsistencyLevel::All => rf,
                };

                // For each shard, pick needed_per_shard replicas from the consistent hash ring.
                // This ensures quorum coverage for every shard in the collection.
                let mut node_set: std::collections::HashSet<String> = std::collections::HashSet::new();
                if let Some(cfg) = self.read_collection_config(collection) {
                    let pm = self.partition_manager.read().await;
                    for shard in 0..cfg.num_vector_shards {
                        let replicas = pm.replicas_for(shard, rf);
                        for replica in replicas.iter().take(needed_per_shard) {
                            if replica.node_id != self.node_id() {
                                node_set.insert(replica.node_id.clone());
                            }
                        }
                    }
                }
                let node_ids: Vec<String> = node_set.into_iter().collect();

                let mut pool = self.peer_pool.write().await;

                if !node_ids.is_empty() {
                    let mut peer_params = params.clone();
                    peer_params.local_only = true;
                    let mut all_peer_candidates: Vec<ScoredPoint> = Vec::new();
                    let mut peer_stats = SearchStats::default();
                    for node_id in &node_ids {
                        if let Some(client) = pool.clients.get_mut(node_id) {
                            match client.try_search(&query, k, &peer_params, collection).await {
                                Ok((candidates, _)) => {
                                    all_peer_candidates.extend(candidates);
                                    client.error_count = 0;
                                }
                                Err(_) => {
                                    if let Some(c) = pool.clients.get_mut(node_id) {
                                        c.error_count += 1;
                                        if c.error_count >= 3 {
                                            pool.clients.remove(node_id);
                                        }
                                    }
                                    peer_stats.warnings.push(format!("peer {node_id} search failed"));
                                }
                            }
                        }
                    }
                    all_peer_candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                    all_peer_candidates.truncate(k * 2);
                    peer_stats.nodes_contacted = node_ids.len() as u32;
                    stats.nodes_contacted = 1 + peer_stats.nodes_contacted;
                    stats.warnings.extend(peer_stats.warnings);
                    candidates.extend(all_peer_candidates);
                } else {
                    stats.nodes_contacted = 1;
                }
            } else {
                stats.nodes_contacted = 1;
            }
        } else {
            stats.nodes_contacted = 1;
        }

        // Dedup by id, keeping highest timestamp version
        let mut seen: HashMap<u64, ScoredPoint> = HashMap::new();
        for c in candidates {
            let id = c.id;
            match seen.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if c.timestamp > entry.get().timestamp {
                        entry.insert(c);
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(c);
                }
            }
        }
        let mut candidates: Vec<ScoredPoint> = seen.into_values().collect();

        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k * 2);

        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        for candidate in candidates.iter_mut().take(k * 2) {
            if let Ok(Some(full_vec)) = ns.get_vector(candidate.id) {
                candidate.score = DistanceMetric::L2.distance(&full_vec, &query);
            }
        }
        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k);

        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.vectors_scanned = candidates.len() as u64;
        Ok((candidates, stats))
    }

    fn maybe_load_payload(&self, collection: &str, include: bool, id: u64) -> Option<Payload> {
        if include {
            let ns = self.store.as_ref().clone().with_namespace(collection.into());
            ns.get_payload(id).ok().flatten().map(Payload::from_bytes)
        } else { None }
    }

    pub async fn replicate_collection(
        &self, name: &str, proto_cfg: &crate::proto::CollectionConfig, timestamp: u64,
    ) -> Result<bool, RekhaError> {
        let cfg = CollectionConfig {
            dim: proto_cfg.dim,
            num_vector_shards: proto_cfg.num_vector_shards,
            replication_factor: proto_cfg.replication_factor,
            num_dim_groups: proto_cfg.num_dim_groups,
            dim_group_size: proto_cfg.dim_group_size,
            nlist: proto_cfg.nlist,
            nprobe: proto_cfg.nprobe,
            pq_num_sub_vectors: proto_cfg.pq_num_sub_vectors,
            pq_num_centroids: proto_cfg.pq_num_centroids,
            re_rank_k: proto_cfg.re_rank_k,
        };
        let key = format!("collection:{name}");

        // LWW: skip if existing metadata has a higher timestamp
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            if let Ok(existing) = serde_json::from_slice::<CollectionMeta>(&data) {
                if existing.timestamp > timestamp {
                    return Ok(false);
                }
            }
        }

        let meta = CollectionMeta {
            config: cfg.clone(),
            timestamp,
            is_deleted: false, vector_count: 0,
        };
        let json = serde_json::to_vec(&meta).map_err(|e| {
            RekhaError::InvalidArgument(format!("serialize config: {e}"))
        })?;
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection(name, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
        }
        Ok(true)
    }

    pub async fn replicate_drop_collection(&self, name: &str, timestamp: u64) -> Result<bool, RekhaError> {
        let key = format!("collection:{name}");

        // LWW: read existing metadata
        let existing = match self.store.get_metadata(&key)? {
            Some(data) => {
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                    meta
                } else {
                    // Old format (bare CollectionConfig) — treat as existing, not deleted
                    CollectionMeta {
                        config: serde_json::from_slice(&data).unwrap_or_default(),
                        timestamp: 0,
                        is_deleted: false, vector_count: 0,
                    }
                }
            }
            None => return Ok(false),
        };

        if existing.timestamp > timestamp && !existing.is_deleted {
            return Ok(false);
        }

        let meta = CollectionMeta {
            config: existing.config,
            timestamp,
                is_deleted: true, vector_count: existing.vector_count,
        };
        let json = serde_json::to_vec(&meta).map_err(|e| {
            RekhaError::InvalidArgument(format!("serialize config: {e}"))
        })?;
        self.store.put_metadata(&key, &json)?;

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.drop_collection(name);
        }
        Ok(true)
    }

    pub async fn drop_collection(
        &self, name: &str, timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let timestamp = if timestamp == 0 { now_micros() } else { timestamp };
        let key = format!("collection:{name}");

        // LWW: read existing metadata
        let existing = match self.store.get_metadata(&key)? {
            Some(data) => {
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                    meta
                } else {
                    // Old format (bare CollectionConfig)
                    CollectionMeta {
                        config: serde_json::from_slice(&data).unwrap_or_default(),
                        timestamp: 0,
                        is_deleted: false, vector_count: 0,
                    }
                }
            }
            None => return Ok(false),
        };

        if existing.timestamp >= timestamp {
            return Ok(false);
        }

        let drop_rf = existing.config.replication_factor as usize;

        // Write tombstone locally
        let meta = CollectionMeta {
            config: existing.config,
            timestamp,
                is_deleted: true, vector_count: existing.vector_count,
        };
        let json = serde_json::to_vec(&meta).map_err(|e| {
            RekhaError::InvalidArgument(format!("serialize config: {e}"))
        })?;
        self.store.put_metadata(&key, &json)?;

        // Drop in-memory index
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.drop_collection(name);
        }
        drop(idx);

        // Broadcast to peers with CL based on collection RF
        let required = match consistency {
            ConsistencyLevel::One => 1usize,
            ConsistencyLevel::Quorum => quorum(drop_rf),
            ConsistencyLevel::All => drop_rf,
        };

        let peer_ids: Vec<String> = {
            let pool = self.peer_pool.read().await;
            pool.clients.keys().cloned().collect()
        };

        let hint_store = self.store.as_ref().clone().with_namespace(String::new());
        let mut acks = 1u64;
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                match client.try_remote_drop_collection(name, timestamp).await {
                    Ok(true) => acks += 1,
                    Ok(false) => {} // LWW skip, not an error
                    Err(_) => {
                        if self.config.cluster.hinted_handoff_enabled {
                            let _ = self.store.put_collection_hint(node_id, name, &[], timestamp, 1);
                        }
                    }
                }
            }
        }

        if (acks as usize) >= required {
            Ok(true)
        } else {
            Err(RekhaError::Unavailable {
                detail: format!("consistency level not met for collection drop: got {acks}/{required} acknowledgments"),
            })
        }
    }

    pub async fn create_collection(
        &self, name: &str, dim: u32, nlist: u32, nprobe: u32, rf: u64,
        timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let timestamp = if timestamp == 0 { now_micros() } else { timestamp };
        let key = format!("collection:{name}");

        // LWW: skip if existing metadata is not deleted and has >= timestamp
        if let Some(data) = self.store.get_metadata(&key)? {
            if let Ok(existing) = serde_json::from_slice::<CollectionMeta>(&data) {
                if !existing.is_deleted && existing.timestamp >= timestamp {
                    return Ok(false);
                }
            } else {
                // Old format (bare CollectionConfig) — treat as existing
                return Ok(false);
            }
        }

        let cfg = CollectionConfig {
            dim, nlist, nprobe,
            num_vector_shards: 6,
            replication_factor: rf,
            num_dim_groups: 4,
            dim_group_size: dim / 4,
            pq_num_sub_vectors: 4,
            pq_num_centroids: 256,
            re_rank_k: 256,
        };

        // Write metadata locally
        let meta = CollectionMeta {
            config: cfg.clone(),
            timestamp,
            is_deleted: false, vector_count: 0,
        };
        let json = serde_json::to_vec(&meta).map_err(|e| {
            RekhaError::InvalidArgument(format!("serialize config: {e}"))
        })?;
        self.store.put_metadata(&key, &json)?;

        // Create in-memory index
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection(name, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
        }
        drop(idx);

        // Broadcast to peers with CL
        let proto_cfg = crate::proto::CollectionConfig {
            dim, nlist, nprobe,
            num_vector_shards: 6,
            replication_factor: rf,
            num_dim_groups: 4,
            dim_group_size: dim / 4,
            pq_num_sub_vectors: 4,
            pq_num_centroids: 256,
            re_rank_k: 256,
        };

        let peer_ids: Vec<String> = {
            let pool = self.peer_pool.read().await;
            pool.clients.keys().cloned().collect()
        };

        let rf_val = rf as usize;
        let required = match consistency {
            ConsistencyLevel::One => 1usize,
            ConsistencyLevel::Quorum => quorum(rf_val),
            ConsistencyLevel::All => rf_val,
        };

        let hint_store = self.store.as_ref().clone().with_namespace(String::new());
        let mut acks = 1u64;
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                match client.try_remote_create_collection(name, &proto_cfg, timestamp).await {
                    Ok(true) => acks += 1,
                    Ok(false) => {}
                    Err(_) => {
                        if self.config.cluster.hinted_handoff_enabled {
                            let _ = self.store.put_collection_hint(node_id, name, &[], timestamp, 0);
                        }
                    }
                }
            }
        }

        if (acks as usize) >= required {
            Ok(true)
        } else {
            Err(RekhaError::Unavailable {
                detail: format!("consistency level not met for collection creation: got {acks}/{required} acknowledgments"),
            })
        }
    }

    /// Replica insert with LWW semantics.
    /// Skips the write if the existing record has a higher timestamp.
    pub async fn replica_insert(
        &self, collection: &str, id: u64, vector: &[f32], payload: &Option<Payload>, timestamp: u64,
    ) -> Result<u64, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());

        // LWW: check existing timestamp
        if let Ok(Some(record)) = ns.get_vector_record(id) {
            if record.timestamp > timestamp {
                return Ok(id);
            }
        }

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            if let Err(e) = index.insert(collection, id, timestamp, vector) {
                if matches!(&e, RekhaError::NotFound(_)) {
                    if let Some(cfg) = self.read_collection_config(collection) {
                        let _ = index.create_collection(collection, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
                        let _ = index.insert(collection, id, timestamp, vector);
                    }
                }
            }
        }
        drop(idx);

        let is_new = match ns.get_vector_record(id) {
            Ok(Some(ref rec)) => rec.is_tombstone,
            Ok(None) => true,
            Err(_) => true,
        };

        ns.put_vector(id, vector, timestamp)?;
        if let Some(ref p) = payload {
            ns.put_payload(id, &p.data)?;
        }

        if is_new {
            self.update_vector_count(collection, 1i64);
        }

        Ok(id)
    }

    fn read_collection_config(&self, name: &str) -> Option<CollectionConfig> {
        let key = format!("collection:{name}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            // Try new CollectionMeta format first
            if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                if !meta.is_deleted {
                    return Some(meta.config);
                }
                return None;
            }
            // Fall back to old bare CollectionConfig format
            serde_json::from_slice(&data).ok()
        } else {
            None
        }
    }

    fn update_vector_count(&self, collection: &str, delta: i64) {
        let key = format!("collection:{collection}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            if let Ok(mut meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                if delta > 0 {
                    meta.vector_count = meta.vector_count.saturating_add(delta as u64);
                } else {
                    meta.vector_count = meta.vector_count.saturating_sub((-delta) as u64);
                }
                if let Ok(json) = serde_json::to_vec(&meta) {
                    let _ = self.store.put_metadata(&key, &json);
                }
            }
        }
    }

    /// Insert with consistency level, timestamp management, and hinted handoff.
    pub async fn insert(
        &self, collection: &str, id: u64, vector: Vec<f32>, payload: Option<Payload>,
        timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let timestamp = if timestamp == 0 { now_micros() } else { timestamp };
        let id = if id == 0 { self.next_auto_id.fetch_add(1, Ordering::SeqCst) } else { id };

        // 1. Write locally (counts as 1 ack)
        self.replica_insert(collection, id, &vector, &payload, timestamp).await?;

        // 2. Determine replicas
        let mut acks = 1u64;
        let mut hints = Vec::new();
        let pdata = payload.as_ref().map(|p| p.data.clone());

        if let Some(cfg) = self.read_collection_config(collection) {
            let rf = cfg.replication_factor as usize;
            let shard = id % cfg.num_vector_shards;
            let pm = self.partition_manager.read().await;
            let replicas = pm.replicas_for(shard, rf);

            for replica in replicas {
                if replica.node_id == self.node_id() { continue; }
                let mut pool = self.peer_pool.write().await;
                if let Some(client) = pool.clients.get_mut(&replica.node_id) {
                    match client.try_remote_insert(collection, id, &vector, &pdata, timestamp).await {
                        Ok(_) => acks += 1,
                        Err(_) => {
                            if self.config.cluster.hinted_handoff_enabled {
                                hints.push((
                                    replica.node_id.clone(), collection.to_string(),
                                    id, vector.clone(), pdata.clone(), timestamp,
                                ));
                            }
                        }
                    }
                } else if self.config.cluster.hinted_handoff_enabled {
                    hints.push((
                        replica.node_id.clone(), collection.to_string(),
                        id, vector.clone(), pdata.clone(), timestamp,
                    ));
                }
            }
        }

        // 3. Store hints for down nodes
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        for (node_id, col, hid, hvec, hpayload, hts) in &hints {
            let _ = ns.put_hint(node_id, col, *hid, hvec, hpayload.as_deref(), *hts);
        }

        // 4. Check consistency level
        let rf = self.read_collection_config(collection)
            .map(|c| c.replication_factor as usize)
            .unwrap_or(1);
        let required = match consistency {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => quorum(rf),
            ConsistencyLevel::All => rf,
        };

        if acks >= required as u64 {
            Ok(id)
        } else {
            Err(RekhaError::Unavailable {
                detail: format!("consistency level not met: got {acks}/{required} acknowledgments"),
            })
        }
    }

    /// Replica delete with LWW tombstone semantics.
    pub async fn replica_delete(
        &self, collection: &str, ids: &[u64], timestamp: u64,
    ) -> Result<u64, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        let mut removed = 0u64;
        for id in ids {
            let was_live = matches!(ns.get_vector_record(*id), Ok(Some(ref r)) if !r.is_tombstone);
            ns.put_tombstone(*id, timestamp)?;
            if was_live { removed += 1; }
        }
        if removed > 0 {
            self.update_vector_count(collection, -(removed as i64));
        }
        Ok(ids.len() as u64)
    }

    /// Delete vectors with fan-out replication, hinted handoff, and consistency checks.
    pub async fn delete(
        &self, collection: &str, ids: &[u64],
        timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let timestamp = if timestamp == 0 { now_micros() } else { timestamp };

        // 1. Write tombstones locally
        self.replica_delete(collection, ids, timestamp).await?;

        // 2. Fan out to replicas
        let mut acks = 1u64;
        let mut hints: Vec<(String, u64, u64)> = Vec::new(); // (node_id, id, timestamp)

        if let Some(cfg) = self.read_collection_config(collection) {
            for id in ids {
                let shard = id % cfg.num_vector_shards;
                let pm = self.partition_manager.read().await;
                let replicas = pm.replicas_for(shard, cfg.replication_factor as usize);
                for replica in replicas {
                    if replica.node_id == self.node_id() { continue; }
                    let mut pool = self.peer_pool.write().await;
                    if let Some(client) = pool.clients.get_mut(&replica.node_id) {
                        match client.try_remote_delete(collection, &[*id], timestamp).await {
                            Ok(_) => acks += 1,
                            Err(_) => {
                                if self.config.cluster.hinted_handoff_enabled {
                                    hints.push((replica.node_id.clone(), *id, timestamp));
                                }
                            }
                        }
                    } else if self.config.cluster.hinted_handoff_enabled {
                        hints.push((replica.node_id.clone(), *id, timestamp));
                    }
                }
            }
        }

        // 3. Check CL
        let rf = self.read_collection_config(collection)
            .map(|c| c.replication_factor as usize)
            .unwrap_or(1);
        let required = match consistency {
            ConsistencyLevel::One => 1,
            ConsistencyLevel::Quorum => quorum(rf),
            ConsistencyLevel::All => rf,
        };

        if acks < required as u64 {
            return Err(RekhaError::Unavailable {
                detail: format!("consistency level not met for delete: got {acks}/{required}"),
            });
        }

        Ok(ids.len() as u64)
    }

    /// Fetch vectors by id, filtering out tombstones.
    pub async fn fetch(
        &self, collection: &str, ids: &[u64],
        _consistency: ConsistencyLevel,
    ) -> Result<Vec<VectorRecord>, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        let mut results = Vec::new();

        for id in ids {
            if let Ok(Some(record)) = ns.get_vector_record(*id) {
                if !record.is_tombstone {
                    results.push(record);
                }
            }
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
            dim_groups: (0..4).collect(),
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
        let mut peers = self.peers.write().await;
        let mut info = info;
        info.last_heartbeat = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let node_id = info.node_id.clone();
        let node_info = info.clone();
        peers.insert(
            node_id,
            PeerState {
                info,
                last_seen: Instant::now(),
            },
        );
        drop(peers);
        self.partition_manager.write().await.register_node(node_info);
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
        let mut recovered_nodes = Vec::new();
        for peer in peers.values_mut() {
            if peer.last_seen.elapsed() > PEER_TIMEOUT
                && peer.info.status == NodeStatus::Healthy
            {
                peer.info.status = NodeStatus::Unreachable;
                changed = true;
            } else if peer.last_seen.elapsed() <= PEER_TIMEOUT
                && peer.info.status == NodeStatus::Unreachable
            {
                peer.info.status = NodeStatus::Healthy;
                changed = true;
                recovered_nodes.push(peer.info.node_id.clone());
            }
        }
        drop(peers);

        if changed {
            self.sync_topology().await;
        }

        // Deliver hints for recovered nodes
        if !recovered_nodes.is_empty() && self.config.cluster.hinted_handoff_enabled {
            for peer_id in &recovered_nodes {
                let ns = self.store.as_ref().clone().with_namespace(String::new());
                if let Ok(hints) = ns.iter_hints_for_node(peer_id) {
                    let max_age = Self::now_epoch_secs().saturating_sub(self.config.cluster.max_hint_window_secs);
                    for hint in hints {
                        if hint.timestamp / 1_000_000 < max_age {
                            let _ = ns.delete_hint(&hint.target_node_id, &hint.collection, hint.id);
                            continue;
                        }
                        let mut pool = self.peer_pool.write().await;
                        if let Some(client) = pool.clients.get_mut(peer_id) {
                            if client
                                .try_remote_insert(&hint.collection, hint.id, &hint.vector, &hint.payload, hint.timestamp)
                                .await
                                .is_ok()
                            {
                                let _ = ns.delete_hint(&hint.target_node_id, &hint.collection, hint.id);
                            }
                        }
                    }
                }
            }
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

    pub fn config_mut(&self) -> &ServerConfig {
        &self.config
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
            HashMap::new(), 4, 768,
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
        coord.initialize(rekha_index::RekhaIndex::new().unwrap()).await;
        assert!(coord.is_initialized().await);
    }

    #[tokio::test]
    async fn test_coordinator_search_before_init() {
        let coord = test_coordinator();
        let result = coord.search("default", vec![0.0; 8], 5, SearchParams::default(), ConsistencyLevel::One).await;
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

    fn temp_store_owned() -> RocksVectorStore {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("rekha_coord_store_{}", n));
        let _ = std::fs::remove_dir_all(&dir);
        rekha_storage::RocksVectorStore::open(&dir).unwrap()
    }

    #[tokio::test]
    async fn test_coordinator_insert() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.insert("default", 42, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], None, 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let v = ns.get_vector(42).unwrap().unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_insert_with_payload() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        let payload = Payload::from_text("test data");
        let vec8 = vec![0.5; 8];
        coord.insert("default", 7, vec8, Some(payload), 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let stored_payload = ns.get_payload(7).unwrap().unwrap();
        assert_eq!(stored_payload, b"test data");
    }

    #[tokio::test]
    async fn test_coordinator_topology() {
        let coord = test_coordinator();
        let topo = coord.topology().await.unwrap();
        assert!(topo.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_replica_insert_lww_skips_stale() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        // Write with timestamp 100
        coord.replica_insert("default", 1, &[0.1; 8], &None, 100).await.unwrap();

        // Stale write with timestamp 50 should be skipped
        coord.replica_insert("default", 1, &[0.9; 8], &None, 50).await.unwrap();

        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 100);
        assert!((rec.data.unwrap()[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_delete_writes_tombstone() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.insert("default", 1, vec![0.1; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        assert!(ns.get_vector(1).unwrap().is_some());

        coord.delete("default", &[1], 0, ConsistencyLevel::One).await.unwrap();
        assert!(ns.get_vector(1).unwrap().is_none());

        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert!(rec.is_tombstone);
    }

    #[tokio::test]
    async fn test_coordinator_fetch_filters_tombstones() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.insert("default", 1, vec![0.1; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        coord.insert("default", 2, vec![0.2; 8], None, 0, ConsistencyLevel::One).await.unwrap();

        let records = coord.fetch("default", &[1, 2], ConsistencyLevel::One).await.unwrap();
        assert_eq!(records.len(), 2);

        coord.delete("default", &[1], 0, ConsistencyLevel::One).await.unwrap();

        let records = coord.fetch("default", &[1, 2], ConsistencyLevel::One).await.unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].id, 2);
    }

    #[tokio::test]
    async fn test_resolve_consistency_default() {
        let coord = test_coordinator();
        assert_eq!(coord.resolve_consistency(0), ConsistencyLevel::Quorum);
        assert_eq!(coord.resolve_consistency(1), ConsistencyLevel::One);
        assert_eq!(coord.resolve_consistency(2), ConsistencyLevel::Quorum);
        assert_eq!(coord.resolve_consistency(3), ConsistencyLevel::All);
    }

    #[tokio::test]
    async fn test_coordinator_insert_auto_id() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let id1 = coord.insert("default", 0, vec![0.1; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        let id2 = coord.insert("default", 0, vec![0.2; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        assert_eq!(id2, id1 + 1);
    }

    #[tokio::test]
    async fn test_create_collection_with_explicit_timestamp() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("ts_test", 8, 16, 4, 1, 777, ConsistencyLevel::One).await.unwrap();
        let key = "collection:ts_test";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.timestamp, 777);
        assert!(!meta.is_deleted);
    }

    #[tokio::test]
    async fn test_create_collection_duplicate_returns_false() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let result = coord.create_collection("dup_test", 8, 16, 4, 1, 100, ConsistencyLevel::One).await.unwrap();
        assert!(result);

        let result = coord.create_collection("dup_test", 8, 16, 4, 1, 100, ConsistencyLevel::One).await.unwrap();
        assert!(!result);

        // Newer timestamp should overwrite
        let result = coord.create_collection("dup_test", 8, 16, 4, 1, 200, ConsistencyLevel::One).await.unwrap();
        assert!(result);
    }

    #[tokio::test]
    async fn test_drop_collection_writes_tombstone() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("dropme", 8, 16, 4, 1, 100, ConsistencyLevel::One).await.unwrap();
        coord.drop_collection("dropme", 200, ConsistencyLevel::One).await.unwrap();

        let key = "collection:dropme";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(meta.is_deleted);
        assert_eq!(meta.timestamp, 200);
    }

    #[tokio::test]
    async fn test_drop_collection_lww_skips_stale() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("stale_drop", 8, 16, 4, 1, 300, ConsistencyLevel::One).await.unwrap();
        coord.drop_collection("stale_drop", 200, ConsistencyLevel::One).await.unwrap();

        let key = "collection:stale_drop";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(!meta.is_deleted);
        assert_eq!(meta.timestamp, 300);
    }

    #[tokio::test]
    async fn test_replicate_collection_lww_skips_stale() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let proto_cfg = crate::proto::CollectionConfig {
            dim: 8, nlist: 16, nprobe: 4, num_vector_shards: 6,
            replication_factor: 1, num_dim_groups: 4, dim_group_size: 2,
            pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
        };

        // Write with ts=500
        coord.replicate_collection("lww_coll", &proto_cfg, 500).await.unwrap();

        // Stale with ts=300 should be skipped
        let mut stale_cfg = proto_cfg.clone();
        stale_cfg.nlist = 999;
        coord.replicate_collection("lww_coll", &stale_cfg, 300).await.unwrap();

        let key = "collection:lww_coll";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.config.nlist, 16); // original, not stale
        assert_eq!(meta.timestamp, 500);
    }

    #[tokio::test]
    async fn test_replicate_drop_collection_lww() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let proto_cfg = crate::proto::CollectionConfig {
            dim: 8, nlist: 16, nprobe: 4, num_vector_shards: 6,
            replication_factor: 1, num_dim_groups: 4, dim_group_size: 2,
            pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
        };

        coord.replicate_collection("droplww", &proto_cfg, 100).await.unwrap();
        coord.replicate_drop_collection("droplww", 50).await.unwrap();

        let key = "collection:droplww";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(!meta.is_deleted); // stale drop skipped

        coord.replicate_drop_collection("droplww", 200).await.unwrap();
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(meta.is_deleted);
    }

    #[tokio::test]
    async fn test_insert_quorum_rf1_succeeds_locally() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let result = coord.insert("default", 99, vec![0.5; 8], None, 0, ConsistencyLevel::Quorum).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_collection_exists_after_create() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("exists_check", 8, 16, 4, 1, 100, ConsistencyLevel::One).await.unwrap();

        let key = "collection:exists_check";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(!meta.is_deleted);
        assert_eq!(meta.config.dim, 8);
    }

    #[tokio::test]
    async fn test_collection_config_timestamp_stored() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        let ts = 999888777;
        coord.create_collection("ts_stored", 16, 32, 8, 2, ts, ConsistencyLevel::One).await.unwrap();

        let key = "collection:ts_stored";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.timestamp, ts);
        assert_eq!(meta.config.dim, 16);
        assert_eq!(meta.config.replication_factor, 2);
    }

    // ── Consistent hashing routing tests ──

    #[tokio::test]
    async fn test_insert_routes_to_replicas_for_shard() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("rt", 8, 16, 4, 2, 100, ConsistencyLevel::One).await.unwrap();
        let result = coord.insert("rt", 42, vec![0.5; 8], None, 100, ConsistencyLevel::One).await;
        assert!(result.is_ok());
        let ns = coord.store().as_ref().clone().with_namespace("rt".into());
        assert!(ns.get_vector(42).unwrap().is_some());
    }

    #[tokio::test]
    async fn test_insert_shard_routing_computes_correct_shard() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("rt2", 4, 8, 2, 2, 100, ConsistencyLevel::One).await.unwrap();

        // Use explicit non-zero IDs to avoid auto-assignment
        let id1 = coord.insert("rt2", 10, vec![0.1; 4], None, 100, ConsistencyLevel::One).await.unwrap();
        assert_eq!(id1, 10);

        let id2 = coord.insert("rt2", 17, vec![0.2; 4], None, 100, ConsistencyLevel::One).await.unwrap();
        assert_eq!(id2, 17);

        let records = coord.fetch("rt2", &[10, 17], ConsistencyLevel::One).await.unwrap();
        assert_eq!(records.len(), 2);
    }

    #[tokio::test]
    async fn test_delete_routes_to_replicas_for_shard() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("rt3", 4, 8, 2, 2, 100, ConsistencyLevel::One).await.unwrap();

        // Insert vectors in different shards
        coord.insert("rt3", 10, vec![0.1; 4], None, 400, ConsistencyLevel::One).await.unwrap();
        coord.insert("rt3", 15, vec![0.2; 4], None, 400, ConsistencyLevel::One).await.unwrap();

        // Delete by id uses shard routing for each id
        let deleted = coord.delete("rt3", &[10, 15], 500, ConsistencyLevel::One).await.unwrap();
        assert_eq!(deleted, 2);

        let ns = coord.store().as_ref().clone().with_namespace("rt3".into());
        assert!(ns.get_vector(10).unwrap().is_none());
        assert!(ns.get_vector(15).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_search_uses_consistent_hashing_for_quorum() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("rt4", 4, 8, 2, 2, 100, ConsistencyLevel::One).await.unwrap();

        // Insert vectors so there's data to search
        for i in 0..20 {
            let v: Vec<f32> = (0..4).map(|d| (i * 4 + d) as f32).collect();
            coord.insert("rt4", i, v, None, 100, ConsistencyLevel::One).await.unwrap();
        }

        // Search with QUORUM — should work even without peers (local search returns results)
        let params = SearchParams { ef_search: 64, nprobe: 4, include_payloads: false, local_only: false };
        let result = coord.search("rt4", vec![0.0; 4], 5, params, ConsistencyLevel::Quorum).await;
        assert!(result.is_ok());
        let (points, stats) = result.unwrap();
        assert!(!points.is_empty());
        assert_eq!(stats.nodes_contacted, 1); // no peers → local only
    }

    #[tokio::test]
    async fn test_search_one_uses_local_only() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        coord.create_collection("rt5", 4, 8, 2, 2, 100, ConsistencyLevel::One).await.unwrap();

        for i in 0..10 {
            coord.insert("rt5", i, vec![0.0; 4], None, 100, ConsistencyLevel::One).await.unwrap();
        }

        let params = SearchParams { ef_search: 64, nprobe: 4, include_payloads: false, local_only: false };
        let result = coord.search("rt5", vec![0.0; 4], 3, params, ConsistencyLevel::One).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_insert_inherits_collection_replication_factor() {
        let coord = test_coordinator();
        let index = rekha_index::RekhaIndex::new().unwrap();
        coord.initialize(index).await;

        // Create collection with RF=3
        let result = coord.create_collection("rf3", 4, 8, 2, 3, 100, ConsistencyLevel::One).await;
        assert!(result.unwrap());

        // Read back the config to verify rf is stored
        let key = "collection:rf3";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.config.replication_factor, 3);

        // Insert with CL=QUORUM against RF=3 but no peers → should still work (local ack = 1, quorum(3)=2, but local ack counts)
        // Actually: quorum(3)=2, acks=1 → should FAIL with CL unmet
        // But with ONE, always works
        let result = coord.insert("rf3", 1, vec![0.5; 4], None, 0, ConsistencyLevel::One).await;
        assert!(result.is_ok());
    }
}
