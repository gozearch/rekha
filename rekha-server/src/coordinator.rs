use rekha_core::{
    ClusterTopology, CollectionConfig, DistanceMetric, NodeInfo, NodeStatus, Payload,
    RekhaError, ScoredPoint, SearchParams, SearchStats, VectorStoreBackend,
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
            .search_with_params(query.to_vec(), collection, k, params.clone())
            .await
    }

    async fn try_remote_insert(
        &mut self, collection: &str, id: u64, vector: &[f32], payload: &Option<Vec<u8>>,
    ) -> Result<(), RekhaError> {
        self.last_used = Instant::now();
        self.client
            .replica_insert(id, vector.to_vec(), collection, payload.clone())
            .await?;
        Ok(())
    }

    async fn try_remote_create_collection(
        &mut self, name: &str, config: &crate::proto::CollectionConfig,
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
            .replica_create_collection(name, client_cfg)
            .await
    }

    async fn try_remote_drop_collection(&mut self, name: &str) -> Result<bool, RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_drop_collection(name).await
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
            if let Ok(json) = serde_json::to_vec(&cfg) {
                let _ = self.store.put_metadata(&default_key, &json);
            }
        }

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection("default", 8, 128, 16);
        }
        drop(idx);

        self.spawn_flush_loop();
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

    pub async fn search(
        &self, collection: &str, query: Vec<f32>, k: usize, params: SearchParams,
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
            candidates.push(ScoredPoint { id: *id, score, payload });
        }

        if !params.local_only {
            let has_peers = { !self.peer_pool.read().await.is_empty() };
            if has_peers {
                let mut pool = self.peer_pool.write().await;
                let (peer_results, peer_stats) =
                    pool.search_fan_out(&query, k, &params, collection).await;
                stats.nodes_contacted = 1 + peer_stats.nodes_contacted;
                stats.warnings.extend(peer_stats.warnings);
                candidates.extend(peer_results);
            } else {
                stats.nodes_contacted = 1;
            }
        } else {
            stats.nodes_contacted = 1;
        }

        let mut seen = std::collections::HashSet::new();
        candidates.retain(|c| seen.insert(c.id));

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
        &self, name: &str, proto_cfg: &crate::proto::CollectionConfig,
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
        let json = serde_json::to_vec(&cfg).map_err(|e| {
            RekhaError::InvalidArgument(format!("serialize config: {e}"))
        })?;
        let key = format!("collection:{name}");
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection(name, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
        }
        Ok(true)
    }

    pub async fn replicate_drop_collection(&self, name: &str) -> Result<bool, RekhaError> {
        let key = format!("collection:{name}");
        self.store.delete_metadata(&key)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.drop_collection(name);
        }
        Ok(true)
    }

    pub async fn drop_collection(&self, name: &str) -> Result<bool, RekhaError> {
        self.replicate_drop_collection(name).await?;

        let peer_ids: Vec<String> = {
            let pool = self.peer_pool.read().await;
            pool.clients.keys().cloned().collect()
        };
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                let _ = client.try_remote_drop_collection(name).await;
            }
        }
        Ok(true)
    }

    pub async fn create_collection(
        &self, name: &str, dim: u32, nlist: u32, nprobe: u32, rf: u64,
    ) -> Result<bool, RekhaError> {
        let key = format!("collection:{name}");
        if self.store.get_metadata(&key)?.is_some() {
            return Ok(false);
        }

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

        self.replicate_collection(name, &proto_cfg).await?;

        let peer_ids: Vec<String> = {
            let pool = self.peer_pool.read().await;
            pool.clients.keys().cloned().collect()
        };
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                let _ = client.try_remote_create_collection(name, &proto_cfg).await;
            }
        }
        Ok(true)
    }

    pub async fn replica_insert(
        &self, collection: &str, id: u64, vector: &[f32], payload: &Option<Payload>,
    ) -> Result<u64, RekhaError> {
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            if let Err(e) = index.insert(collection, id, vector) {
                if matches!(&e, RekhaError::NotFound(_)) {
                    if let Some(cfg) = self.read_collection_config(collection) {
                        let _ = index.create_collection(collection, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
                        let _ = index.insert(collection, id, vector);
                    }
                }
            }
        }
        drop(idx);
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        ns.put_vector(id, vector)?;
        if let Some(ref p) = payload {
            ns.put_payload(id, &p.data)?;
        }
        Ok(id)
    }

    fn read_collection_config(&self, name: &str) -> Option<CollectionConfig> {
        let key = format!("collection:{name}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            serde_json::from_slice(&data).ok()
        } else {
            None
        }
    }

    pub async fn insert(
        &self, collection: &str, id: u64, vector: Vec<f32>, payload: Option<Payload>,
    ) -> Result<u64, RekhaError> {
        let id = if id == 0 { self.next_auto_id.fetch_add(1, Ordering::SeqCst) } else { id };

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            index.insert(collection, id, &vector)?;
        }
        drop(idx);

        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        ns.put_vector(id, &vector)?;
        if let Some(ref p) = payload {
            ns.put_payload(id, &p.data)?;
        }

        let pdata = payload.as_ref().map(|p| p.data.clone());
        if let Some(cfg) = self.read_collection_config(collection) {
            let shard = id % cfg.num_vector_shards;
            let pm = self.partition_manager.read().await;
            let replicas = pm.replicas_for(shard, cfg.replication_factor as usize);
            for replica in replicas {
                if replica.node_id != self.node_id() {
                    let mut pool = self.peer_pool.write().await;
                    if let Some(client) = pool.clients.get_mut(&replica.node_id) {
                        let _ = client.try_remote_insert(collection, id, &vector, &pdata).await;
                    }
                }
            }
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
        let result = coord.search("default", vec![0.0; 8], 5, SearchParams::default()).await;
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
        coord.insert("default", 42, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], None).await.unwrap();
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
        coord.insert("default", 7, vec8, Some(payload)).await.unwrap();
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
}
