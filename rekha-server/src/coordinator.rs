use rekha_core::{
    ClusterTopology, CollectionConfig, DistanceMetric, NodeInfo, NodeStatus, Payload,
    RekhaError, ScoredPoint, SearchParams, SearchStats, VectorIndex, VectorStoreBackend,
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

    pub async fn refresh(&mut self, peers: &[NodeInfo]) {
        let active: std::collections::HashSet<String> =
            peers.iter().map(|p| p.node_id.clone()).collect();
        self.clients.retain(|node_id, _| active.contains(node_id));

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

        let node_ids: Vec<String> = self.clients.keys().cloned().collect();
        for node_id in &node_ids {
            if let Some(client) = self.clients.get_mut(node_id) {
                match client.try_search(query, k, &peer_params).await {
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
    #[allow(dead_code)]
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
                peer_pool: Arc::new(RwLock::new(PeerPool::new("default"))),
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
                dim: 8,
                num_vector_shards: 6,
                replication_factor: 1,
                num_dim_groups: 4,
                dim_group_size: 2,
                nlist: 128,
                nprobe: 16,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 256,
                re_rank_k: 256,
            };
            if let Ok(json) = serde_json::to_vec(&cfg) {
                let _ = self.store.put_metadata(&default_key, &json);
            }
        }

        self.spawn_flush_loop();
        info!("Coordinator initialized");
    }

    fn spawn_flush_loop(&self) {
        let flush_ms = 1000;
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
                            tracing::warn!(
                                "Buffer flush failed: {e} (buffer: {buf_len} vectors)"
                            );
                        }
                    }
                }
            }
        });
    }

    fn collection_dim(&self) -> Result<usize, RekhaError> {
        let key = "collection:default".to_string();
        let data = self.store.get_metadata(&key)?
            .ok_or_else(|| RekhaError::NotFound("no default collection".into()))?;
        let cfg: CollectionConfig = serde_json::from_slice(&data)
            .map_err(|e| RekhaError::InvalidArgument(format!("bad collection config: {e}")))?;
        Ok(cfg.dim as usize)
    }

    pub async fn search(
        &self, query: Vec<f32>, k: usize, params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let index_guard = self.index.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal {
            detail: "index not initialized".into(),
        })?;

        let expected = self.collection_dim()?;
        if query.len() != expected {
            return Err(RekhaError::InvalidDimension { expected, actual: query.len() });
        }

        let mut candidates: Vec<ScoredPoint> = Vec::new();

        let (ids, dists) = index.search(&query, k * 2, &params).map_err(|e| {
            stats.warnings.push(format!("local search failed: {e}"));
            e
        })?;
        for (i, id) in ids.iter().enumerate() {
            let score = dists.get(i).copied().unwrap_or(f32::MAX);
            let payload = self.maybe_load_payload(params.include_payloads, *id);
            candidates.push(ScoredPoint { id: *id, score, payload });
        }

        if !params.local_only {
            let has_peers = { !self.peer_pool.read().await.is_empty() };
            if has_peers {
                let mut pool = self.peer_pool.write().await;
                let (peer_results, peer_stats) =
                    pool.search_fan_out(&query, k, &params).await;
                stats.nodes_contacted = 1 + peer_stats.nodes_contacted;
                stats.warnings.extend(peer_stats.warnings);
                candidates.extend(peer_results);
            } else {
                stats.nodes_contacted = 1;
            }
        } else {
            stats.nodes_contacted = 1;
        }

        candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        candidates.truncate(k * 2);

        let metric = DistanceMetric::L2;
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
        stats.vectors_scanned = candidates.len() as u64;

        Ok((candidates, stats))
    }

    fn maybe_load_payload(&self, include: bool, id: u64) -> Option<Payload> {
        if include {
            self.store
                .get_payload(id)
                .ok()
                .flatten()
                .map(Payload::from_bytes)
        } else {
            None
        }
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

        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            index.buffer_insert_internal(id, vector.clone());
        }
        drop(idx);

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
        let mut config = ServerConfig::dev_default("test-node", "/tmp/rekha_coord_test");
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
        let store = (*coord.store()).clone();
        let mut index = rekha_index::RekhaIndex::new((*store).clone()).unwrap();
        for i in 0..10 {
            let v: Vec<f32> = (0..8).map(|d| (i * 8 + d) as f32).collect();
            index.buffer_insert_internal(i, v);
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
}
