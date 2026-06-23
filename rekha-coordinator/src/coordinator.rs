use crate::config::parse_consistency;
use crate::config::CoordinatorConfig;
use crate::peer_pool::PeerPool;
use rekha_cluster::Membership;
use rekha_core::{ClusterTopology, CollectionConfig, CollectionMeta, ConsistencyLevel, NodeInfo, NodeStatus, VectorStoreBackend, now_micros};
use rekha_index::RekhaIndex;
use rekha_replication::HintedHandoff;
use rekha_storage::RocksVectorStore;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

pub struct Coordinator {
    pub(super) config: CoordinatorConfig,
    pub(super) index: Arc<RwLock<Option<RekhaIndex>>>,
    pub(super) store: Arc<RocksVectorStore>,
    pub(super) topology: Arc<RwLock<ClusterTopology>>,
    pub(super) initialized: Arc<RwLock<bool>>,
    pub(super) membership: Arc<RwLock<Membership>>,
    pub(super) peer_pool: Arc<RwLock<PeerPool>>,
    pub(super) handoff: HintedHandoff,
    pub(super) next_auto_id: AtomicU64,
}

impl Coordinator {
    pub fn new(
        config: CoordinatorConfig,
        store: Arc<RocksVectorStore>,
    ) -> Self {
        let starting_id = Self::starting_auto_id(&store);
        let handoff = HintedHandoff::new(config.hinted_handoff_enabled, config.max_hint_window_secs);
        let peer_timeout = Duration::from_millis(config.peer_timeout_ms);
        let self_node_id = config.node_id.clone();
        Self {
            config,
            index: Arc::new(RwLock::new(None)),
            store,
            topology: Arc::new(RwLock::new(ClusterTopology {
                cluster_id: String::new(), nodes: HashMap::new(), partition_map: HashMap::new(),
            })),
            initialized: Arc::new(RwLock::new(false)),
            membership: Arc::new(RwLock::new(Membership::with_timeout(self_node_id, peer_timeout))),
            peer_pool: Arc::new(RwLock::new(PeerPool::new())),
            handoff,
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
            let meta = CollectionMeta { config: cfg, timestamp: now_micros(), is_deleted: false, vector_count: 0 };
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
        let gc_grace = self.config.gc_grace_seconds;
        let hint_ttl = self.config.max_hint_window_secs;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600));
            loop {
                interval.tick().await;
                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                if let Ok(tombstones) = store.scan_tombstones() {
                    let expired: Vec<u64> = tombstones.into_iter()
                        .filter(|(_, ts)| { let tomb_secs = ts / 1_000_000; now_secs.saturating_sub(tomb_secs) >= gc_grace })
                        .map(|(id, _)| id).collect();
                    if !expired.is_empty() {
                        let count = expired.len();
                        let _ = store.physically_delete_vectors(&expired);
                        info!("GC collected {count} expired tombstones");
                    }
                }
            }
        });

        let store2 = self.store.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1800));
            loop {
                interval.tick().await;
                let deleted = store2.hint_store().delete_expired_hints(hint_ttl).unwrap_or(0);
                if deleted > 0 { info!("Expired {deleted} stale hints"); }
            }
        });
    }

    pub fn resolve_consistency(&self, proto_cl: i32) -> ConsistencyLevel {
        match proto_cl {
            1 => ConsistencyLevel::One,
            2 => ConsistencyLevel::Quorum,
            3 => ConsistencyLevel::All,
            _ => parse_consistency(&self.config.default_write_consistency).unwrap_or(ConsistencyLevel::Quorum),
        }
    }

    pub fn store(&self) -> &Arc<RocksVectorStore> { &self.store }

    pub fn cluster_id(&self) -> &str { "rekha-dev" }

    pub fn node_id(&self) -> &str { &self.config.node_id }

    pub fn bind_addr(&self) -> &str { &self.config.bind_addr }

    pub fn seed_nodes(&self) -> &[String] { &self.config.seed_nodes }

    pub fn config_ref(&self) -> &CoordinatorConfig { &self.config }

    pub fn local_node_info(&self) -> NodeInfo {
        NodeInfo {
            node_id: self.config.node_id.clone(),
            address: self.config.bind_addr.clone(),
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

    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use tempfile::TempDir;

    #[allow(deprecated)]
    pub(crate) fn temp_store() -> Arc<RocksVectorStore> {
        let dir = TempDir::new().unwrap();
        let p = dir.into_path();
        Arc::new(RocksVectorStore::open(&p).unwrap())
    }

    pub(crate) fn test_coordinator() -> Coordinator {
        let config = CoordinatorConfig::dev_default("test-node");
        let store = temp_store();
        Coordinator::new(config, store)
    }

    #[tokio::test]
    async fn test_coordinator_new() {
        let coord = test_coordinator();
        assert!(!coord.is_initialized().await);
    }

    #[tokio::test]
    async fn test_coordinator_initialize() {
        let coord = test_coordinator();
        coord.initialize(RekhaIndex::new().unwrap()).await;
        assert!(coord.is_initialized().await);
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
    async fn test_coordinator_topology() {
        let coord = test_coordinator();
        let topo = coord.topology().await.unwrap();
        assert!(topo.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_resolve_consistency_default() {
        let coord = test_coordinator();
        assert_eq!(coord.resolve_consistency(0), ConsistencyLevel::Quorum);
        assert_eq!(coord.resolve_consistency(1), ConsistencyLevel::One);
        assert_eq!(coord.resolve_consistency(2), ConsistencyLevel::Quorum);
        assert_eq!(coord.resolve_consistency(3), ConsistencyLevel::All);
    }
}
