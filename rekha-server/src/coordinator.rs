use async_trait::async_trait;
use rekha_core::{
    ClusterTopology, Coordinator as CoordinatorTrait, NodeInfo, NodeStatus, Payload,
    RekhaError, ScoredPoint, SearchParams, SearchStats, VectorIndex, VectorStoreBackend,
};
use rekha_index::RekhaIndex;
use rekha_partition::PartitionManager;
use rekha_raft::RaftNode;
use rekha_storage::RocksVectorStore;

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

use crate::config::ServerConfig;

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
    #[allow(dead_code)]
    raft_nodes: DashMap<u64, Arc<RaftNode>>,
    /// Cluster topology (peer nodes).
    topology: Arc<RwLock<ClusterTopology>>,
    /// Whether this coordinator is initialized.
    initialized: Arc<RwLock<bool>>,
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
        }
    }

    /// Initialize the coordinator with an index.
    pub async fn initialize(&self, index: RekhaIndex) {
        let mut idx = self.index.write().await;
        *idx = Some(index);
        *self.initialized.write().await = true;
        info!("Coordinator initialized");
    }

    /// Check if initialized.
    pub async fn is_initialized(&self) -> bool {
        *self.initialized.read().await
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
        }
    }

    /// Get a reference to the local store.
    pub fn store(&self) -> &Arc<RocksVectorStore> {
        &self.store
    }
}

#[async_trait]
impl CoordinatorTrait for Coordinator {
    async fn search(
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

        let _dim_count = self.config.partition.num_dim_groups as usize;
        let num_groups = self.config.partition.num_dim_groups;
        let total_dim = query.len();
        let dims_per_group = total_dim / num_groups as usize;

        // Multi-granularity search:
        // 1. Run search on each dimension group with early-stop
        // 2. Merge results from all groups
        let mut all_candidates: Vec<ScoredPoint> = Vec::new();
        let mut dim_groups_contacted = 0u32;

        for group in 0..num_groups {
            let start_dim = (group as usize) * dims_per_group;
            let end_dim = start_dim + dims_per_group;

            match index.search_dim_range(&query, k * 2, start_dim, end_dim, &params) {
                Ok((ids, dists)) => {
                    dim_groups_contacted += 1;
                    for (i, id) in ids.iter().enumerate() {
                        let score = dists.get(i).copied().unwrap_or(f32::MAX);

                        // Early-stop: if this candidate's partial distance
                        // already exceeds our current k-th best, skip.
                        if all_candidates.len() >= k {
                            all_candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
                            if score > all_candidates[k - 1].score {
                                continue;
                            }
                        }

                        // Fetch payload if requested.
                        let payload = if params.include_payloads {
                            self.store
                                .get_payload(*id)
                                .ok()
                                .flatten()
                                .map(|data| Payload::from_bytes(data))
                        } else {
                            None
                        };

                        all_candidates.push(ScoredPoint {
                            id: *id,
                            score,
                            payload,
                        });
                    }
                }
                Err(e) => {
                    stats.warnings.push(format!(
                        "dim_group {group} search failed: {e}"
                    ));
                }
            }
        }

        // 2. Sort all candidates by score and take top-k.
        all_candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_candidates.truncate(k);

        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.nodes_contacted = dim_groups_contacted;
        stats.vectors_scanned = index.len() as u64;

        Ok((all_candidates, stats))
    }

    async fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Payload>,
    ) -> Result<(), RekhaError> {
        // Store locally.
        self.store.put_vector(id, &vector)?;

        if let Some(ref p) = payload {
            self.store.put_payload(id, &p.data)?;
        }

        Ok(())
    }

    async fn topology(&self) -> Result<ClusterTopology, RekhaError> {
        let topo = self.topology.read().await;
        Ok(topo.clone())
    }

    async fn node_info(&self, _node_id: &str) -> Result<NodeInfo, RekhaError> {
        Ok(self.local_node_info())
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
        let pm = Arc::new(RwLock::new(
            rekha_partition::PartitionManager::new(HashMap::new(), 4, 768)
        ));
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
            8, 4, 16, 4, (*store).clone(), rekha_core::DistanceMetric::L2
        ).unwrap();
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
        let result = coord.search(
            vec![0.0; 8], 5, SearchParams::default()
        ).await;
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
}
