use rekha_core::{NodeInfo, NodeStatus, OwnedRange, PartitionError, RekhaError};
use std::collections::HashMap;

/// Manages shard-to-node assignments and cluster topology.
///
/// Tracks:
/// - Which nodes own which shards (shard_id → [node_ids])
/// - Node health and leadership status
pub struct PartitionManager {
    /// Node ID -> NodeInfo mapping.
    nodes: HashMap<String, NodeInfo>,
    /// shard_id -> list of node IDs that own this shard.
    shard_topology: HashMap<u64, Vec<String>>,
    /// Node ID -> list of owned ranges.
    owned_ranges: HashMap<String, Vec<OwnedRange>>,
    /// Total number of shards in the cluster.
    num_shards: u64,
}

impl PartitionManager {
    /// Create a new partition manager.
    pub fn new(nodes: HashMap<String, NodeInfo>, num_shards: u64) -> Self {
        let mut manager = Self {
            nodes,
            shard_topology: HashMap::new(),
            owned_ranges: HashMap::new(),
            num_shards,
        };
        manager.rebuild_topology();
        manager
    }

    /// Rebuild topology from current node assignments.
    fn rebuild_topology(&mut self) {
        self.shard_topology.clear();
        self.owned_ranges.clear();

        for (node_id, info) in &self.nodes {
            let range = OwnedRange {
                vector_shard: info.partition_id,
                dim_start: 0,
                dim_end: 0,
            };

            self.owned_ranges
                .entry(node_id.clone())
                .or_default()
                .push(range);

            self.shard_topology
                .entry(info.partition_id)
                .or_default()
                .push(node_id.clone());
        }
    }

    /// Register a node and its shard ownership.
    pub fn register_node(&mut self, info: NodeInfo) {
        self.nodes.insert(info.node_id.clone(), info);
        self.rebuild_topology();
    }

    /// Remove a node from the cluster.
    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        self.rebuild_topology();
    }

    /// Get all nodes that own a given shard.
    pub fn nodes_for_shard(&self, shard_id: u64) -> Result<&[String], RekhaError> {
        self.shard_topology
            .get(&shard_id)
            .map(|v| v.as_slice())
            .ok_or_else(|| {
                PartitionError::NoNodesAvailable {
                    partition_id: shard_id,
                }
                .into()
            })
    }

    /// Get the owned ranges for a specific node.
    pub fn node_ranges(&self, node_id: &str) -> Option<&[OwnedRange]> {
        self.owned_ranges.get(node_id).map(|v| v.as_slice())
    }

    /// Get all healthy nodes.
    pub fn healthy_nodes(&self) -> Vec<&NodeInfo> {
        self.nodes
            .values()
            .filter(|n| matches!(n.status, NodeStatus::Healthy | NodeStatus::Degraded))
            .collect()
    }

    /// Number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// All registered nodes.
    pub fn all_nodes(&self) -> &HashMap<String, NodeInfo> {
        &self.nodes
    }

    /// Total number of shards configured.
    pub fn num_shards(&self) -> u64 {
        self.num_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, addr: &str, shard: u64) -> NodeInfo {
        NodeInfo {
            node_id: id.into(),
            address: addr.into(),
            partition_id: shard,
            dim_groups: vec![],
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        }
    }

    #[test]
    fn test_partition_manager_single_node() {
        let mut nodes = HashMap::new();
        nodes.insert("node-1".into(), make_node("node-1", "10.0.0.1:50051", 0));
        let manager = PartitionManager::new(nodes, 1);
        assert_eq!(manager.node_count(), 1);

        let shard_nodes = manager.nodes_for_shard(0).unwrap();
        assert_eq!(shard_nodes.len(), 1);
        assert_eq!(shard_nodes[0], "node-1");
    }

    #[test]
    fn test_manager_empty() {
        let manager = PartitionManager::new(HashMap::new(), 4);
        assert_eq!(manager.node_count(), 0);
        assert!(manager.nodes_for_shard(0).is_err());
    }

    #[test]
    fn test_manager_multiple_nodes() {
        let mut nodes = HashMap::new();
        for i in 0..4 {
            nodes.insert(
                format!("node-{}", i),
                make_node(
                    &format!("node-{}", i),
                    &format!("10.0.0.{}:50051", i + 1),
                    i as u64,
                ),
            );
        }
        let manager = PartitionManager::new(nodes, 4);
        for i in 0..4 {
            let shard_nodes = manager.nodes_for_shard(i).unwrap();
            assert_eq!(shard_nodes.len(), 1);
        }
    }

    #[test]
    fn test_register_node() {
        let mut manager = PartitionManager::new(HashMap::new(), 4);
        assert_eq!(manager.node_count(), 0);
        manager.register_node(make_node("new-node", "10.0.0.5:50051", 1));
        assert_eq!(manager.node_count(), 1);
        let shard_nodes = manager.nodes_for_shard(1).unwrap();
        assert_eq!(shard_nodes[0], "new-node");
    }

    #[test]
    fn test_remove_node() {
        let mut nodes = HashMap::new();
        nodes.insert("node-1".into(), make_node("node-1", "10.0.0.1:50051", 0));
        let mut manager = PartitionManager::new(nodes, 4);
        assert_eq!(manager.node_count(), 1);
        manager.remove_node("node-1");
        assert_eq!(manager.node_count(), 0);
        assert!(manager.nodes_for_shard(0).is_err());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut manager = PartitionManager::new(HashMap::new(), 4);
        manager.remove_node("nonexistent");
        assert_eq!(manager.node_count(), 0);
    }

    #[test]
    fn test_healthy_nodes() {
        let mut nodes = HashMap::new();
        nodes.insert("n1".into(), make_node("n1", "addr1", 0));
        let mut n2 = make_node("n2", "addr2", 1);
        n2.status = NodeStatus::Unreachable;
        nodes.insert("n2".into(), n2);
        let manager = PartitionManager::new(nodes, 4);
        let healthy = manager.healthy_nodes();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].node_id, "n1");
    }

    #[test]
    fn test_node_ranges() {
        let mut nodes = HashMap::new();
        nodes.insert("n1".into(), make_node("n1", "addr", 0));
        let manager = PartitionManager::new(nodes, 4);
        let ranges = manager.node_ranges("n1").unwrap();
        assert!(!ranges.is_empty());
        assert_eq!(ranges[0].vector_shard, 0);
    }

    #[test]
    fn test_num_shards() {
        let manager = PartitionManager::new(HashMap::new(), 8);
        assert_eq!(manager.num_shards(), 8);
    }
}
