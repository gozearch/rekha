use rekha_core::{NodeInfo, NodeStatus, OwnedRange, PartitionError, RekhaError};
use std::collections::{HashMap, HashSet};

use crate::ring::ConsistentHashRing;

pub struct PartitionManager {
    nodes: HashMap<String, NodeInfo>,
    topology: HashMap<(u64, u32), Vec<String>>,
    owned_ranges: HashMap<String, Vec<OwnedRange>>,
    ring: ConsistentHashRing,
    num_dim_groups: u32,
    dims_per_group: usize,
}

impl PartitionManager {
    /// Create a new partition manager.
    pub fn new(nodes: HashMap<String, NodeInfo>, num_dim_groups: u32, total_dim: usize) -> Self {
        let dims_per_group = if num_dim_groups > 0 {
            total_dim / num_dim_groups as usize
        } else {
            total_dim
        };

        let mut manager = Self {
            nodes,
            topology: HashMap::new(),
            owned_ranges: HashMap::new(),
            ring: ConsistentHashRing::new(128),
            num_dim_groups,
            dims_per_group,
        };

        manager.rebuild_topology();
        manager
    }

    /// Rebuild topology from current node assignments.
    fn rebuild_topology(&mut self) {
        self.topology.clear();
        self.owned_ranges.clear();
        self.rebuild_ring();

        for (node_id, info) in &self.nodes {
            let range = OwnedRange {
                vector_shard: info.partition_id,
                dim_start: info.dim_groups.first().copied().unwrap_or(0) as usize
                    * self.dims_per_group,
                dim_end: (info.dim_groups.last().copied().unwrap_or(0) as usize + 1)
                    * self.dims_per_group,
            };

            self.owned_ranges
                .entry(node_id.clone())
                .or_default()
                .push(range.clone());

            for &dim_group in &info.dim_groups {
                self.topology
                    .entry((info.partition_id, dim_group))
                    .or_default()
                    .push(node_id.clone());
            }
        }
    }

    /// Register a node and its partition ownership.
    pub fn register_node(&mut self, info: NodeInfo) {
        self.nodes.insert(info.node_id.clone(), info);
        self.rebuild_topology();
    }

    /// Remove a node from the cluster.
    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        self.rebuild_topology();
    }

    /// Get all nodes that can serve a given (vector_shard, dim_group) partition.
    pub fn nodes_for_partition(
        &self,
        vector_shard: u64,
        dim_group: u32,
    ) -> Result<&[String], RekhaError> {
        self.topology
            .get(&(vector_shard, dim_group))
            .map(|v| v.as_slice())
            .ok_or_else(|| {
                PartitionError::NoNodesAvailable {
                    partition_id: vector_shard,
                }
                .into()
            })
    }

    /// Get the owned ranges for a specific node.
    pub fn node_ranges(&self, node_id: &str) -> Option<&[OwnedRange]> {
        self.owned_ranges.get(node_id).map(|v| v.as_slice())
    }

    /// Get dimension range for a dimension group.
    pub fn dim_group_range(&self, group: u32) -> Option<(usize, usize)> {
        if group >= self.num_dim_groups {
            return None;
        }
        let start = (group as usize) * self.dims_per_group;
        let end = start + self.dims_per_group;
        Some((start, end))
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

    pub fn all_nodes(&self) -> &HashMap<String, NodeInfo> {
        &self.nodes
    }

    pub fn replicas_for(&self, shard: u64, rf: usize) -> Vec<&NodeInfo> {
        let healthy: HashSet<&str> = self
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.status, NodeStatus::Healthy))
            .map(|(id, _)| id.as_str())
            .collect();

        let ids = self.ring.replicas_for(shard, rf, &healthy);
        ids.iter()
            .filter_map(|id| self.nodes.get(id))
            .collect()
    }

    pub fn mark_node_down(&mut self, node_id: &str) {
        if let Some(info) = self.nodes.get_mut(node_id) {
            info.status = NodeStatus::Unreachable;
        }
        self.ring.remove_node(node_id);
    }

    fn rebuild_ring(&mut self) {
        self.ring = ConsistentHashRing::new(128);
        for node_id in self.nodes.keys() {
            self.ring.add_node(node_id);
        }
    }

    pub fn node_ring_contains(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_manager() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "node-1".into(),
            NodeInfo {
                node_id: "node-1".into(),
                address: "10.0.0.1:50051".into(),
                partition_id: 0,
                dim_groups: vec![0, 1],
                is_leader: true,
                raft_term: 1,
                commit_index: 100,
                storage_bytes: 1024,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );

        let manager = PartitionManager::new(nodes, 4, 768);
        assert_eq!(manager.node_count(), 1);
        assert_eq!(manager.dims_per_group, 192);

        let nodes = manager.nodes_for_partition(0, 0).unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0], "node-1");
    }

    #[test]
    fn test_manager_empty() {
        let manager = PartitionManager::new(HashMap::new(), 4, 768);
        assert_eq!(manager.node_count(), 0);
        assert!(manager.nodes_for_partition(0, 0).is_err());
    }

    #[test]
    fn test_manager_node_for_shard() {
        let mut nodes = HashMap::new();
        for i in 0..4 {
            let dims: Vec<u32> = (0..2).collect();
            nodes.insert(
                format!("node-{}", i),
                NodeInfo {
                    node_id: format!("node-{}", i),
                    address: format!("10.0.0.{}:50051", i + 1),
                    partition_id: i as u64,
                    dim_groups: dims,
                    is_leader: i == 0,
                    raft_term: 1,
                    commit_index: 50,
                    storage_bytes: 512,
                    status: NodeStatus::Healthy,
                    last_heartbeat: 0,
                },
            );
        }

        let manager = PartitionManager::new(nodes, 4, 768);
        let nodes_for_p0 = manager.nodes_for_partition(0, 0).unwrap();
        assert!(!nodes_for_p0.is_empty());
    }

    #[test]
    fn test_manager_dims_per_group_calculation() {
        let manager = PartitionManager::new(HashMap::new(), 4, 1024);
        assert_eq!(manager.dims_per_group, 256);
    }

    #[test]
    fn test_register_node() {
        let mut manager = PartitionManager::new(HashMap::new(), 4, 768);
        assert_eq!(manager.node_count(), 0);
        let info = NodeInfo {
            node_id: "new-node".into(),
            address: "10.0.0.5:50051".into(),
            partition_id: 1,
            dim_groups: vec![0, 2],
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        manager.register_node(info);
        assert_eq!(manager.node_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "node-1".into(),
            NodeInfo {
                node_id: "node-1".into(),
                address: "10.0.0.1:50051".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: true,
                raft_term: 1,
                commit_index: 10,
                storage_bytes: 100,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        let mut manager = PartitionManager::new(nodes, 4, 768);
        assert_eq!(manager.node_count(), 1);
        manager.remove_node("node-1");
        assert_eq!(manager.node_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut manager = PartitionManager::new(HashMap::new(), 4, 768);
        manager.remove_node("nonexistent"); // should not panic
        assert_eq!(manager.node_count(), 0);
    }

    #[test]
    fn test_healthy_nodes_all_healthy() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n1".into(),
            NodeInfo {
                node_id: "n1".into(),
                address: "addr1".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        let manager = PartitionManager::new(nodes, 4, 768);
        let healthy = manager.healthy_nodes();
        assert_eq!(healthy.len(), 1);
    }

    #[test]
    fn test_healthy_nodes_mixed() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n1".into(),
            NodeInfo {
                node_id: "n1".into(),
                address: "addr1".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        nodes.insert(
            "n2".into(),
            NodeInfo {
                node_id: "n2".into(),
                address: "addr2".into(),
                partition_id: 1,
                dim_groups: vec![1],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Unreachable,
                last_heartbeat: 0,
            },
        );
        let manager = PartitionManager::new(nodes, 4, 768);
        let healthy = manager.healthy_nodes();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].node_id, "n1");
    }

    #[test]
    fn test_node_ranges() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n1".into(),
            NodeInfo {
                node_id: "n1".into(),
                address: "addr".into(),
                partition_id: 0,
                dim_groups: vec![0, 1],
                is_leader: true,
                raft_term: 1,
                commit_index: 5,
                storage_bytes: 100,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        let manager = PartitionManager::new(nodes, 4, 768);
        let ranges = manager.node_ranges("n1").unwrap();
        assert!(!ranges.is_empty());
        assert_eq!(ranges[0].vector_shard, 0);
    }

    #[test]
    fn test_dim_group_range() {
        let manager = PartitionManager::new(HashMap::new(), 4, 768);
        let (start, end) = manager.dim_group_range(0).unwrap();
        assert_eq!(start, 0);
        assert_eq!(end, 192);
        let (start, end) = manager.dim_group_range(3).unwrap();
        assert_eq!(start, 576);
        assert_eq!(end, 768);
        assert!(manager.dim_group_range(4).is_none());
    }

    #[test]
    fn test_manager_all_nodes() {
        let mut nodes = HashMap::new();
        for i in 0..3 {
            nodes.insert(
                format!("n{}", i),
                NodeInfo {
                    node_id: format!("n{}", i),
                    address: format!("addr{}", i),
                    partition_id: i as u64,
                    dim_groups: vec![0],
                    is_leader: false,
                    raft_term: 0,
                    commit_index: 0,
                    storage_bytes: 0,
                    status: NodeStatus::Healthy,
                    last_heartbeat: 0,
                },
            );
        }
        let manager = PartitionManager::new(nodes, 4, 768);
        assert_eq!(manager.node_count(), 3);
    }

    #[test]
    fn test_replicas_rf2_3_nodes() {
        let mut nodes = HashMap::new();
        for i in 0..3 {
            nodes.insert(
                format!("n{}", i),
                NodeInfo {
                    node_id: format!("n{}", i),
                    address: format!("addr{}", i),
                    partition_id: i as u64,
                    dim_groups: vec![0],
                    is_leader: false,
                    raft_term: 0,
                    commit_index: 0,
                    storage_bytes: 0,
                    status: NodeStatus::Healthy,
                    last_heartbeat: 0,
                },
            );
        }
        let manager = PartitionManager::new(nodes, 4, 768);
        let replicas = manager.replicas_for(0, 2);
        assert_eq!(replicas.len(), 2);
        let mut ids: Vec<&str> = replicas.iter().map(|n| n.node_id.as_str()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_replicas_skips_unreachable() {
        let mut nodes = HashMap::new();
        for i in 0..3 {
            nodes.insert(
                format!("n{}", i),
                NodeInfo {
                    node_id: format!("n{}", i),
                    address: format!("addr{}", i),
                    partition_id: i as u64,
                    dim_groups: vec![0],
                    is_leader: false,
                    raft_term: 0,
                    commit_index: 0,
                    storage_bytes: 0,
                    status: if i == 1 { NodeStatus::Unreachable } else { NodeStatus::Healthy },
                    last_heartbeat: 0,
                },
            );
        }
        let mut manager = PartitionManager::new(nodes, 4, 768);
        manager.mark_node_down("n1");
        let replicas = manager.replicas_for(0, 2);
        assert_eq!(replicas.len(), 2);
        let ids: Vec<&str> = replicas.iter().map(|n| n.node_id.as_str()).collect();
        assert!(!ids.contains(&"n1"));
    }

    #[test]
    fn test_replicas_rf_exceeds_nodes() {
        let mut nodes = HashMap::new();
        for i in 0..2 {
            nodes.insert(
                format!("n{}", i),
                NodeInfo {
                    node_id: format!("n{}", i),
                    address: format!("addr{}", i),
                    partition_id: i as u64,
                    dim_groups: vec![0],
                    is_leader: false,
                    raft_term: 0,
                    commit_index: 0,
                    storage_bytes: 0,
                    status: NodeStatus::Healthy,
                    last_heartbeat: 0,
                },
            );
        }
        let manager = PartitionManager::new(nodes, 4, 768);
        let replicas = manager.replicas_for(0, 5);
        assert_eq!(replicas.len(), 2);
    }

    #[test]
    fn test_replicas_no_healthy_nodes() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n0".into(),
            NodeInfo {
                node_id: "n0".into(),
                address: "addr".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Unreachable,
                last_heartbeat: 0,
            },
        );
        let mut manager = PartitionManager::new(nodes, 4, 768);
        manager.mark_node_down("n0");
        let replicas = manager.replicas_for(0, 3);
        assert!(replicas.is_empty());
    }

    #[test]
    fn test_replicas_ring_order() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "node-b".into(),
            NodeInfo {
                node_id: "node-b".into(),
                address: "b".into(),
                partition_id: 1,
                dim_groups: vec![0],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        nodes.insert(
            "node-a".into(),
            NodeInfo {
                node_id: "node-a".into(),
                address: "a".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: true,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        let manager = PartitionManager::new(nodes, 4, 768);
        let replicas_0 = manager.replicas_for(0, 1);
        assert_eq!(replicas_0.len(), 1);
        assert!(["node-a", "node-b"].contains(&replicas_0[0].node_id.as_str()));
        let replicas_1 = manager.replicas_for(1, 1);
        assert_eq!(replicas_1.len(), 1);
        assert!(["node-a", "node-b"].contains(&replicas_1[0].node_id.as_str()));
    }

    #[test]
    fn test_mark_node_down() {
        let mut nodes = HashMap::new();
        nodes.insert(
            "n0".into(),
            NodeInfo {
                node_id: "n0".into(),
                address: "addr".into(),
                partition_id: 0,
                dim_groups: vec![0],
                is_leader: false,
                raft_term: 0,
                commit_index: 0,
                storage_bytes: 0,
                status: NodeStatus::Healthy,
                last_heartbeat: 0,
            },
        );
        let mut manager = PartitionManager::new(nodes, 4, 768);
        manager.mark_node_down("n0");
        let replicas = manager.replicas_for(0, 1);
        assert!(replicas.is_empty());
    }

    #[test]
    fn test_replicas_empty_when_no_nodes() {
        let manager = PartitionManager::new(HashMap::new(), 4, 768);
        let replicas = manager.replicas_for(0, 1);
        assert!(replicas.is_empty());
    }
}
