use rekha_core::NodeInfo;
use rekha_partition::PartitionManager;

pub struct ReplicaRouter<'a> {
    partition_manager: &'a PartitionManager,
}

impl<'a> ReplicaRouter<'a> {
    pub fn new(partition_manager: &'a PartitionManager) -> Self {
        Self { partition_manager }
    }

    pub fn replicas_for(&self, shard: u64, rf: usize, exclude_self: &str) -> Vec<&NodeInfo> {
        self.partition_manager.replicas_for(shard, rf)
            .into_iter()
            .filter(|n| n.node_id != exclude_self)
            .collect()
    }

    pub fn shard_for(&self, id: u64, num_shards: u64) -> u64 {
        id % num_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::{NodeInfo, NodeStatus};
    use std::collections::HashMap;

    fn make_manager() -> PartitionManager {
        let mut nodes = HashMap::new();
        for i in 0..3 {
            nodes.insert(format!("n{}", i), NodeInfo {
                node_id: format!("n{}", i), address: format!("addr{}", i),
                partition_id: i as u64, dim_groups: vec![0],
                is_leader: false, raft_term: 0, commit_index: 0,
                storage_bytes: 0, status: NodeStatus::Healthy, last_heartbeat: 0,
            });
        }
        PartitionManager::new(nodes, 4, 768)
    }

    #[test]
    fn test_router() {
        let pm = make_manager();
        let router = ReplicaRouter::new(&pm);
        let replicas = router.replicas_for(0, 3, "n0");
        assert_eq!(replicas.len(), 2);
        assert!(!replicas.iter().any(|n| n.node_id == "n0"));
    }

    #[test]
    fn test_shard_for() {
        let pm = make_manager();
        let router = ReplicaRouter::new(&pm);
        assert_eq!(router.shard_for(42, 6), 0);
        assert_eq!(router.shard_for(7, 6), 1);
    }
}
