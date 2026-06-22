use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub partition_id: u64,
    pub dim_groups: Vec<u32>,
    pub is_leader: bool,
    pub raft_term: u64,
    pub commit_index: u64,
    pub storage_bytes: u64,
    pub status: NodeStatus,
    #[serde(default)]
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    Healthy,
    Degraded,
    Unreachable,
    Recovering,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTopology {
    pub cluster_id: String,
    pub nodes: HashMap<String, NodeInfo>,
    pub partition_map: HashMap<(u64, u32), Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnedRange {
    pub vector_shard: u64,
    pub dim_start: usize,
    pub dim_end: usize,
}

impl OwnedRange {
    pub fn dim_count(&self) -> usize {
        self.dim_end.saturating_sub(self.dim_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_range_dim_count() {
        let r = OwnedRange {
            vector_shard: 0,
            dim_start: 0,
            dim_end: 128,
        };
        assert_eq!(r.dim_count(), 128);
        let r = OwnedRange {
            vector_shard: 1,
            dim_start: 128,
            dim_end: 64,
        };
        assert_eq!(r.dim_count(), 0);
    }
}
