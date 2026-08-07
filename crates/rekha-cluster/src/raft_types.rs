//! Raft type configuration and cluster operations for openraft integration.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterOperation {
    AddNode {
        node_id: u64,
        addr: String,
    },
    RemoveNode {
        node_id: u64,
    },
    AddCollection {
        collection_id: Uuid,
        name: String,
        dimension: u32,
        distance: String,
        tenant: String,
        database: String,
        owner_nodes: Vec<u64>,
    },
    RemoveCollection {
        collection_id: Uuid,
    },
    TransferLeadership {
        new_leader: u64,
    },
}

openraft::declare_raft_types!(
    pub RaftTypeConfig:
        D            = ClusterOperation,
        R            = String,
        NodeId       = u64,
        Node         = u64,
        Entry        = openraft::Entry<Self>,
        SnapshotData = std::io::Cursor<Vec<u8>>,
        AsyncRuntime = openraft::TokioRuntime,
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: u64,
    pub addr: String,
    pub is_leader: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cluster_operation_roundtrip() {
        let ops = vec![
            ClusterOperation::AddNode {
                node_id: 1,
                addr: "127.0.0.1:8001".into(),
            },
            ClusterOperation::RemoveNode { node_id: 2 },
            ClusterOperation::AddCollection {
                collection_id: Uuid::new_v4(),
                name: "test".into(),
                dimension: 128,
                distance: "l2".into(),
                tenant: "default".into(),
                database: "default".into(),
                owner_nodes: vec![1, 2, 3],
            },
            ClusterOperation::RemoveCollection {
                collection_id: Uuid::new_v4(),
            },
            ClusterOperation::TransferLeadership { new_leader: 3 },
        ];
        for op in &ops {
            let bytes = bincode::serialize(op).unwrap();
            let decoded: ClusterOperation = bincode::deserialize(&bytes).unwrap();
            assert_eq!(*op, decoded);
        }
    }
}
