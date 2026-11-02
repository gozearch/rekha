//! Raft type configuration and cluster operations for openraft integration.

use rekha_core::auth::{Permission, RoleRecord, UserRecord};
use rekha_core::cluster::ShardId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// `Eq` is not derivable: a collection config carries `f64` metadata values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClusterOperation {
    AddNode {
        node_id: u64,
        addr: String,
    },
    RemoveNode {
        node_id: u64,
    },
    /// Create a tenant namespace.
    ///
    /// A namespace is a cluster fact, not a node fact: a tenant created on one
    /// node and unknown to the others would make `GET /tenants/{name}` answer
    /// differently depending on which node a load balancer picked. Idempotent,
    /// so a replayed entry or a duplicate request is not an error.
    CreateTenant {
        name: String,
    },
    /// Create a database inside a tenant. Idempotent for the same reason.
    CreateDatabase {
        tenant: String,
        name: String,
    },
    AddCollection {
        /// The exact collection definition. Replicating the resolved config
        /// rather than a few of its fields means a follower materializes the
        /// same collection the leader created, instead of reconstructing one
        /// that may differ in distance space, HNSW tuning, metadata or creation
        /// time.
        config: rekha_core::config::CollectionConfig,
        /// The shard this collection was placed on.
        ///
        /// Decided once by the metadata leader (see `Placement::place`) and
        /// recorded here, in the state machine, and on the catalog row. A hash
        /// over the collection id cannot reproduce the decision, so it is
        /// carried rather than recomputed.
        ///
        /// `None` on a single-node deployment, which has no shards at all. It is
        /// not "unplaced": it is the absence of the concept.
        shard: Option<ShardId>,
    },
    RemoveCollection {
        collection_id: Uuid,
    },
    /// A client's data batch, proposed as a single log entry.
    ///
    /// One batch is one entry on purpose: the batch is all-or-nothing for the
    /// caller, and splitting it would turn a validation failure into a partial
    /// write. The entry index becomes the records' sequence number, so the log
    /// index *is* the dataset sequence.
    Batch {
        collection_id: Uuid,
        ops: Vec<rekha_core::op::Operation>,
        /// Effective vector width after inference on the leader. A follower
        /// whose collection has not fixed its width adopts this one, so every
        /// node builds the index at the same width.
        dimension: usize,
    },
    // RBAC
    CreateUser {
        user: UserRecord,
        key_hash: [u8; 32],
        key_label: String,
    },
    /// The bootstrap superadmin, as one entry.
    ///
    /// Separate from [`Self::CreateUser`] because it also grants the role, and
    /// it has to happen in the same entry: a user that exists but holds no
    /// permissions is a window of `403`s for every request that arrives between
    /// the two entries.
    BootstrapSuperadmin {
        user: UserRecord,
        key_hash: [u8; 32],
        key_label: String,
        role_id: String,
    },
    DeleteUser {
        user_id: String,
    },
    SetUserEnabled {
        user_id: String,
        enabled: bool,
    },
    CreateApiKey {
        user_id: String,
        key_hash: [u8; 32],
        label: String,
    },
    DeleteApiKey {
        user_id: String,
        key_hash: [u8; 32],
    },
    CreateRole {
        role: RoleRecord,
    },
    DeleteRole {
        role_id: String,
    },
    AddPermission {
        role_id: String,
        permission: Permission,
    },
    RemovePermission {
        role_id: String,
        perm_id: String,
    },
    AssignRole {
        user_id: String,
        role_id: String,
        assigned_by: String,
    },
    UnassignRole {
        user_id: String,
        role_id: String,
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

/// What the state machine reports back for a committed data batch.
///
/// `R = String` in the Raft type config, so this is the JSON payload of that
/// string. It exists because an apply failure must not be invisible to the
/// client: openraft requires `apply` to return `Ok` even when the application
/// could not apply the entry, so without a field like this a write that was
/// committed and never stored would still answer 2xx.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResponse {
    /// Operations applied.
    pub applied: usize,
    /// Log index the batch committed at.
    pub seq: u64,
    /// Why this node could not apply it. `None` on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl BatchResponse {
    pub fn ok(applied: usize, seq: u64) -> Self {
        Self {
            applied,
            seq,
            error: None,
        }
    }

    pub fn failed(seq: u64, error: String) -> Self {
        Self {
            applied: 0,
            seq,
            error: Some(error),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: u64,
    /// Address of the node's internal (cluster) listener.
    pub addr: String,
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
                config: rekha_core::config::CollectionConfig::new(
                    "test".into(),
                    128,
                    rekha_core::types::Distance::L2,
                ),
                shard: Some(ShardId(4)),
            },
            ClusterOperation::RemoveCollection {
                collection_id: Uuid::new_v4(),
            },
        ];
        for op in &ops {
            let bytes = bincode::serialize(op).unwrap();
            let decoded: ClusterOperation = bincode::deserialize(&bytes).unwrap();
            assert_eq!(*op, decoded);
        }
    }
}
