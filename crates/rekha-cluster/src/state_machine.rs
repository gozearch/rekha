//! Cluster metadata state machine — applies Raft-committed entries.

use std::collections::HashMap;
use std::sync::Arc;

use openraft::Entry;
use openraft::EntryPayload;
use openraft::LogId;
use openraft::Snapshot;
use openraft::SnapshotMeta;
use openraft::StorageError;
use openraft::StoredMembership;
use openraft::storage::RaftSnapshotBuilder;
use openraft::storage::RaftStateMachine;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::cluster::ClockMap;
use crate::raft_types::{ClusterOperation, NodeInfo, RaftTypeConfig};

/// Trait for engine snapshot/restore, avoiding circular dependency with rekha-engine.
pub trait EngineSnapshotProvider: Send + Sync + 'static {
    fn snapshot(&self) -> Result<Vec<u8>, String>;
    fn restore_snapshot(&self, data: &[u8]) -> Result<(), String>;
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct ClusterStateMachine {
    pub nodes: HashMap<u64, NodeInfo>,
    pub collections: HashMap<Uuid, Vec<u64>>,
    pub clock_map: ClockMap,
    pub last_applied: Option<LogId<u64>>,
    pub membership: StoredMembership<u64, u64>,
    #[serde(skip)]
    pub engine: Option<Arc<dyn EngineSnapshotProvider>>,
}

impl std::fmt::Debug for ClusterStateMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClusterStateMachine")
            .field("nodes", &self.nodes)
            .field("collections", &self.collections)
            .field("clock_map", &self.clock_map)
            .field("last_applied", &self.last_applied)
            .field("membership", &self.membership)
            .field("engine", &self.engine.as_ref().map(|_| "..."))
            .finish()
    }
}

impl ClusterStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_operation(&mut self, op: &ClusterOperation) {
        match op {
            ClusterOperation::AddNode { node_id, addr } => {
                self.nodes.insert(
                    *node_id,
                    NodeInfo {
                        node_id: *node_id,
                        addr: addr.clone(),
                        is_leader: false,
                    },
                );
            }
            ClusterOperation::RemoveNode { node_id } => {
                self.nodes.remove(node_id);
            }
            ClusterOperation::AddCollection {
                collection_id,
                owner_nodes,
                ..
            } => {
                self.collections.insert(*collection_id, owner_nodes.clone());
            }
            ClusterOperation::RemoveCollection { collection_id } => {
                self.collections.remove(collection_id);
            }
            ClusterOperation::TransferLeadership { new_leader } => {
                for info in self.nodes.values_mut() {
                    info.is_leader = info.node_id == *new_leader;
                }
            }
        }
    }
}

pub struct ClusterSnapshotBuilder {
    sm: ClusterStateMachine,
}

impl RaftSnapshotBuilder<RaftTypeConfig> for ClusterSnapshotBuilder {
    async fn build_snapshot(&mut self) -> Result<Snapshot<RaftTypeConfig>, StorageError<u64>> {
        let cluster_data = bincode::serialize(&self.sm).map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                "cluster snapshot serialize failed: {e}"
            ))),
        })?;

        let engine_data = if let Some(ref engine) = self.sm.engine {
            engine.snapshot().map_err(|e| StorageError::IO {
                source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                    "engine snapshot failed: {e}"
                ))),
            })?
        } else {
            Vec::new()
        };

        let mut combined = Vec::new();
        let cluster_len = cluster_data.len() as u32;
        combined.extend_from_slice(&cluster_len.to_le_bytes());
        combined.extend_from_slice(&cluster_data);
        combined.extend_from_slice(&engine_data);

        let cursor = std::io::Cursor::new(combined);

        let meta = SnapshotMeta {
            last_log_id: self.sm.last_applied,
            last_membership: self.sm.membership.clone(),
            snapshot_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Snapshot {
            meta,
            snapshot: Box::new(cursor),
        })
    }
}

impl RaftStateMachine<RaftTypeConfig> for ClusterStateMachine {
    type SnapshotBuilder = ClusterSnapshotBuilder;

    async fn applied_state(
        &mut self,
    ) -> Result<(Option<LogId<u64>>, StoredMembership<u64, u64>), StorageError<u64>> {
        Ok((self.last_applied, self.membership.clone()))
    }

    async fn apply<I>(&mut self, entries: I) -> Result<Vec<String>, StorageError<u64>>
    where
        I: IntoIterator<Item = Entry<RaftTypeConfig>> + Send,
        I::IntoIter: Send,
    {
        let mut responses = Vec::new();
        for entry in entries {
            self.last_applied = Some(entry.log_id);

            match &entry.payload {
                EntryPayload::Normal(op) => {
                    self.apply_operation(op);
                    responses.push(format!("applied: {:?}", op));
                }
                EntryPayload::Membership(m) => {
                    self.membership = StoredMembership::new(Some(entry.log_id), m.clone());
                    responses.push(format!("membership: {:?}", m));
                }
                EntryPayload::Blank => {
                    responses.push("blank".to_string());
                }
            }
        }
        Ok(responses)
    }

    async fn get_snapshot_builder(&mut self) -> Self::SnapshotBuilder {
        ClusterSnapshotBuilder { sm: self.clone() }
    }

    async fn begin_receiving_snapshot(
        &mut self,
    ) -> Result<Box<std::io::Cursor<Vec<u8>>>, StorageError<u64>> {
        Ok(Box::new(std::io::Cursor::new(Vec::new())))
    }

    async fn install_snapshot(
        &mut self,
        meta: &SnapshotMeta<u64, u64>,
        snapshot: Box<std::io::Cursor<Vec<u8>>>,
    ) -> Result<(), StorageError<u64>> {
        let data = snapshot.into_inner();

        if data.len() < 4 {
            return Ok(());
        }
        let cluster_len = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| StorageError::IO {
                    source: openraft::StorageIOError::read(&std::io::Error::other(
                        "snapshot header too short",
                    )),
                })?,
        ) as usize;

        if data.len() >= 4 + cluster_len {
            let cluster_data = &data[4..4 + cluster_len];
            // Propagate deserialization failures as IO errors so corrupted
            // snapshots are not silently ignored.
            let sm: ClusterStateMachine =
                bincode::deserialize(cluster_data).map_err(|e| StorageError::IO {
                    source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                        "cluster snapshot deserialize failed: {e}"
                    ))),
                })?;
            let engine = self.engine.clone();
            *self = sm;
            self.engine = engine;
        }

        if data.len() > 4 + cluster_len {
            let engine_data = &data[4 + cluster_len..];
            if let Some(ref engine) = self.engine {
                engine.restore_snapshot(engine_data).map_err(|e| StorageError::IO {
                    source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                        "engine restore failed: {e}"
                    ))),
                })?;
            }
        }

        self.last_applied = meta.last_log_id;
        self.membership = meta.last_membership.clone();
        Ok(())
    }

    async fn get_current_snapshot(
        &mut self,
    ) -> Result<Option<Snapshot<RaftTypeConfig>>, StorageError<u64>> {
        let cluster_data = bincode::serialize(self).map_err(|e| StorageError::IO {
            source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                "cluster snapshot serialize failed: {e}"
            ))),
        })?;

        let engine_data = if let Some(ref engine) = self.engine {
            engine.snapshot().map_err(|e| StorageError::IO {
                source: openraft::StorageIOError::read(&std::io::Error::other(format!(
                    "engine snapshot failed: {e}"
                ))),
            })?
        } else {
            Vec::new()
        };

        let mut combined = Vec::new();
        let cluster_len = cluster_data.len() as u32;
        combined.extend_from_slice(&cluster_len.to_le_bytes());
        combined.extend_from_slice(&cluster_data);
        combined.extend_from_slice(&engine_data);

        let cursor = std::io::Cursor::new(combined);

        let meta = SnapshotMeta {
            last_log_id: self.last_applied,
            last_membership: self.membership.clone(),
            snapshot_id: uuid::Uuid::new_v4().to_string(),
        };

        Ok(Some(Snapshot {
            meta,
            snapshot: Box::new(cursor),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use openraft::CommittedLeaderId;

    fn make_log_id(index: u64) -> LogId<u64> {
        LogId::new(CommittedLeaderId::new(1, 0), index)
    }

    #[test]
    fn test_apply_add_node() {
        let mut sm = ClusterStateMachine::new();
        let op = ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        };
        sm.apply_operation(&op);
        assert_eq!(sm.nodes.len(), 1);
        assert_eq!(sm.nodes[&1].addr, "127.0.0.1:8001");
        assert!(!sm.nodes[&1].is_leader);
    }

    #[test]
    fn test_apply_remove_node() {
        let mut sm = ClusterStateMachine::new();
        sm.apply_operation(&ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        });
        sm.apply_operation(&ClusterOperation::RemoveNode { node_id: 1 });
        assert!(sm.nodes.is_empty());
    }

    #[test]
    fn test_apply_add_collection() {
        let mut sm = ClusterStateMachine::new();
        let cid = Uuid::new_v4();
        sm.apply_operation(&ClusterOperation::AddCollection {
            collection_id: cid,
            name: "test".into(),
            dimension: 128,
            distance: "l2".into(),
            tenant: "default".into(),
            database: "default".into(),
            owner_nodes: vec![1, 2],
        });
        assert_eq!(sm.collections[&cid], vec![1, 2]);
    }

    #[test]
    fn test_apply_transfer_leadership() {
        let mut sm = ClusterStateMachine::new();
        sm.apply_operation(&ClusterOperation::AddNode {
            node_id: 1,
            addr: "a".into(),
        });
        sm.apply_operation(&ClusterOperation::AddNode {
            node_id: 2,
            addr: "b".into(),
        });
        sm.apply_operation(&ClusterOperation::TransferLeadership { new_leader: 2 });
        assert!(!sm.nodes[&1].is_leader);
        assert!(sm.nodes[&2].is_leader);
    }

    #[tokio::test]
    async fn test_apply_entries() {
        let mut sm = ClusterStateMachine::new();

        let op = ClusterOperation::AddNode {
            node_id: 1,
            addr: "127.0.0.1:8001".into(),
        };

        let entries = vec![Entry {
            log_id: make_log_id(0),
            payload: EntryPayload::Normal(op),
        }];

        let responses = sm.apply(entries).await.unwrap();
        assert_eq!(responses.len(), 1);
        assert!(responses[0].contains("applied"));
        assert_eq!(sm.nodes.len(), 1);
        assert_eq!(sm.last_applied, Some(make_log_id(0)));
    }

    #[tokio::test]
    async fn test_applied_state() {
        let mut sm = ClusterStateMachine::new();
        let (last_applied, _membership) = sm.applied_state().await.unwrap();
        assert_eq!(last_applied, None);
    }
}
