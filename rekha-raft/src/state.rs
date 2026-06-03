use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The state that is replicated via Raft across all nodes in a partition.
///
/// Each Raft group owns a single vector shard. The replicated state includes:
/// - Vector data (full precision)
/// - PQ codes
/// - Payloads
/// - Index metadata
///
/// For simplicity, this is an in-memory replica that is persisted to RocksDB.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicatedState {
    /// Partition (vector shard) ID this state belongs to.
    pub partition_id: u64,
    /// Current Raft term.
    pub current_term: u64,
    /// Voted for in current term.
    pub voted_for: Option<String>,
    /// Last committed index in the Raft log.
    pub commit_index: u64,
    /// Last applied index in the Raft log.
    pub last_applied: u64,
    /// Vectors stored in this partition: map<ID, byte-encoded data>.
    pub vectors: HashMap<u64, Vec<u8>>,
    /// Payloads stored in this partition.
    pub payloads: HashMap<u64, Vec<u8>>,
}

impl ReplicatedState {
    /// Create a new, empty replicated state for a partition.
    pub fn new(partition_id: u64) -> Self {
        Self {
            partition_id,
            current_term: 0,
            voted_for: None,
            commit_index: 0,
            last_applied: 0,
            vectors: HashMap::new(),
            payloads: HashMap::new(),
        }
    }

    /// Apply an insert command to the state.
    pub fn apply_insert(&mut self, id: u64, vector: Vec<f32>, payload: Option<Vec<u8>>) {
        let vec_bytes = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.vectors.insert(id, vec_bytes);
        if let Some(p) = payload {
            self.payloads.insert(id, p);
        }
    }

    /// Apply a delete command to the state.
    pub fn apply_delete(&mut self, ids: &[u64]) -> u64 {
        let mut count = 0;
        for id in ids {
            if self.vectors.remove(id).is_some() {
                count += 1;
            }
            self.payloads.remove(id);
        }
        count
    }

    /// Get a vector by ID.
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.vectors.get(&id).map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
    }

    /// Get a payload by ID.
    pub fn get_payload(&self, id: u64) -> Option<&[u8]> {
        self.payloads.get(&id).map(|v| v.as_slice())
    }

    /// Number of vectors in this partition.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get all vector IDs in this partition.
    pub fn all_ids(&self) -> Vec<u64> {
        self.vectors.keys().copied().collect()
    }
}

/// A Raft log entry command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftCommand {
    Insert {
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    },
    Delete {
        ids: Vec<u64>,
    },
    NoOp,
}

impl RaftCommand {
    /// Apply this command to the replicated state.
    pub fn apply(&self, state: &mut ReplicatedState) {
        match self {
            Self::Insert {
                id,
                vector,
                payload,
            } => {
                state.apply_insert(*id, vector.clone(), payload.clone());
            }
            Self::Delete { ids } => {
                state.apply_delete(ids);
            }
            Self::NoOp => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replicated_state_new() {
        let state = ReplicatedState::new(0);
        assert_eq!(state.partition_id, 0);
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_replicated_state_apply_insert() {
        let mut state = ReplicatedState::new(1);
        state.apply_insert(42, vec![1.0, 2.0, 3.0], None);
        assert_eq!(state.len(), 1);
        let v = state.get_vector(42).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_replicated_state_apply_insert_with_payload() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(10, vec![0.5], Some(b"data".to_vec()));
        assert_eq!(state.len(), 1);
        let payload = state.get_payload(10).unwrap();
        assert_eq!(payload, b"data");
    }

    #[test]
    fn test_replicated_state_get_payload_nonexistent() {
        let state = ReplicatedState::new(0);
        assert!(state.get_payload(999).is_none());
    }

    #[test]
    fn test_replicated_state_apply_delete() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(1, vec![1.0], None);
        state.apply_insert(2, vec![2.0], None);
        assert_eq!(state.len(), 2);

        let deleted = state.apply_delete(&[1]);
        assert_eq!(deleted, 1);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_replicated_state_delete_nonexistent() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(1, vec![1.0], None);
        let deleted = state.apply_delete(&[999]);
        assert_eq!(deleted, 0);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_replicated_state_all_ids() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(1, vec![1.0], None);
        state.apply_insert(2, vec![2.0], None);
        state.apply_insert(3, vec![3.0], None);
        let mut ids = state.all_ids();
        ids.sort();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_replicated_state_get_vector_nonexistent() {
        let state = ReplicatedState::new(0);
        assert!(state.get_vector(999).is_none());
    }

    #[test]
    fn test_raft_command_noop_apply() {
        let mut state = ReplicatedState::new(0);
        RaftCommand::NoOp.apply(&mut state);
        assert!(state.is_empty());
    }

    #[test]
    fn test_raft_command_insert_apply() {
        let mut state = ReplicatedState::new(0);
        let cmd = RaftCommand::Insert {
            id: 7,
            vector: vec![0.1, 0.2],
            payload: None,
        };
        cmd.apply(&mut state);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_raft_command_delete_apply() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(5, vec![5.0], None);
        let cmd = RaftCommand::Delete { ids: vec![5] };
        cmd.apply(&mut state);
        assert!(state.is_empty());
    }

    #[test]
    fn test_replicated_state_delete_with_payload() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert(1, vec![1.0], Some(b"p1".to_vec()));
        state.apply_delete(&[1]);
        assert_eq!(state.len(), 0);
        assert!(state.get_payload(1).is_none());
    }
}
