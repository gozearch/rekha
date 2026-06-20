use rekha_core::{CollectionMetadata, UserConfig};
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
    /// For data Raft groups, this stores all vectors (single-collection mode).
    pub vectors: HashMap<u64, Vec<u8>>,
    /// Payloads stored in this partition.
    /// For data Raft groups, this stores all payloads (single-collection mode).
    pub payloads: HashMap<u64, Vec<u8>>,
    /// Collection registry (metadata Raft group only).
    #[serde(default)]
    pub collections: HashMap<String, CollectionMetadata>,
    /// Per-collection vector data (data Raft groups, multi-collection mode).
    #[serde(default)]
    pub collections_data: HashMap<String, HashMap<u64, Vec<u8>>>,
    /// Per-collection payload data (data Raft groups, multi-collection mode).
    #[serde(default)]
    pub collections_payloads: HashMap<String, HashMap<u64, Vec<u8>>>,
    /// Peer list for this Raft group (set via MembershipChange).
    #[serde(default)]
    pub peers: Vec<String>,
    /// User registry (metadata Raft group only).
    #[serde(default)]
    pub users: HashMap<String, UserConfig>,
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
            collections: HashMap::new(),
            collections_data: HashMap::new(),
            collections_payloads: HashMap::new(),
            peers: Vec::new(),
            users: HashMap::new(),
        }
    }

    /// Apply an insert command to the state.
    /// Routes to the appropriate collection within this partition.
    pub fn apply_insert(
        &mut self,
        collection_name: &str,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    ) {
        let vec_bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Store in legacy flat map (backward compat for single-collection tests)
        self.vectors.insert(id, vec_bytes.clone());
        if let Some(ref p) = payload {
            self.payloads.insert(id, p.clone());
        }
        // Store in per-collection map
        self.collections_data
            .entry(collection_name.to_string())
            .or_default()
            .insert(id, vec_bytes);
        if let Some(p) = payload {
            self.collections_payloads
                .entry(collection_name.to_string())
                .or_default()
                .insert(id, p);
        }
    }

    /// Apply a delete command to the state.
    pub fn apply_delete(&mut self, collection_name: &str, ids: &[u64]) -> u64 {
        let mut count = 0;
        for id in ids {
            let existed = self.vectors.remove(id).is_some();
            self.payloads.remove(id);
            if let Some(data) = self.collections_data.get_mut(collection_name) {
                data.remove(id);
            }
            if let Some(payloads) = self.collections_payloads.get_mut(collection_name) {
                payloads.remove(id);
            }
            if existed {
                count += 1;
            }
        }
        count
    }

    /// Apply a create-collection command to the state.
    pub fn apply_create_collection(
        &mut self,
        name: String,
        dim: usize,
        config: rekha_core::CollectionConfig,
    ) {
        let meta = CollectionMetadata {
            name: name.clone(),
            dim,
            config,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        self.collections.insert(name, meta);
    }

    /// Apply a drop-collection command to the state.
    pub fn apply_drop_collection(&mut self, name: &str) {
        self.collections.remove(name);
        self.collections_data.remove(name);
        self.collections_payloads.remove(name);
    }

    /// Apply a membership change command to the state.
    pub fn apply_membership_change(&mut self, new_peers: Vec<String>) {
        self.peers = new_peers;
    }

    /// Apply a create-user command to the state.
    pub fn apply_create_user(&mut self, username: String, config: UserConfig) {
        self.users.insert(username, config);
    }

    /// Apply a drop-user command to the state.
    pub fn apply_drop_user(&mut self, username: &str) {
        self.users.remove(username);
    }

    /// Apply an update-user command to the state.
    pub fn apply_update_user(&mut self, username: &str, config: UserConfig) {
        self.users.insert(username.to_string(), config);
    }

    /// Get a vector by ID (from legacy flat map).
    pub fn get_vector(&self, id: u64) -> Option<Vec<f32>> {
        self.vectors.get(&id).map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        })
    }

    /// Get a vector by ID within a specific collection.
    pub fn get_vector_in_collection(&self, collection_name: &str, id: u64) -> Option<Vec<f32>> {
        self.collections_data
            .get(collection_name)?
            .get(&id)
            .map(|bytes| {
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

    /// Number of vectors in this partition (from legacy flat map).
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

    /// Get all vector IDs in a specific collection.
    pub fn all_ids_in_collection(&self, collection_name: &str) -> Vec<u64> {
        self.collections_data
            .get(collection_name)
            .map(|m| m.keys().copied().collect())
            .unwrap_or_default()
    }
}

/// A Raft log entry command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RaftCommand {
    /// Insert a vector into a collection.
    Insert {
        collection_name: String,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    },
    /// Delete vectors from a collection.
    Delete {
        collection_name: String,
        ids: Vec<u64>,
    },
    /// Create a new collection (metadata Raft group).
    CreateCollection {
        name: String,
        dim: usize,
        config: rekha_core::CollectionConfig,
    },
    /// Drop a collection (metadata Raft group).
    DropCollection { name: String },
    /// Membership change (update peer list).
    MembershipChange {
        new_peers: Vec<String>,
    },
    /// Create a user (metadata Raft group).
    CreateUser {
        username: String,
        config: UserConfig,
    },
    /// Drop a user (metadata Raft group).
    DropUser {
        username: String,
    },
    /// Update a user's config (metadata Raft group).
    UpdateUser {
        username: String,
        config: UserConfig,
    },
    /// No-op (heartbeat, linearizable read).
    NoOp,
}

impl RaftCommand {
    /// Apply this command to the replicated state.
    pub fn apply(&self, state: &mut ReplicatedState) {
        match self {
            Self::Insert {
                collection_name,
                id,
                vector,
                payload,
            } => {
                state.apply_insert(collection_name, *id, vector.clone(), payload.clone());
            }
            Self::Delete {
                collection_name,
                ids,
            } => {
                state.apply_delete(collection_name, ids);
            }
            Self::CreateCollection { name, dim, config } => {
                state.apply_create_collection(name.clone(), *dim, config.clone());
            }
            Self::DropCollection { name } => {
                state.apply_drop_collection(name);
            }
            Self::MembershipChange { new_peers } => {
                state.apply_membership_change(new_peers.clone());
            }
            Self::CreateUser { username, config } => {
                state.apply_create_user(username.clone(), config.clone());
            }
            Self::DropUser { username } => {
                state.apply_drop_user(username);
            }
            Self::UpdateUser { username, config } => {
                state.apply_update_user(username, config.clone());
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
        state.apply_insert("default", 42, vec![1.0, 2.0, 3.0], None);
        assert_eq!(state.len(), 1);
        let v = state.get_vector(42).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_replicated_state_apply_insert_with_payload() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("default", 10, vec![0.5], Some(b"data".to_vec()));
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
        state.apply_insert("default", 1, vec![1.0], None);
        state.apply_insert("default", 2, vec![2.0], None);
        assert_eq!(state.len(), 2);

        let deleted = state.apply_delete("default", &[1]);
        assert_eq!(deleted, 1);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_replicated_state_delete_nonexistent() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("default", 1, vec![1.0], None);
        let deleted = state.apply_delete("default", &[999]);
        assert_eq!(deleted, 0);
        assert_eq!(state.len(), 1);
    }

    #[test]
    fn test_replicated_state_all_ids() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("default", 1, vec![1.0], None);
        state.apply_insert("default", 2, vec![2.0], None);
        state.apply_insert("default", 3, vec![3.0], None);
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
            collection_name: "default".to_string(),
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
        state.apply_insert("default", 5, vec![5.0], None);
        let cmd = RaftCommand::Delete {
            collection_name: "default".to_string(),
            ids: vec![5],
        };
        cmd.apply(&mut state);
        assert!(state.is_empty());
    }

    #[test]
    fn test_replicated_state_delete_with_payload() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("default", 1, vec![1.0], Some(b"p1".to_vec()));
        state.apply_delete("default", &[1]);
        assert_eq!(state.len(), 0);
        assert!(state.get_payload(1).is_none());
    }

    #[test]
    fn test_replicated_state_create_collection() {
        let mut state = ReplicatedState::new(0);
        let config = rekha_core::CollectionConfig::default();
        state.apply_create_collection("test".into(), 128, config.clone());
        assert!(state.collections.contains_key("test"));
        assert_eq!(state.collections["test"].dim, 128);
    }

    #[test]
    fn test_replicated_state_drop_collection() {
        let mut state = ReplicatedState::new(0);
        let config = rekha_core::CollectionConfig::default();
        state.apply_create_collection("test".into(), 128, config);
        assert!(state.collections.contains_key("test"));
        state.apply_drop_collection("test");
        assert!(!state.collections.contains_key("test"));
    }

    #[test]
    fn test_get_vector_in_collection() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("images", 42, vec![1.0, 2.0], None);
        let v = state.get_vector_in_collection("images", 42).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!(state.get_vector_in_collection("audio", 42).is_none());
    }

    #[test]
    fn test_all_ids_in_collection() {
        let mut state = ReplicatedState::new(0);
        state.apply_insert("images", 1, vec![1.0], None);
        state.apply_insert("images", 2, vec![2.0], None);
        state.apply_insert("audio", 10, vec![10.0], None);
        let img_ids = state.all_ids_in_collection("images");
        assert_eq!(img_ids.len(), 2);
        let aud_ids = state.all_ids_in_collection("audio");
        assert_eq!(aud_ids.len(), 1);
        let missing = state.all_ids_in_collection("nonexistent");
        assert!(missing.is_empty());
    }
}
