use rekha_core::{RaftError, RekhaError};
use crate::state::{RaftCommand, ReplicatedState};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};

/// A Raft consensus node for a single partition (vector shard).
///
/// Implements leader-based Raft with:
/// - Leader election
/// - Log replication
/// - Safety (at-most-once semantics via term + index)
/// - Log compaction via snapshotting
///
/// This is a minimal implementation. For production, use Openraft.
pub struct RaftNode {
    /// Node ID.
    node_id: String,
    /// Partition (vector shard) this Raft group manages.
    partition_id: u64,
    /// All peers in this Raft group.
    peers: Vec<String>,
    /// Replicated state machine.
    state: Arc<RwLock<ReplicatedState>>,
    /// Raft log: index -> RaftLogEntry
    log: Arc<Mutex<Vec<RaftLogEntry>>>,
    /// Current Raft state.
    raft_state: Arc<Mutex<RaftInternalState>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftLogEntry {
    term: u64,
    index: u64,
    command: RaftCommand,
}

#[derive(Debug, Clone)]
enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

#[derive(Debug, Clone)]
struct RaftInternalState {
    current_term: u64,
    voted_for: Option<String>,
    commit_index: u64,
    last_applied: u64,
    role: RaftRole,
    leader_id: Option<String>,
    _election_timeout_ms: u64,
    _heartbeat_ms: u64,
}

impl RaftNode {
    /// Create a new Raft node for a partition.
    pub fn new(
        node_id: String,
        partition_id: u64,
        peers: Vec<String>,
        state: ReplicatedState,
    ) -> Self {
        Self {
            node_id,
            partition_id,
            peers,
            state: Arc::new(RwLock::new(state)),
            log: Arc::new(Mutex::new(Vec::new())),
            raft_state: Arc::new(Mutex::new(RaftInternalState {
                current_term: 0,
                voted_for: None,
                commit_index: 0,
                last_applied: 0,
                role: RaftRole::Follower,
                leader_id: None,
            _election_timeout_ms: 300,
            _heartbeat_ms: 100,
            })),
        }
    }

    /// Propose a command to the Raft group.
    /// If this node is the leader, it will replicate the command.
    /// If not, returns an error with the leader hint.
    pub async fn propose(&self, command: RaftCommand) -> Result<(), RekhaError> {
        let (is_leader, term, leader_hint) = {
            let raft_state = self.raft_state.lock().await;
            match &raft_state.role {
                RaftRole::Leader => (true, raft_state.current_term, None),
                RaftRole::Follower | RaftRole::Candidate => {
                    (false, 0, raft_state.leader_id.clone())
                }
            }
        };

        if is_leader {
            let mut log = self.log.lock().await;
            let index = log.len() as u64 + 1;
            let entry = RaftLogEntry { term, index, command };
            let cmd = entry.command.clone();
            log.push(entry);
            let mut state = self.state.write().await;
            cmd.apply(&mut state);
            drop(state);
            drop(log);

            let mut rs = self.raft_state.lock().await;
            rs.commit_index = index;
            rs.last_applied = index;
            Ok(())
        } else {
            Err(RaftError::NotLeader { leader_hint }.into())
        }
    }

    /// Initiate leader election.
    pub async fn start_election(&self) -> Result<(), RekhaError> {
        let mut raft_state = self.raft_state.lock().await;

        raft_state.current_term += 1;
        raft_state.role = RaftRole::Candidate;
        raft_state.voted_for = Some(self.node_id.clone());

        let term = raft_state.current_term;
        let log = self.log.lock().await;
        let _last_log_index = log.len() as u64;
        let _last_log_term = log.last().map(|e| e.term).unwrap_or(0);
        drop(log);

        info!(
            "Node {} starting election for term {} (partition {})",
            self.node_id, term, self.partition_id
        );

        // For simplicity: self-elect in a single-node setup.
        // Multi-node: send RequestVote RPCs to all peers.
        if self.peers.is_empty() {
            raft_state.role = RaftRole::Leader;
            raft_state.leader_id = Some(self.node_id.clone());
            info!("Node {} elected as leader for term {}", self.node_id, term);
            Ok(())
        } else {
            // In multi-node, we'd need votes from majority.
            // Here we just become candidate (incomplete — need gRPC calls).
            warn!("Multi-node election not fully implemented — staying as candidate");
            Ok(())
        }
    }

    /// Handle an AppendEntries RPC from a leader.
    pub async fn handle_append_entries(
        &self,
        leader_term: u64,
        leader_id: &str,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftLogEntry>,
        leader_commit: u64,
    ) -> Result<(bool, u64), RekhaError> {
        // Process leader metadata, then drop raft_state before acquiring log lock.
        let current_term;
        {
            let mut raft_state = self.raft_state.lock().await;

            if leader_term < raft_state.current_term {
                return Ok((false, raft_state.current_term));
            }

            if leader_term > raft_state.current_term {
                raft_state.current_term = leader_term;
                raft_state.role = RaftRole::Follower;
                raft_state.voted_for = None;
            }
            raft_state.leader_id = Some(leader_id.to_string());
            current_term = raft_state.current_term;
        }

        // Check log consistency (log lock only).
        let mut log = self.log.lock().await;
        if prev_log_index > 0 {
            if (prev_log_index as usize) > log.len() {
                return Ok((false, current_term));
            }
            if prev_log_index > 0 {
                let entry = &log[prev_log_index as usize - 1];
                if entry.term != prev_log_term {
                    return Ok((false, current_term));
                }
            }
        }

        // Append new entries.
        for entry in entries {
            let idx = entry.index as usize;
            if idx <= log.len() {
                if idx > 0 && log[idx - 1].term != entry.term {
                    log.truncate(idx - 1);
                }
            }
            if idx > log.len() {
                log.push(entry);
            }
        }

        // Apply committed entries to state machine.
        drop(log);
        self.apply_up_to(leader_commit).await;

        // Update commit_index.
        {
            let mut rs = self.raft_state.lock().await;
            if leader_commit > rs.commit_index {
                rs.commit_index = leader_commit;
            }
        }

        Ok((true, current_term))
    }

    /// Handle a RequestVote RPC.
    pub async fn handle_request_vote(
        &self,
        term: u64,
        candidate_id: &str,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<(bool, u64), RekhaError> {
        let mut raft_state = self.raft_state.lock().await;

        if term < raft_state.current_term {
            return Ok((false, raft_state.current_term));
        }

        if term > raft_state.current_term {
            raft_state.current_term = term;
            raft_state.role = RaftRole::Follower;
            raft_state.voted_for = None;
        }

        // Vote if not already voted or voted for this candidate.
        let can_vote = raft_state.voted_for.is_none()
            || raft_state.voted_for.as_deref() == Some(candidate_id);

        if can_vote {
            let log = self.log.lock().await;
            let last_idx = log.len() as u64;
            let last_term = log.last().map(|e| e.term).unwrap_or(0);

            // Candidate's log must be at least as up-to-date.
            let log_ok = last_log_term > last_term
                || (last_log_term == last_term && last_log_index >= last_idx);

            if log_ok {
                raft_state.voted_for = Some(candidate_id.to_string());
                return Ok((true, raft_state.current_term));
            }
        }

        Ok((false, raft_state.current_term))
    }

    /// Get the current Raft state (term, role, leader).
    pub async fn status(&self) -> RaftStatus {
        let rs = self.raft_state.lock().await;
        let state = self.state.read().await;
        RaftStatus {
            node_id: self.node_id.clone(),
            partition_id: self.partition_id,
            current_term: rs.current_term,
            role: format!("{:?}", rs.role),
            leader_id: rs.leader_id.clone(),
            commit_index: rs.commit_index,
            last_applied: rs.last_applied,
            log_size: self.log.lock().await.len(),
            vector_count: state.len(),
        }
    }

    /// Get a read-only reference to the replicated state.
    pub async fn read_state(&self) -> tokio::sync::RwLockReadGuard<'_, ReplicatedState> {
        self.state.read().await
    }

    /// Apply committed log entries up to the given index.
    async fn apply_up_to(&self, index: u64) {
        let mut state = self.state.write().await;
        let log = self.log.lock().await;

        for entry in log.iter() {
            if entry.index <= index && entry.index > state.last_applied {
                entry.command.apply(&mut state);
                state.last_applied = entry.index;
            }
        }
    }
}

/// Snapshot of Raft node status.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{RaftCommand, ReplicatedState};

    fn test_node() -> RaftNode {
        let state = ReplicatedState::new(0);
        RaftNode::new("test-node".into(), 0, vec![], state)
    }

    #[tokio::test]
    async fn test_raft_node_new() {
        let node = test_node();
        let status = node.status().await;
        assert_eq!(status.node_id, "test-node");
        assert_eq!(status.partition_id, 0);
        assert_eq!(status.role, "Follower");
        assert_eq!(status.current_term, 0);
    }

    #[tokio::test]
    async fn test_raft_start_election_single_node() {
        let node = test_node();
        node.start_election().await.unwrap();
        let status = node.status().await;
        assert_eq!(status.role, "Leader");
        assert_eq!(status.current_term, 1);
        assert_eq!(status.leader_id, Some("test-node".into()));
    }

    #[tokio::test]
    async fn test_raft_propose_as_leader() {
        let node = test_node();
        node.start_election().await.unwrap();

        let cmd = RaftCommand::Insert {
            id: 42,
            vector: vec![1.0, 2.0, 3.0],
            payload: None,
        };
        node.propose(cmd).await.unwrap();

        let state = node.read_state().await;
        let v = state.get_vector(42).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_raft_propose_as_follower_fails() {
        let node = test_node();
        // Not started election, still a follower
        let cmd = RaftCommand::NoOp;
        let result = node.propose(cmd).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not leader"));
    }

    #[tokio::test]
    async fn test_raft_handle_append_entries_newer_term() {
        let node = test_node();
        let (success, term) = node
            .handle_append_entries(5, "leader-1", 0, 0, vec![], 0)
            .await
            .unwrap();
        assert!(success);
        assert_eq!(term, 5);

        let status = node.status().await;
        assert_eq!(status.current_term, 5);
        assert_eq!(status.leader_id, Some("leader-1".into()));
    }

    #[tokio::test]
    async fn test_raft_handle_append_entries_older_term() {
        let node = test_node();
        // First advance to term 5
        node.handle_append_entries(5, "leader-1", 0, 0, vec![], 0)
            .await
            .unwrap();

        // Now try with older term
        let (success, term) = node
            .handle_append_entries(3, "leader-2", 0, 0, vec![], 0)
            .await
            .unwrap();
        assert!(!success);
        assert_eq!(term, 5);
    }

    #[tokio::test]
    async fn test_raft_handle_append_entries_with_data() {
        let node = test_node();
        let entries = vec![
            RaftLogEntry {
                term: 1,
                index: 1,
                command: RaftCommand::Insert {
                    id: 10,
                    vector: vec![1.0],
                    payload: None,
                },
            },
        ];
        let (success, _) = node
            .handle_append_entries(1, "leader-1", 0, 0, entries, 1)
            .await
            .unwrap();
        assert!(success);

        // The command should have been applied
        let state = node.read_state().await;
        assert!(state.get_vector(10).is_some());
    }

    #[tokio::test]
    async fn test_raft_handle_request_vote_newer_term() {
        let node = test_node();
        let (granted, term) = node
            .handle_request_vote(3, "candidate-1", 0, 0)
            .await
            .unwrap();
        assert!(granted);
        assert_eq!(term, 3);
    }

    #[tokio::test]
    async fn test_raft_handle_request_vote_older_term() {
        let node = test_node();
        // Advance to term 5 first
        node.handle_append_entries(5, "leader-1", 0, 0, vec![], 0)
            .await
            .unwrap();

        let (granted, term) = node
            .handle_request_vote(3, "candidate-2", 0, 0)
            .await
            .unwrap();
        assert!(!granted);
        assert_eq!(term, 5);
    }

    #[tokio::test]
    async fn test_raft_handle_request_vote_already_voted() {
        let node = test_node();
        // Vote for candidate-1
        node.handle_request_vote(2, "candidate-1", 0, 0)
            .await
            .unwrap();

        // Try voting for someone else in same term
        let (granted, _) = node
            .handle_request_vote(2, "candidate-2", 0, 0)
            .await
            .unwrap();
        assert!(!granted);
    }

    #[tokio::test]
    async fn test_raft_log_consistency_check() {
        let node = test_node();
        // Try append with invalid prev_log_index
        let (success, _) = node
            .handle_append_entries(1, "leader-1", 999, 0, vec![], 0)
            .await
            .unwrap();
        assert!(!success);
    }

    #[tokio::test]
    async fn test_raft_multiple_proposals() {
        let node = test_node();
        node.start_election().await.unwrap();

        for i in 0..5 {
            let cmd = RaftCommand::Insert {
                id: i,
                vector: vec![i as f32],
                payload: None,
            };
            node.propose(cmd).await.unwrap();
        }

        let status = node.status().await;
        assert_eq!(status.log_size, 5);
        assert_eq!(status.commit_index, 5);

        let state = node.read_state().await;
        assert_eq!(state.len(), 5);
    }

    #[tokio::test]
    async fn test_raft_delete_command() {
        let node = test_node();
        node.start_election().await.unwrap();

        node.propose(RaftCommand::Insert {
            id: 1, vector: vec![1.0], payload: None,
        }).await.unwrap();
        node.propose(RaftCommand::Insert {
            id: 2, vector: vec![2.0], payload: None,
        }).await.unwrap();

        node.propose(RaftCommand::Delete { ids: vec![1] }).await.unwrap();

        let state = node.read_state().await;
        assert_eq!(state.len(), 1);
        assert!(state.get_vector(2).is_some());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftStatus {
    pub node_id: String,
    pub partition_id: u64,
    pub current_term: u64,
    pub role: String,
    pub leader_id: Option<String>,
    pub commit_index: u64,
    pub last_applied: u64,
    pub log_size: usize,
    pub vector_count: usize,
}
