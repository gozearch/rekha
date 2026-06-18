use async_trait::async_trait;
use rekha_core::RekhaError;

use crate::node::RaftLogEntry;

/// Abstraction for sending Raft RPCs to peer nodes.
///
/// The server layer provides a gRPC implementation of this trait.
/// Tests provide a mock implementation for deterministic behavior.
#[async_trait]
#[allow(clippy::too_many_arguments)]
pub trait RaftPeerNetwork: Send + Sync {
    /// Send AppendEntries to a peer. Returns (success, current_term).
    async fn append_entries(
        &self,
        peer_id: &str,
        partition_id: u64,
        leader_term: u64,
        leader_id: &str,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<RaftLogEntry>,
        leader_commit: u64,
    ) -> Result<(bool, u64), RekhaError>;

    /// Send RequestVote to a peer. Returns (vote_granted, current_term).
    async fn request_vote(
        &self,
        peer_id: &str,
        partition_id: u64,
        term: u64,
        candidate_id: &str,
        last_log_index: u64,
        last_log_term: u64,
    ) -> Result<(bool, u64), RekhaError>;
}
