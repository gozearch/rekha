//! Leadership, as observed from consensus — never configured.
//!
//! Every component that needs to know who leads reads it from here, and this
//! module reads it from `raft.metrics()`. There is deliberately no other
//! source: a cached leader in application state, a `--leader` flag, or a
//! replicated `is_leader` boolean are all the same bug in different clothes,
//! because they let a node act on a fact the consensus layer has already
//! revised.
//!
//! Two consequences are load-bearing:
//!
//! - [`ClusterView::leader`] is an `Option`. During an election there is no
//!   leader, and callers must treat that as "retry later", not as "use a
//!   default address".
//! - [`LeaderInfo::addr`] is an `Option` too. A leader can be elected before
//!   this node's peer directory knows where it lives; that is reported
//!   honestly rather than papered over with a fallback.

use std::sync::Arc;

use openraft::{Raft, ServerState};
use tokio::sync::watch;

use crate::identity::PeerDirectory;
use crate::raft_types::RaftTypeConfig;

type RaftHandle = Arc<Raft<RaftTypeConfig>>;

/// The Raft role of this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    Learner,
    Follower,
    Candidate,
    Leader,
    Shutdown,
}

impl From<ServerState> for RaftRole {
    fn from(state: ServerState) -> Self {
        match state {
            ServerState::Learner => Self::Learner,
            ServerState::Follower => Self::Follower,
            ServerState::Candidate => Self::Candidate,
            ServerState::Leader => Self::Leader,
            ServerState::Shutdown => Self::Shutdown,
        }
    }
}

impl RaftRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Learner => "learner",
            Self::Follower => "follower",
            Self::Candidate => "candidate",
            Self::Leader => "leader",
            Self::Shutdown => "shutdown",
        }
    }
}

/// The elected leader, its term, and where to reach it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaderInfo {
    pub node_id: u64,
    /// The term in which this node was elected. A change in term invalidates
    /// any cached routing decision.
    pub term: u64,
    /// Base URL, when the peer directory knows it.
    pub addr: Option<String>,
}

/// An immutable snapshot of cluster leadership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterView {
    /// This node's id.
    pub node_id: u64,
    /// `None` while there is no leader — during an election, or after quorum loss.
    pub leader: Option<LeaderInfo>,
    /// This node's role.
    pub state: RaftRole,
    /// The current Raft term.
    pub term: u64,
    /// Voters in the effective membership, ascending.
    pub voters: Vec<u64>,
    /// Highest applied log index, when anything has been applied.
    pub applied_index: Option<u64>,
    /// Milliseconds since a quorum acknowledged this leader. Large values mean
    /// this node is isolated even if it still believes it leads.
    pub millis_since_quorum_ack: Option<u64>,
}

impl ClusterView {
    /// Build a view from raw consensus fields.
    ///
    /// Split out from the metrics plumbing so the mapping is testable without
    /// constructing an openraft `RaftMetrics`.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        node_id: u64,
        term: u64,
        state: RaftRole,
        current_leader: Option<u64>,
        voters: Vec<u64>,
        applied_index: Option<u64>,
        millis_since_quorum_ack: Option<u64>,
        peers: &PeerDirectory,
    ) -> Self {
        let leader = current_leader.map(|leader_id| LeaderInfo {
            node_id: leader_id,
            term,
            addr: peers.url(leader_id),
        });
        Self {
            node_id,
            leader,
            state,
            term,
            voters,
            applied_index,
            millis_since_quorum_ack,
        }
    }

    /// Whether this node currently believes it leads.
    ///
    /// Use this for observing and reporting. Do **not** use it to authorize a
    /// write: leadership can move between this read and the write. Writes go
    /// through `Raft::client_write`, which answers with
    /// `ForwardToLeader` — see the API layer.
    pub fn is_leader(&self) -> bool {
        self.state == RaftRole::Leader
    }

    /// The leader's base URL, if a leader exists and its address is known.
    pub fn leader_url(&self) -> Option<&str> {
        self.leader.as_ref().and_then(|l| l.addr.as_deref())
    }
}

/// Publishes [`ClusterView`] updates from the Raft metrics stream.
///
/// Cloning the tracker is cheap; each clone shares the same subscription.
#[derive(Debug, Clone)]
pub struct LeaderTracker {
    view: watch::Receiver<ClusterView>,
}

impl LeaderTracker {
    /// Subscribe to consensus leadership for `raft`.
    ///
    /// The first value is available immediately; it does not wait for a change.
    pub fn spawn(raft: RaftHandle, peers: PeerDirectory) -> Self {
        let mut metrics = raft.metrics();
        let initial = {
            let m = metrics.borrow_and_update();
            view_from_metrics(&m, &peers)
        };
        let (tx, rx) = watch::channel(initial);

        tokio::spawn(async move {
            while metrics.changed().await.is_ok() {
                let next = {
                    let m = metrics.borrow_and_update();
                    view_from_metrics(&m, &peers)
                };
                // A closed receiver means every consumer is gone; stop cleanly
                // rather than spinning the loop.
                if tx.send(next).is_err() {
                    break;
                }
            }
        });

        Self { view: rx }
    }

    /// The most recent view.
    pub fn view(&self) -> ClusterView {
        self.view.borrow().clone()
    }

    /// The current leader, if any.
    pub fn leader(&self) -> Option<LeaderInfo> {
        self.view.borrow().leader.clone()
    }
}

/// This leader's replication progress for `node`.
///
/// `None` means either this node is not the leader or the peer never
/// replicated. Callers treat both as "no evidence of catch-up" (infinite lag),
/// never as zero — promoting on an idle cluster is how a learner that
/// replicated nothing becomes a voter.
pub fn replication_matched<NID, N>(
    metrics: &openraft::RaftMetrics<NID, N>,
    node: &NID,
) -> Option<u64>
where
    NID: openraft::NodeId,
    N: openraft::Node,
{
    metrics
        .replication
        .as_ref()
        .and_then(|r| r.get(node))
        .and_then(|id| id.as_ref().map(|log_id| log_id.index))
}

fn view_from_metrics(m: &openraft::RaftMetrics<u64, u64>, peers: &PeerDirectory) -> ClusterView {
    let voters = m
        .membership_config
        .membership()
        .voter_ids()
        .collect::<Vec<u64>>();
    ClusterView::build(
        m.id,
        m.current_term,
        RaftRole::from(m.state),
        m.current_leader,
        voters,
        m.last_applied.map(|log_id| log_id.index),
        m.millis_since_quorum_ack,
        peers,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::parse_peers;

    fn peers() -> PeerDirectory {
        PeerDirectory::new(&parse_peers("1@10.0.0.1:8000,2@10.0.0.2:8000").unwrap())
    }

    #[test]
    fn a_known_leader_resolves_to_its_address() {
        let view = ClusterView::build(
            1,
            7,
            RaftRole::Follower,
            Some(2),
            vec![1, 2],
            Some(42),
            Some(10),
            &peers(),
        );

        let leader = view.leader.clone().expect("leader present");
        assert_eq!(leader.node_id, 2);
        assert_eq!(leader.term, 7);
        assert_eq!(leader.addr.as_deref(), Some("http://10.0.0.2:8000"));
        assert_eq!(view.leader_url(), Some("http://10.0.0.2:8000"));
        assert!(!view.is_leader());
    }

    /// During an election there is no leader. The type must say so rather than
    /// inventing one.
    #[test]
    fn no_leader_is_represented_as_none() {
        let view = ClusterView::build(
            1,
            3,
            RaftRole::Candidate,
            None,
            vec![1, 2],
            None,
            None,
            &peers(),
        );
        assert!(view.leader.is_none());
        assert_eq!(view.leader_url(), None);
        assert_eq!(view.state, RaftRole::Candidate);
    }

    /// A leader whose address this node does not know is still reported, but
    /// without a fabricated URL.
    #[test]
    fn an_unresolvable_leader_keeps_its_id_but_not_an_address() {
        let view = ClusterView::build(
            1,
            4,
            RaftRole::Follower,
            Some(9),
            vec![1, 9],
            None,
            None,
            &peers(),
        );
        let leader = view.leader.clone().expect("leader present");
        assert_eq!(leader.node_id, 9);
        assert_eq!(leader.addr, None);
        assert_eq!(view.leader_url(), None);
    }

    #[test]
    fn role_mapping_matches_openraft_states() {
        assert_eq!(RaftRole::from(ServerState::Leader), RaftRole::Leader);
        assert_eq!(RaftRole::from(ServerState::Follower), RaftRole::Follower);
        assert_eq!(RaftRole::from(ServerState::Candidate), RaftRole::Candidate);
        assert_eq!(RaftRole::from(ServerState::Learner), RaftRole::Learner);
        assert_eq!(RaftRole::from(ServerState::Shutdown), RaftRole::Shutdown);
        assert_eq!(RaftRole::Leader.as_str(), "leader");
    }
}
