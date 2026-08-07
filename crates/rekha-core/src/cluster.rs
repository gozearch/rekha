//! Replication and fault-tolerance primitives.
//!
//! Design: **the WAL is the source of truth; indexes are derived.** Replicas
//! track their position in the log and replay it to rebuild derived state.
//!
//! - [`Epoch`] is a fencing token that increases on every leader change. A
//!   write tagged with an epoch below the current leader epoch is rejected, so
//!   a stale leader can never overwrite a newer one (classic fencing / Fencing
//!   Token pattern).
//! - [`ClockTag`] is a Qdrant-style `(node, seq)` ordering tag: within a node
//!   the `seq` counter is monotonic, and node ids break ties, so every
//!   operation gets a globally unique, totally ordered tag.

use serde::{Deserialize, Serialize};

/// Fencing token identifying the current leader generation. Increases on every
/// leader change; stale leaders carrying an older epoch cannot write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Epoch(pub u64);

/// Qdrant-style ordering tag. `(node, seq)` is globally unique: `seq` is
/// monotonic per node and `node` breaks ties across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClockTag {
    /// Owning node id.
    pub node: u64,
    /// Per-node monotonic sequence number.
    pub seq: u64,
}

/// Membership/health state of a replica as observed by the coordinator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicaState {
    /// Joining the cluster; still catching up on the log.
    Initializing,
    /// Fully caught up and serving reads/writes.
    Active,
    /// Lagging or missing a fraction of the log; serving reads only.
    Partial,
    /// Unreachable or permanently failed.
    Dead,
}

/// Description of one replica's membership and log position. The WAL being the
/// source of truth means a replica's usefulness is measured by how far through
/// the log it has replayed (`log_offset`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaInfo {
    /// Node id, matching the `node` field of a [`ClockTag`].
    pub node: u64,
    /// Observed membership/health state.
    pub state: ReplicaState,
    /// Highest WAL offset this replica has replayed (inclusive).
    pub log_offset: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_tag_ordering() {
        assert!(ClockTag { node: 1, seq: 0 } < ClockTag { node: 1, seq: 1 });
        assert!(
            ClockTag {
                node: 1,
                seq: u64::MAX
            } < ClockTag { node: 2, seq: 0 }
        );
        assert_eq!(ClockTag { node: 3, seq: 7 }, ClockTag { node: 3, seq: 7 });
        assert_ne!(ClockTag { node: 3, seq: 7 }, ClockTag { node: 3, seq: 8 });
    }

    #[test]
    fn epoch_fencing_comparison() {
        assert!(Epoch(0) < Epoch(1));
        assert!(Epoch(5) > Epoch(4));
        assert_eq!(Epoch(2), Epoch(2));
        // The derived Ord is total: leader election picks the max epoch.
        let current = [Epoch(3), Epoch(1), Epoch(9), Epoch(2)]
            .into_iter()
            .max()
            .unwrap();
        assert_eq!(current, Epoch(9));
    }
}
