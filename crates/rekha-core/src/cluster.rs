//! Replication and fault-tolerance primitives.
//!
//! Design: **the WAL is the source of truth; indexes are derived.** Replicas
//! track their position in the log and replay it to rebuild derived state.
//!
//! - [`Epoch`] is a fencing token that increases on every leader change. A
//!   write tagged with an epoch below the current leader epoch is rejected, so
//!   a stale leader can never overwrite a newer one (classic fencing / Fencing
//!   Token pattern).

use serde::{Deserialize, Serialize};

/// Identifier of one placement shard.
///
/// A shard is the unit of replication: one Raft group and one set of replicas.
/// Shards are numbered `0..shard_count` for the life of the cluster. They are
/// not collections; several collections share one shard.
///
/// This lives in `rekha-core` rather than the cluster crate because a
/// collection's catalog row records the shard it was placed on, and the
/// catalog cannot depend on the consensus layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Fencing token identifying the current leader generation. Increases on every
/// leader change; stale leaders carrying an older epoch cannot write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Epoch(pub u64);

#[cfg(test)]
mod tests {
    use super::*;

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
