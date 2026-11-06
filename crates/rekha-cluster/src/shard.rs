//! The set of consensus groups this process participates in.
//!
//! One process runs one `Raft` instance per group it votes on: one metadata
//! group and one per shard it replicates. [`ShardCluster`] is the registry that
//! maps a shard to its instance and answers the two questions the request path
//! asks — "do I host this shard?" and "who leads it?" — without a metadata
//! round trip.
//!
//! Leadership is read from `raft.metrics()` at the moment it is needed. There
//! is deliberately no cached leader here: a group's term changes independently
//! of the metadata group's, so a shared cache would be stale for exactly the
//! groups that are in the middle of an election.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use openraft::Raft;
use uuid::Uuid;

use crate::network::GroupId;
use crate::placement::{Placement, ShardId};
use crate::raft_types::RaftTypeConfig;

/// The consensus groups this node runs.
pub struct ShardCluster {
    placement: parking_lot::RwLock<Arc<Placement>>,
    /// Only groups this node votes on. A node that is not an owner of a shard
    /// has no instance for it, which is what makes "not my data" structural
    /// rather than a check that could be forgotten.
    groups: parking_lot::RwLock<HashMap<ShardId, Arc<Raft<RaftTypeConfig>>>>,
    /// (Collection, Shard) pairs this process has already ensured exist in
    /// their shard group's log.
    ///
    /// In memory on purpose: losing it costs one idempotent `AddCollection`
    /// entry per collection per shard after a restart, and a persisted copy
    /// would be a second source of truth about what a log contains.
    ensured: parking_lot::Mutex<HashSet<(Uuid, ShardId)>>,
    /// 2PC transaction IDs that have committed on the metadata consensus group.
    committed_txs: parking_lot::RwLock<HashSet<Uuid>>,
    /// In-flight 2PC transactions staged on shards: tx_id -> (shard, timestamp).
    staged_txs: parking_lot::RwLock<HashMap<Uuid, (ShardId, std::time::Instant)>>,
}

impl std::fmt::Debug for ShardCluster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardCluster")
            .field("shards", &self.shards())
            .finish()
    }
}

impl ShardCluster {
    /// A registry with no groups yet.
    pub fn new(placement: Arc<Placement>) -> Self {
        Self {
            placement: parking_lot::RwLock::new(placement),
            groups: parking_lot::RwLock::new(HashMap::new()),
            ensured: parking_lot::Mutex::new(HashSet::new()),
            committed_txs: parking_lot::RwLock::new(HashSet::new()),
            staged_txs: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Register an instance for `shard`.
    pub fn insert(&self, shard: ShardId, raft: Arc<Raft<RaftTypeConfig>>) {
        self.groups.write().insert(shard, raft);
    }

    /// Remove an instance for `shard`.
    pub fn remove(&self, shard: ShardId) -> Option<Arc<Raft<RaftTypeConfig>>> {
        self.groups.write().remove(&shard)
    }

    /// The placement this cluster runs by.
    pub fn placement(&self) -> Arc<Placement> {
        self.placement.read().clone()
    }

    /// Update the placement dynamically.
    pub fn set_placement(&self, placement: Arc<Placement>) {
        *self.placement.write() = placement;
    }

    /// The owners of `shard`.
    pub fn owners(&self, shard: ShardId) -> Vec<u64> {
        self.placement.read().owners(shard).to_vec()
    }

    /// The group id used on the wire for `shard`.
    pub fn group_id(shard: ShardId) -> GroupId {
        // 0 is the metadata group; shard 0 is group 1.
        shard.0 + 1
    }

    /// Whether this node runs a consensus group for `shard`.
    pub fn hosts(&self, shard: ShardId) -> bool {
        self.groups.read().contains_key(&shard)
    }

    /// This node's `Raft` instance for `shard`, if it hosts one.
    pub fn raft(&self, shard: ShardId) -> Option<Arc<Raft<RaftTypeConfig>>> {
        self.groups.read().get(&shard).cloned()
    }

    /// Shards this node hosts, ascending, for metrics and status.
    pub fn shards(&self) -> Vec<ShardId> {
        let mut shards: Vec<ShardId> = self.groups.read().keys().copied().collect();
        shards.sort_unstable();
        shards
    }

    /// The elected leader of `shard`, read from that group's own metrics.
    ///
    /// `None` while there is no leader — during an election, or after quorum
    /// loss. Callers must treat that as "retry later", never as "use a default
    /// node".
    pub fn leader(&self, shard: ShardId) -> Option<u64> {
        let raft = self.groups.read().get(&shard)?.clone();
        let metrics = raft.metrics();
        let m = metrics.borrow();
        m.current_leader
    }

    /// Whether the shard group for `collection` has already been told the
    /// collection exists.
    pub fn is_ensured(&self, collection: Uuid, shard: ShardId) -> bool {
        self.ensured.lock().contains(&(collection, shard))
    }

    /// Record that a shard group now knows about `collection`.
    pub fn mark_ensured(&self, collection: Uuid, shard: ShardId) {
        self.ensured.lock().insert((collection, shard));
    }

    /// Check whether a 2PC transaction has been committed globally.
    pub fn is_tx_committed(&self, tx_id: &Uuid) -> bool {
        self.committed_txs.read().contains(tx_id)
    }

    /// Mark a 2PC transaction as committed globally.
    pub fn mark_tx_committed(&self, tx_id: Uuid) {
        self.committed_txs.write().insert(tx_id);
    }

    /// Mark a 2PC transaction as aborted globally.
    pub fn mark_tx_aborted(&self, tx_id: &Uuid) {
        self.committed_txs.write().remove(tx_id);
    }

    /// Register an in-flight prepared transaction on a shard.
    pub fn mark_tx_staged(&self, shard: ShardId, tx_id: Uuid) {
        self.staged_txs
            .write()
            .insert(tx_id, (shard, std::time::Instant::now()));
    }

    /// Mark an in-flight transaction resolved on a shard.
    pub fn mark_tx_resolved(&self, tx_id: &Uuid) {
        self.staged_txs.write().remove(tx_id);
    }

    /// Collect any in-flight transactions staged longer than `max_age`.
    pub fn stale_staged_txs(&self, max_age: std::time::Duration) -> Vec<(Uuid, ShardId)> {
        let now = std::time::Instant::now();
        self.staged_txs
            .read()
            .iter()
            .filter(|(_, (_, created))| now.duration_since(*created) >= max_age)
            .map(|(tx_id, (shard, _))| (*tx_id, *shard))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_ids_do_not_collide_with_the_metadata_group() {
        assert_eq!(ShardCluster::group_id(ShardId(0)), 1);
        assert_eq!(ShardCluster::group_id(ShardId(255)), 256);
    }

    #[test]
    fn a_node_hosts_only_its_own_shards() {
        let placement = Arc::new(Placement::new(8, 3, [1, 2, 3, 4]).expect("placement"));
        let cluster = ShardCluster::new(placement.clone());
        let hosted: Vec<ShardId> = (0..8)
            .map(ShardId)
            .filter(|s| placement.owns(*s, 1))
            .collect();
        assert!(
            hosted.len() < 8,
            "with R=3 on 4 nodes a node must not own every shard"
        );
        // `insert` needs a Raft instance, so assert the predicate the registry
        // is built from: `hosts` is false until a group is registered.
        for shard in &hosted {
            assert!(!cluster.hosts(*shard));
            assert!(placement.owns(*shard, 1));
        }
        assert_eq!(cluster.shards(), Vec::<ShardId>::new());
    }
}
