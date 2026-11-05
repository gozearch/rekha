#![deny(clippy::unwrap_used, clippy::expect_used)]

//! RekhaDB cluster types — node identity, topology, the consensus leadership
//! view, and openraft integration.
//!
//! There is no replication module here on purpose. Data reaches followers as
//! committed Raft log entries, applied by [`ClusterStateMachine`]; a separate
//! WAL-shipping path existed and was removed because per-node sequence numbers
//! made a follower's cursor meaningless across an election.

pub mod cluster;
pub mod identity;
pub mod leader;
pub mod log_store;
pub mod network;
pub mod placement;
pub mod raft_types;
pub mod redb_log_store;
pub mod registry;
pub mod shard;
pub mod snapshot_store;
pub mod state_machine;

pub use cluster::{ClockMap, ClusterConfig};
pub use identity::{
    IdentityError, NodeIdentity, Peer, PeerDirectory, PeerParseError, node_id_path, parse_peers,
};
pub use leader::{ClusterView, LeaderInfo, LeaderTracker, RaftRole};
pub use log_store::MemoryLogStore;
pub use network::channel::{ChannelHub, ChannelMessage, ChannelNetwork, ChannelNetworkFactory};
pub use network::{GroupId, METADATA_GROUP, RaftNetworkFactoryImpl, RaftNetworkImpl};
pub use placement::{Placement, PlacementConfig, PlacementError, PlacementOpenError, ShardId};
pub use raft_types::{BatchResponse, ClusterOperation, NodeInfo, RaftTypeConfig};
pub use redb_log_store::RedbLogStore;
pub use shard::ShardCluster;
pub use snapshot_store::{SnapshotStore, SnapshotStoreError, StoredSnapshot};
pub use state_machine::{ClusterStateMachine, EngineSnapshotProvider, GroupRole};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn raft_initialization() {
        let log_store = MemoryLogStore::new();
        let sm = ClusterStateMachine::new();
        let network = RaftNetworkFactoryImpl::new(PeerDirectory::default());

        let config = Arc::new(openraft::Config::default());
        let raft = openraft::Raft::<RaftTypeConfig>::new(1, config, network, log_store, sm)
            .await
            .expect("Raft initialization failed");

        // Raft initialized — it starts in Learner state.
        let metrics = raft.metrics();
        let m = metrics.borrow().clone();
        assert_eq!(m.id, 1);
    }
}
