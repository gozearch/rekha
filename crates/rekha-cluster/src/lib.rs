//! RekhaDB cluster types — ClockMap, WAL shipping, replica management,
//! and openraft integration for cluster metadata consensus.

pub mod cluster;
pub mod log_store;
pub mod network;
pub mod raft_types;
pub mod redb_log_store;
pub mod replication;
pub mod state_machine;

pub use cluster::{ClockMap, ClusterConfig, WalDelta, WalDeltaRecord};
pub use log_store::MemoryLogStore;
pub use network::channel::{ChannelHub, ChannelMessage, ChannelNetwork, ChannelNetworkFactory};
pub use network::{RaftNetworkFactoryImpl, RaftNetworkImpl};
pub use raft_types::{ClusterOperation, NodeInfo, RaftTypeConfig};
pub use redb_log_store::RedbLogStore;
pub use replication::WalReplication;
pub use state_machine::{ClusterStateMachine, EngineSnapshotProvider};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Arc;

    #[tokio::test]
    async fn raft_initialization() {
        let log_store = MemoryLogStore::new();
        let sm = ClusterStateMachine::new();
        let network = RaftNetworkFactoryImpl::new(HashMap::new());

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
