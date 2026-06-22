use rekha_core::{NodeInfo, NodeStatus};
use std::collections::HashMap;
use std::time::Duration;
use crate::peer_state::PeerState;

const PEER_TIMEOUT: Duration = Duration::from_secs(10);

pub struct Membership {
    peers: HashMap<String, PeerState>,
    #[allow(dead_code)]
    self_node_id: String,
}

impl Membership {
    pub fn new(self_node_id: String) -> Self {
        Self { peers: HashMap::new(), self_node_id }
    }

    pub fn register(&mut self, info: NodeInfo) {
        let node_id = info.node_id.clone();
        self.peers.insert(node_id, PeerState::new(info));
    }

    pub fn register_from_heartbeat(&mut self, node_id: String, address: String, storage_bytes: u64) {
        self.register(NodeInfo {
            node_id, address,
            partition_id: 0, dim_groups: Vec::new(),
            is_leader: false, raft_term: 0, commit_index: 0,
            storage_bytes,
            status: NodeStatus::Healthy, last_heartbeat: 0,
        });
    }

    pub fn check_health(&mut self) -> Vec<String> {
        let mut recovered = Vec::new();
        for peer in self.peers.values_mut() {
            if peer.last_seen.elapsed() > PEER_TIMEOUT && peer.info.status == NodeStatus::Healthy {
                peer.info.status = NodeStatus::Unreachable;
            } else if peer.last_seen.elapsed() <= PEER_TIMEOUT && peer.info.status == NodeStatus::Unreachable {
                peer.info.status = NodeStatus::Healthy;
                recovered.push(peer.info.node_id.clone());
            }
        }
        recovered
    }

    pub fn remove(&mut self, node_id: &str) {
        self.peers.remove(node_id);
    }

    pub fn healthy_peers(&self) -> Vec<NodeInfo> {
        self.peers.values()
            .filter(|p| p.info.status == NodeStatus::Healthy)
            .map(|p| p.info.clone())
            .collect()
    }

    pub fn get(&self, node_id: &str) -> Option<&PeerState> {
        self.peers.get(node_id)
    }

    pub fn peers_for_handshake(&self, exclude: &str) -> Vec<NodeInfo> {
        self.peers.values()
            .filter(|p| p.info.node_id != exclude)
            .map(|p| p.info.clone())
            .collect()
    }

    pub fn count(&self) -> usize {
        self.peers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_healthy() {
        let mut m = Membership::new("self".into());
        assert_eq!(m.count(), 0);
        m.register(NodeInfo {
            node_id: "peer1".into(), address: "addr".into(),
            partition_id: 0, dim_groups: vec![], is_leader: false,
            raft_term: 0, commit_index: 0, storage_bytes: 0,
            status: NodeStatus::Healthy, last_heartbeat: 0,
        });
        assert_eq!(m.count(), 1);
        assert_eq!(m.healthy_peers().len(), 1);
    }

    #[test]
    fn test_remove() {
        let mut m = Membership::new("self".into());
        m.register(NodeInfo { node_id: "p1".into(), address: "a".into(),
            partition_id: 0, dim_groups: vec![], is_leader: false,
            raft_term: 0, commit_index: 0, storage_bytes: 0,
            status: NodeStatus::Healthy, last_heartbeat: 0,
        });
        m.remove("p1");
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_peers_for_handshake_excludes_self() {
        let mut m = Membership::new("self".into());
        m.register(NodeInfo { node_id: "peer".into(), address: "a".into(),
            partition_id: 0, dim_groups: vec![], is_leader: false,
            raft_term: 0, commit_index: 0, storage_bytes: 0,
            status: NodeStatus::Healthy, last_heartbeat: 0,
        });
        assert_eq!(m.peers_for_handshake("peer").len(), 0);
        assert_eq!(m.peers_for_handshake("other").len(), 1);
    }
}
