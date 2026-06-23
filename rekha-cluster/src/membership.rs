use crate::ring::ConsistentHashRing;
use crate::peer_state::PeerState;
use rekha_core::{NodeInfo, NodeStatus};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub struct Membership {
    peers: HashMap<String, PeerState>,
    #[allow(dead_code)]
    self_node_id: String,
    ring: ConsistentHashRing,
    peer_timeout: Duration,
}

impl Membership {
    pub fn new(self_node_id: String) -> Self {
        Self::with_timeout(self_node_id, Duration::from_secs(10))
    }

    pub fn with_timeout(self_node_id: String, peer_timeout: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            self_node_id,
            ring: ConsistentHashRing::new(128),
            peer_timeout,
        }
    }

    pub fn register(&mut self, info: NodeInfo) {
        let node_id = info.node_id.clone();
        if let Some(existing) = self.peers.get_mut(&node_id) {
            let preserved_status = existing.info.status.clone();
            existing.info = info;
            existing.info.status = preserved_status;
            existing.last_seen = Instant::now();
        } else {
            self.peers.insert(node_id.clone(), PeerState::new(info));
            self.ring.add_node(&node_id);
        }
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
            if peer.last_seen.elapsed() > self.peer_timeout && peer.info.status == NodeStatus::Healthy {
                peer.info.status = NodeStatus::Unreachable;
            } else if peer.last_seen.elapsed() <= self.peer_timeout && peer.info.status == NodeStatus::Unreachable {
                peer.info.status = NodeStatus::Healthy;
                recovered.push(peer.info.node_id.clone());
            }
        }
        recovered
    }

    pub fn remove(&mut self, node_id: &str) {
        self.peers.remove(node_id);
        self.ring.remove_node(node_id);
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

    pub fn all_peers(&self) -> Vec<NodeInfo> {
        self.peers.values().map(|p| p.info.clone()).collect()
    }

    pub fn replicas_for(&self, shard: u64, rf: usize) -> Vec<NodeInfo> {
        let healthy: HashSet<&str> = self.peers.iter()
            .filter(|(_, p)| p.info.status == NodeStatus::Healthy)
            .map(|(id, _)| id.as_str())
            .collect();
        self.ring.replicas_for(shard, rf, &healthy)
            .iter()
            .filter_map(|id| self.peers.get(id).map(|p| p.info.clone()))
            .collect()
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

    fn make_info(node_id: &str) -> NodeInfo {
        NodeInfo {
            node_id: node_id.into(), address: "addr".into(),
            partition_id: 0, dim_groups: vec![], is_leader: false,
            raft_term: 0, commit_index: 0, storage_bytes: 0,
            status: NodeStatus::Healthy, last_heartbeat: 0,
        }
    }

    #[test]
    fn test_register_preserves_unreachable_status() {
        let mut m = Membership::with_timeout("self".into(), Duration::from_millis(50));
        m.register(make_info("peer1"));
        assert_eq!(m.healthy_peers().len(), 1);
        std::thread::sleep(Duration::from_millis(60));
        let recovered = m.check_health();
        assert!(recovered.is_empty());
        assert_eq!(m.healthy_peers().len(), 0);
        // Re-register — status should remain Unreachable
        let mut refreshed = make_info("peer1");
        refreshed.status = NodeStatus::Healthy;
        m.register(refreshed);
        assert_eq!(m.healthy_peers().len(), 0, "register must preserve Unreachable status");
        let recovered = m.check_health();
        assert_eq!(recovered.len(), 1, "check_health should detect the recovery");
        assert_eq!(m.healthy_peers().len(), 1);
    }

    #[test]
    fn test_replicas_for_returns_healthy() {
        let m = three_nodes(&Duration::from_secs(10));
        let replicas = m.replicas_for(0, 2);
        assert_eq!(replicas.len(), 2);
    }

    #[test]
    fn test_replicas_for_skips_unreachable() {
        let mut m = three_nodes(&Duration::from_millis(50));
        // Simulate making one node unreachable
        if let Some(peer) = m.peers.get_mut("node-c") {
            peer.info.status = NodeStatus::Unreachable;
        }
        // Should only return 2 (node-c is skipped)
        let replicas = m.replicas_for(0, 3);
        assert_eq!(replicas.len(), 2);
        for r in &replicas {
            assert_ne!(r.node_id, "node-c");
        }
    }

    #[test]
    fn test_replicas_for_empty_when_no_nodes() {
        let m = Membership::new("self".into());
        assert!(m.replicas_for(0, 3).is_empty());
    }

    #[test]
    fn test_remove_also_removes_from_ring() {
        let mut m = three_nodes(&Duration::from_secs(10));
        m.remove("node-b");
        let replicas = m.replicas_for(0, 3);
        assert_eq!(replicas.len(), 2);
        for r in &replicas {
            assert_ne!(r.node_id, "node-b");
        }
    }

    fn three_nodes(timeout: &Duration) -> Membership {
        let mut m = Membership::with_timeout("self".into(), timeout.clone());
        m.register(make_info("node-a"));
        m.register(make_info("node-b"));
        m.register(make_info("node-c"));
        m
    }
}
