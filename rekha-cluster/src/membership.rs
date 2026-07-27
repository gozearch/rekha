use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use tokio::sync::RwLock;

use crate::ring::HashRing;

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub is_alive: bool,
    pub last_seen: Instant,
}

#[allow(dead_code)]
pub struct Membership {
    nodes: DashMap<String, NodeInfo>,
    ring: Arc<RwLock<HashRing>>,
    self_id: String,
    heartbeat_timeout_ms: u64,
}

impl Membership {
    pub fn new(self_id: &str, heartbeat_timeout_ms: u64) -> Self {
        Membership {
            nodes: DashMap::new(),
            ring: Arc::new(RwLock::new(HashRing::new())),
            self_id: self_id.to_string(),
            heartbeat_timeout_ms,
        }
    }

    pub fn self_id(&self) -> &str {
        &self.self_id
    }

    pub fn add_peer(&self, node_id: &str, address: &str) {
        self.nodes.insert(
            node_id.to_string(),
            NodeInfo {
                node_id: node_id.to_string(),
                address: address.to_string(),
                is_alive: true,
                last_seen: Instant::now(),
            },
        );
    }

    pub fn mark_alive(&self, node_id: &str) {
        if let Some(mut n) = self.nodes.get_mut(node_id) {
            n.is_alive = true;
            n.last_seen = Instant::now();
        }
    }

    pub fn mark_dead(&self, node_id: &str) {
        if let Some(mut n) = self.nodes.get_mut(node_id) {
            n.is_alive = false;
        }
    }

    pub fn remove_peer(&self, node_id: &str) {
        self.nodes.remove(node_id);
    }

    pub fn get_peer(&self, node_id: &str) -> Option<NodeInfo> {
        self.nodes.get(node_id).map(|n| n.clone())
    }

    pub fn alive_peers(&self) -> Vec<NodeInfo> {
        self.nodes
            .iter()
            .filter(|n| n.is_alive)
            .map(|n| n.clone())
            .collect()
    }

    pub fn all_peers(&self) -> Vec<NodeInfo> {
        self.nodes.iter().map(|n| n.clone()).collect()
    }

    pub async fn rebuild_ring(&self) {
        let mut ring = self.ring.write().await;
        let mut new_ring = HashRing::new();
        new_ring.add_node(&self.self_id);
        for node in self.alive_peers() {
            new_ring.add_node(&node.node_id);
        }
        *ring = new_ring;
    }

    pub async fn replicas_for(&self, shard: u64, rf: usize) -> Vec<String> {
        let ring = self.ring.read().await;
        ring.replicas_for(shard, rf)
            .into_iter()
            .filter(|n| n != &self.self_id)
            .collect()
    }

    pub fn check_timeouts(&self, timeout_ms: u64) {
        let now = Instant::now();
        let timeout = Duration::from_millis(timeout_ms);
        let dead_peers: Vec<String> = self
            .nodes
            .iter()
            .filter(|entry| {
                let age = now - entry.value().last_seen;
                age >= timeout
            })
            .map(|entry| entry.key().clone())
            .collect();

        for node_id in dead_peers {
            self.mark_dead(&node_id);
        }
    }

    pub async fn handle_heartbeat(&self, node_id: &str, address: &str) {
        if !self.nodes.contains_key(node_id) {
            self.add_peer(node_id, address);
            self.rebuild_ring().await;
        } else {
            self.mark_alive(node_id);
        }
    }

    pub async fn handle_peer_leaving(&self, node_id: &str) {
        self.remove_peer(node_id);
        self.rebuild_ring().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_get_peer() {
        let m = Membership::new("self", 5000);
        m.add_peer("node1", "127.0.0.1:5001");
        let peer = m.get_peer("node1").unwrap();
        assert_eq!(peer.node_id, "node1");
        assert!(peer.is_alive);
    }

    #[tokio::test]
    async fn test_mark_dead() {
        let m = Membership::new("self", 5000);
        m.add_peer("node1", "127.0.0.1:5001");
        m.mark_dead("node1");
        let peer = m.get_peer("node1").unwrap();
        assert!(!peer.is_alive);
    }

    #[tokio::test]
    async fn test_remove_peer() {
        let m = Membership::new("self", 5000);
        m.add_peer("node1", "127.0.0.1:5001");
        m.remove_peer("node1");
        assert!(m.get_peer("node1").is_none());
    }

    #[tokio::test]
    async fn test_rebuild_ring() {
        let m = Membership::new("self", 5000);
        m.add_peer("node1", "127.0.0.1:5001");
        m.add_peer("node2", "127.0.0.1:5002");
        m.rebuild_ring().await;
        let replicas = m.replicas_for(42, 2).await;
        assert!(
            replicas.len() <= 2,
            "should return up to 2 non-self replicas"
        );
    }
}
