use std::collections::BTreeMap;

const VNODE_COUNT: u32 = 128;

pub struct HashRing {
    ring: BTreeMap<u64, String>,
    nodes: Vec<String>,
}

impl HashRing {
    pub fn new() -> Self {
        HashRing {
            ring: BTreeMap::new(),
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node_id: &str) {
        if self.nodes.contains(&node_id.to_string()) {
            return;
        }
        for v in 0..VNODE_COUNT {
            let key = self.hash_vnode(node_id, v);
            self.ring.insert(key, node_id.to_string());
        }
        self.nodes.push(node_id.to_string());
    }

    #[allow(dead_code)]
    pub fn remove_node(&mut self, node_id: &str) {
        for v in 0..VNODE_COUNT {
            let key = self.hash_vnode(node_id, v);
            self.ring.remove(&key);
        }
        self.nodes.retain(|n| n != node_id);
    }

    pub fn replicas_for(&self, shard: u64, rf: usize) -> Vec<String> {
        if self.nodes.is_empty() || rf == 0 {
            return Vec::new();
        }
        let hash = self.hash_shard(shard);
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (_hash, node) in self.ring.range(hash..) {
            if seen.insert(node.clone()) {
                results.push(node.clone());
                if results.len() >= rf {
                    return results;
                }
            }
        }
        for node in self.ring.values() {
            if seen.insert(node.clone()) {
                results.push(node.clone());
                if results.len() >= rf {
                    return results;
                }
            }
        }
        results
    }

    #[allow(dead_code)]
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    fn hash_vnode(&self, node_id: &str, vnode: u32) -> u64 {
        let data = format!("{}:{}", node_id, vnode);
        sip_hash(data.as_bytes())
    }

    fn hash_shard(&self, shard: u64) -> u64 {
        let data = shard.to_le_bytes();
        sip_hash(&data)
    }
}

fn sip_hash(data: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    data.hash(&mut hasher);
    hasher.finish()
}

impl Default for HashRing {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        assert_eq!(ring.nodes().len(), 2);
    }

    #[test]
    fn test_remove_node() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        ring.remove_node("node1");
        assert_eq!(ring.nodes().len(), 1);
    }

    #[test]
    fn test_replicas_for() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");
        let replicas = ring.replicas_for(42, 2);
        assert_eq!(replicas.len(), 2);
    }

    #[test]
    fn test_replicas_for_rf_larger_than_nodes() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        let replicas = ring.replicas_for(42, 3);
        assert_eq!(replicas.len(), 1);
    }

    #[test]
    fn test_replicas_for_empty_ring() {
        let ring = HashRing::new();
        let replicas = ring.replicas_for(42, 2);
        assert!(replicas.is_empty());
    }

    #[test]
    fn test_duplicate_add() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        ring.add_node("node1");
        assert_eq!(ring.nodes().len(), 1);
    }

    #[test]
    fn test_deterministic_replicas() {
        let mut ring = HashRing::new();
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");
        let r1 = ring.replicas_for(42, 2);
        let r2 = ring.replicas_for(42, 2);
        assert_eq!(r1, r2);
    }
}
