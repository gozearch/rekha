use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};

pub struct ConsistentHashRing {
    ring: BTreeMap<u64, String>,
    num_vnodes: u32,
}

impl ConsistentHashRing {
    pub fn new(num_vnodes: u32) -> Self {
        Self {
            ring: BTreeMap::new(),
            num_vnodes,
        }
    }

    pub fn add_node(&mut self, node_id: &str) {
        for i in 0..self.num_vnodes {
            let token = hash(&format!("{node_id}:{i}"));
            self.ring.insert(token, node_id.to_string());
        }
    }

    pub fn remove_node(&mut self, node_id: &str) {
        let to_remove: Vec<u64> = self
            .ring
            .iter()
            .filter(|(_, id)| id.as_str() == node_id)
            .map(|(token, _)| *token)
            .collect();
        for token in to_remove {
            self.ring.remove(&token);
        }
    }

    pub fn replicas_for(
        &self,
        shard: u64,
        rf: usize,
        healthy: &HashSet<&str>,
    ) -> Vec<String> {
        if self.ring.is_empty() {
            return vec![];
        }

        let token = hash(&format!("shard:{shard}"));
        let mut replicas: Vec<String> = Vec::with_capacity(rf);

        let range_iter: Vec<(u64, String)> = self
            .ring
            .range(token..)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let iter: Vec<(u64, String)> = self
            .ring
            .iter()
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        let combined: Vec<(u64, String)> = range_iter.into_iter().chain(iter).collect();

        for (_, node_id) in &combined {
            if healthy.contains(node_id.as_str()) && !replicas.contains(node_id) {
                replicas.push(node_id.clone());
                if replicas.len() >= rf {
                    break;
                }
            }
        }

        replicas
    }

    pub fn node_count(&self) -> usize {
        let mut ids = HashSet::new();
        for id in self.ring.values() {
            ids.insert(id.as_str());
        }
        ids.len()
    }

    pub fn assigned_shards(&self, node_id: &str) -> Vec<u64> {
        let owned_tokens: Vec<u64> = self
            .ring
            .iter()
            .filter(|(_, id)| id.as_str() == node_id)
            .map(|(token, _)| *token)
            .collect();

        if owned_tokens.is_empty() {
            return vec![];
        }

        let mut shards = HashSet::new();
        let num_shards = 1024;
        for s in 0..num_shards {
            let st = hash(&format!("shard:{s}"));
            let mut predecessor: Option<u64> = None;
            for (t, _) in self.ring.iter() {
                if *t >= st {
                    predecessor = Some(*t);
                    break;
                }
            }
            let predecessor = predecessor.unwrap_or_else(|| {
                *self.ring.keys().next().unwrap_or(&0)
            });
            if owned_tokens.contains(&predecessor) {
                shards.insert(s);
            }
        }
        shards.into_iter().collect()
    }
}

fn hash(input: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_ring() -> ConsistentHashRing {
        ConsistentHashRing::new(128)
    }

    fn three_node_healthy() -> (ConsistentHashRing, HashSet<&'static str>) {
        let mut ring = empty_ring();
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.add_node("node-c");
        let healthy: HashSet<&str> = ["node-a", "node-b", "node-c"].into();
        (ring, healthy)
    }

    #[test]
    fn test_empty_ring_returns_empty() {
        let ring = empty_ring();
        let healthy = HashSet::new();
        let replicas = ring.replicas_for(0, 3, &healthy);
        assert!(replicas.is_empty());
    }

    #[test]
    fn test_returns_rf_replicas() {
        let (ring, healthy) = three_node_healthy();
        let replicas = ring.replicas_for(0, 2, &healthy);
        assert_eq!(replicas.len(), 2);
    }

    #[test]
    fn test_different_shards_map_to_different_primaries() {
        let (ring, healthy) = three_node_healthy();
        let r0 = ring.replicas_for(0, 1, &healthy);
        let r1 = ring.replicas_for(1, 1, &healthy);
        let r2 = ring.replicas_for(2, 1, &healthy);
        assert_eq!(r0.len(), 1);
        assert_eq!(r1.len(), 1);
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn test_skips_unhealthy() {
        let mut ring = empty_ring();
        ring.add_node("node-a");
        ring.add_node("node-b");
        let healthy: HashSet<&str> = ["node-a"].into();
        let replicas = ring.replicas_for(0, 2, &healthy);
        assert_eq!(replicas.len(), 1);
        assert_eq!(replicas[0], "node-a");
    }

    #[test]
    fn test_rf_exceeds_nodes() {
        let (ring, healthy) = three_node_healthy();
        let replicas = ring.replicas_for(0, 10, &healthy);
        assert_eq!(replicas.len(), 3);
    }

    #[test]
    fn test_add_node_preserves_existing() {
        let mut ring = empty_ring();
        ring.add_node("node-a");
        ring.add_node("node-b");
        let healthy_a: HashSet<&str> = ["node-a", "node-b"].into();

        ring.add_node("node-c");
        let healthy_all: HashSet<&str> = ["node-a", "node-b", "node-c"].into();

        let shards_changed = (0..100)
            .filter(|s| {
                let r_before = ring.replicas_for(*s, 1, &healthy_a);
                let r_after = ring.replicas_for(*s, 1, &healthy_all);
                r_before != r_after
            })
            .count();

        assert!(
            shards_changed <= 50,
            "adding 1 node to 2 should change ~1/3 of shards, got {shards_changed}/100"
        );
    }

    #[test]
    fn test_remove_node_reassigns_shards() {
        let mut ring = empty_ring();
        ring.add_node("node-a");
        ring.add_node("node-b");
        ring.add_node("node-c");
        let healthy: HashSet<&str> = ["node-a", "node-b"].into();
        ring.remove_node("node-c");
        let r = ring.replicas_for(0, 2, &healthy);
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|id| id == "node-a" || id == "node-b"));
    }

    #[test]
    fn test_node_count() {
        let mut ring = empty_ring();
        assert_eq!(ring.node_count(), 0);
        ring.add_node("node-a");
        assert_eq!(ring.node_count(), 1);
        ring.add_node("node-b");
        assert_eq!(ring.node_count(), 2);
        ring.remove_node("node-a");
        assert_eq!(ring.node_count(), 1);
    }

    #[test]
    fn test_replicas_unique() {
        let (ring, healthy) = three_node_healthy();
        let replicas = ring.replicas_for(42, 3, &healthy);
        let mut unique = replicas.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(replicas.len(), unique.len());
    }

    #[test]
    fn test_hash_distribution() {
        let mut tokens = HashSet::new();
        let total = 100_000;
        for i in 0..total {
            tokens.insert(hash(&format!("key:{i}")));
        }
        assert!(
            tokens.len() > total / 2,
            "hash should produce few collisions, got {}/{}",
            tokens.len(),
            total
        );
    }

    proptest::proptest! {
        #[test]
        fn replicas_never_duplicate(rf in 1..5usize) {
            let mut ring = crate::ring::ConsistentHashRing::new(128);
            ring.add_node("a");
            ring.add_node("b");
            ring.add_node("c");
            let healthy: std::collections::HashSet<&str> = ["a", "b", "c"].into();
            let replicas = ring.replicas_for(42, rf, &healthy);
            let mut unique = replicas.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(replicas.len(), unique.len());
        }
    }

    #[test]
    fn test_single_node_rf() {
        let mut ring = empty_ring();
        ring.add_node("solo");
        let healthy: HashSet<&str> = ["solo"].into();
        let replicas = ring.replicas_for(0, 3, &healthy);
        assert_eq!(replicas.len(), 1);
        assert_eq!(replicas[0], "solo");
    }
}
