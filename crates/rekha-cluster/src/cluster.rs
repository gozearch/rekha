//! Cluster types: the per-node clock map and cluster configuration.

use std::collections::HashMap;

use rekha_core::cluster::ClockTag;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClockMap {
    inner: HashMap<u64, ClockTag>,
}

impl ClockMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn advance(&mut self, node: u64, tag: ClockTag) {
        let entry = self.inner.entry(node).or_insert(tag);
        if tag.seq > entry.seq {
            *entry = tag;
        }
    }
    pub fn get(&self, node: u64) -> Option<&ClockTag> {
        self.inner.get(&node)
    }
    pub fn global_high_water(&self) -> Option<&ClockTag> {
        self.inner.values().max()
    }
    pub fn nodes(&self) -> impl Iterator<Item = u64> + '_ {
        self.inner.keys().copied()
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub node_id: u64,
    pub peers: Vec<String>,
    pub owned_collections: Vec<uuid::Uuid>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_map_advance() {
        let mut cm = ClockMap::new();
        assert!(cm.is_empty());
        cm.advance(1, ClockTag { node: 1, seq: 5 });
        assert_eq!(cm.len(), 1);
        assert_eq!(cm.get(1).unwrap().seq, 5);
        cm.advance(1, ClockTag { node: 1, seq: 10 });
        assert_eq!(cm.get(1).unwrap().seq, 10);
        cm.advance(1, ClockTag { node: 1, seq: 3 });
        assert_eq!(cm.get(1).unwrap().seq, 10);
    }

    #[test]
    fn clock_map_high_water() {
        let mut cm = ClockMap::new();
        cm.advance(1, ClockTag { node: 1, seq: 5 });
        cm.advance(2, ClockTag { node: 2, seq: 8 });
        let hw = cm.global_high_water().unwrap();
        assert_eq!(hw.node, 2);
        assert_eq!(hw.seq, 8);
    }
}
