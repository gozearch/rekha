use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::hash::{Hash, Hasher};
use rekha_core::NodeInfo;
use tokio::sync::RwLock;

pub type ChordId = u128;

pub const M: usize = 128;
pub const R: usize = 16;

pub fn hash_to_chord_id(data: &[u8]) -> ChordId {
    let mut h1 = std::hash::DefaultHasher::new();
    data.hash(&mut h1);
    let low = h1.finish() as u128;
    let mut h2 = std::hash::DefaultHasher::new();
    (data, 1u8).hash(&mut h2);
    let high = h2.finish() as u128;
    (high << 64) | low
}

pub fn between(id: ChordId, a: ChordId, b: ChordId, inclusive_a: bool, inclusive_b: bool) -> bool {
    if a == b {
        return true;
    }
    if a < b {
        let left = if inclusive_a { id >= a } else { id > a };
        let right = if inclusive_b { id <= b } else { id < b };
        left && right
    } else {
        id >= a || id <= b
    }
}

pub fn advance(id: ChordId, step: u128) -> ChordId {
    id.wrapping_add(step)
}

#[derive(Debug, Clone)]
pub struct FingerEntry {
    pub start: ChordId,
    pub node: Option<String>,
    pub node_address: Option<String>,
}

pub struct ChordNode {
    pub id: ChordId,
    pub address: String,
    pub self_id_string: String,
    pub predecessor: Arc<RwLock<Option<String>>>,
    pub predecessor_address: Arc<RwLock<Option<String>>>,
    pub finger: Arc<RwLock<Vec<FingerEntry>>>,
    pub successor_list: Arc<RwLock<Vec<String>>>,
    pub successor_addresses: Arc<RwLock<Vec<String>>>,
    pub next_finger: AtomicU32,
}

impl ChordNode {
    pub fn new(id: ChordId, address: &str) -> Self {
        let finger_start: Vec<ChordId> = (0..M).map(|i| advance(id, 1u128 << (i as u128 % 128))).collect();
        let finger: Vec<FingerEntry> = finger_start.iter().map(|s| FingerEntry {
            start: *s,
            node: None,
            node_address: None,
        }).collect();

        let self_id_string = id.to_string();
        ChordNode {
            id,
            address: address.to_string(),
            self_id_string,
            predecessor: Arc::new(RwLock::new(None)),
            predecessor_address: Arc::new(RwLock::new(None)),
            finger: Arc::new(RwLock::new(finger)),
            successor_list: Arc::new(RwLock::new(Vec::new())),
            successor_addresses: Arc::new(RwLock::new(Vec::new())),
            next_finger: AtomicU32::new(0),
        }
    }

    pub fn is_owner(&self, id: ChordId) -> bool {
        if let Ok(pred) = self.predecessor.try_read() {
            match pred.as_ref() {
                Some(_p) => {
                    let pred_id = hash_to_chord_id(_p.as_bytes());
                    between(id, pred_id, self.id, false, true)
                }
                None => true,
            }
        } else {
            true
        }
    }

    pub fn successor(&self) -> Option<(String, String)> {
        if let Ok(finger) = self.finger.try_read() {
            finger[0].node.clone().zip(finger[0].node_address.clone())
        } else {
            None
        }
    }

    pub fn set_successor(&self, node_id: &str, address: &str) {
        if let Ok(mut finger) = self.finger.try_write() {
            finger[0].node = Some(node_id.to_string());
            finger[0].node_address = Some(address.to_string());
        }
    }

    pub fn closest_preceding_node(&self, id: ChordId) -> Option<(String, String)> {
        if let Ok(finger) = self.finger.try_read() {
            for entry in finger.iter().rev() {
                if let (Some(ref nid), Some(ref addr)) = (&entry.node, &entry.node_address) {
                    let nid_hash = hash_to_chord_id(nid.as_bytes());
                    if between(nid_hash, self.id, id, false, false) {
                        return Some((nid.clone(), addr.clone()));
                    }
                }
            }
        }
        None
    }

    pub fn notify(&self, candidate_id: &str, candidate_addr: &str) -> bool {
        let candidate_hash = hash_to_chord_id(candidate_id.as_bytes());
        if let Ok(mut pred) = self.predecessor.try_write() {
            match pred.as_ref() {
                None => {
                    *pred = Some(candidate_id.to_string());
                    if let Ok(mut pred_addr) = self.predecessor_address.try_write() {
                        *pred_addr = Some(candidate_addr.to_string());
                    }
                    true
                }
                Some(current) => {
                    let current_hash = hash_to_chord_id(current.as_bytes());
                    if between(candidate_hash, current_hash, self.id, false, false) {
                        *pred = Some(candidate_id.to_string());
                        if let Ok(mut pred_addr) = self.predecessor_address.try_write() {
                            *pred_addr = Some(candidate_addr.to_string());
                        }
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        }
    }

    pub fn stabilize_with(&self, succ_id: &str, succ_addr: &str) {
        let _succ_hash = hash_to_chord_id(succ_id.as_bytes());
        let _pred_of_succ = "PLACEHOLDER";

        if let Ok(mut finger) = self.finger.try_write() {
            finger[0].node = Some(succ_id.to_string());
            finger[0].node_address = Some(succ_addr.to_string());
        }
    }

    pub async fn run_stabilize<F>(&self, successor_fn: F)
    where
        F: Fn(&str, &str) -> Option<(String, String)>,
    {
        let succ = self.successor();
        if let Some((succ_id, succ_addr)) = succ {
            if let Some((pred_id, pred_addr)) = successor_fn(&succ_id, &succ_addr) {
                let pred_hash = hash_to_chord_id(pred_id.as_bytes());
                let succ_hash = hash_to_chord_id(succ_id.as_bytes());
                if between(pred_hash, self.id, succ_hash, false, false) {
                    self.set_successor(&pred_id, &pred_addr);
                }
            }
        }
    }

    pub fn fix_next_finger<F>(&self, lookup_fn: F) -> bool
    where
        F: FnOnce(ChordId) -> Option<(String, String)>,
    {
        let idx = self.next_finger.load(Ordering::Relaxed) as usize;
        let start = advance(self.id, 1u128 << (idx as u128 % 128));
        if let Some((node_id, addr)) = lookup_fn(start) {
            if let Ok(mut finger) = self.finger.try_write() {
                if idx < finger.len() {
                    finger[idx].node = Some(node_id);
                    finger[idx].node_address = Some(addr);
                }
            }
        }
        let next = (idx + 1) % M;
        self.next_finger.store(next as u32, Ordering::Relaxed);
        true
    }

    pub fn check_predecessor(&self, predecessor_alive: bool) {
        if !predecessor_alive {
            if let Ok(mut pred) = self.predecessor.try_write() {
                *pred = None;
            }
            if let Ok(mut pred_addr) = self.predecessor_address.try_write() {
                *pred_addr = None;
            }
        }
    }

    pub fn handle_find_successor(&self, id: ChordId) -> Option<(String, String)> {
        if self.is_owner(id) {
            Some((self.id.to_string(), self.address.clone()))
        } else {
            self.closest_preceding_node(id).or_else(|| self.successor())
        }
    }

    pub async fn replicas_for_chord_id(&self, _id: ChordId, rf: usize) -> Vec<NodeInfo> {
        let mut results = Vec::new();

        results.push(NodeInfo {
            node_id: self.self_id_string.clone(),
            address: self.address.clone(),
            is_alive: true,
        });

        let successors = self.successor_list.read().await;
        let addresses = self.successor_addresses.read().await;
        for (i, succ) in successors.iter().enumerate() {
            if results.len() >= rf { break; }
            let addr = addresses.get(i).cloned().unwrap_or_default();
            results.push(NodeInfo {
                node_id: succ.clone(),
                address: addr,
                is_alive: true,
            });
        }
        drop(addresses);

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_chord() -> ChordNode {
        let id = hash_to_chord_id(b"test-node");
        ChordNode::new(id, "127.0.0.1:5000")
    }

    #[tokio::test]
    async fn test_replicas_for_chord_id_returns_addresses() {
        let chord = test_chord();
        chord.successor_list.write().await.push("nodeA".to_string());
        chord.successor_list.write().await.push("nodeB".to_string());
        chord.successor_addresses.write().await.push("addrA:5001".to_string());
        chord.successor_addresses.write().await.push("addrB:5002".to_string());

        let replicas = chord.replicas_for_chord_id(0, 3).await;
        assert_eq!(replicas.len(), 3, "should return self + 2 successors");

        assert_eq!(replicas[0].node_id, chord.self_id_string, "first should be self");
        assert!(!replicas[1].address.is_empty(), "successor address must not be empty");
        assert!(!replicas[2].address.is_empty(), "successor address must not be empty");
        assert_eq!(replicas[1].address, "addrA:5001", "address should match successor_addresses entry");
        assert_eq!(replicas[2].address, "addrB:5002", "address should match successor_addresses entry");
    }

    #[tokio::test]
    async fn test_replicas_for_chord_id_with_empty_successors() {
        let chord = test_chord();
        let replicas = chord.replicas_for_chord_id(0, 3).await;
        assert_eq!(replicas.len(), 1, "only self when no successors");
        assert_eq!(replicas[0].node_id, chord.self_id_string);
    }

    #[tokio::test]
    async fn test_replicas_for_chord_id_truncates_by_rf() {
        let chord = test_chord();
        for i in 0..5 {
            chord.successor_list.write().await.push(format!("node{}", i));
            chord.successor_addresses.write().await.push(format!("addr{}.local:{}", i, 5000 + i));
        }

        // rf=3 means 3 total entries (1 self + up to 2 successors)
        let replicas = chord.replicas_for_chord_id(0, 3).await;
        assert_eq!(replicas.len(), 3, "rf=3 should return self + 2 successors = 3");
        assert_eq!(replicas[0].node_id, chord.self_id_string);
        assert_eq!(replicas[1].node_id, "node0");
        assert_eq!(replicas[2].node_id, "node1");
    }
}
