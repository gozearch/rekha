use rekha_core::{RekhaError, StorageError, now_micros};
use rocksdb::{DBWithThreadMode, IteratorMode, MultiThreaded};
use std::sync::Arc;

pub(crate) const CF_HINTS: &str = "hints";
const HINT_PREFIX_COLLECTION: &str = "coll:";

#[derive(Debug, Clone)]
pub struct HintEntry {
    pub target_node_id: String,
    pub collection: String,
    pub id: u64,
    pub vector: Vec<f32>,
    pub payload: Option<Vec<u8>>,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct HintStore {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
}

impl HintStore {
    pub fn new(db: Arc<DBWithThreadMode<MultiThreaded>>) -> Self {
        Self { db }
    }

    pub fn put_hint(
        &self, target_node_id: &str, collection: &str, id: u64, vector: &[f32],
        payload: Option<&[u8]>, timestamp: u64,
    ) -> Result<(), RekhaError> {
        let mut key = Vec::new();
        key.extend_from_slice(target_node_id.as_bytes());
        key.push(0);
        key.extend_from_slice(collection.as_bytes());
        key.push(0);
        key.extend_from_slice(&id.to_be_bytes());

        let vector_len = vector.len() as u32;
        let payload_data = payload.unwrap_or_default();
        let payload_len = payload_data.len() as u32;

        let mut value = Vec::with_capacity(8 + 4 + vector_len as usize * 4 + 4 + payload_data.len());
        value.extend_from_slice(&timestamp.to_le_bytes());
        value.extend_from_slice(&vector_len.to_le_bytes());
        for &v in vector {
            value.extend_from_slice(&v.to_le_bytes());
        }
        value.extend_from_slice(&payload_len.to_le_bytes());
        value.extend_from_slice(payload_data);

        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_HINTS.into(),
                msg: "handle not found".into(),
            }
        })?;
        self.db.put_cf(&cf, key, value).map_err(|e| {
            StorageError::Write { msg: e.to_string() }.into()
        })
    }

    pub fn iter_hints_for_node(&self, target_node_id: &str) -> Result<Vec<HintEntry>, RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_HINTS.into(),
                msg: "handle not found".into(),
            }
        })?;

        let mut prefix = target_node_id.as_bytes().to_vec();
        prefix.push(0);

        let mut results = Vec::new();
        let iter = self
            .db
            .iterator_cf(&cf, IteratorMode::From(&prefix, rocksdb::Direction::Forward));
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Internal {
                detail: format!("hints iteration failed: {e}"),
            })?;
            if key.len() < prefix.len() || key[..prefix.len()] != prefix[..] {
                break;
            }
            if value.len() < 16 {
                continue;
            }

            let timestamp = u64::from_le_bytes(value[0..8].try_into().unwrap());
            let vector_len = u32::from_le_bytes(value[8..12].try_into().unwrap()) as usize;
            let vector_end = 12 + vector_len * 4;
            if value.len() < vector_end + 4 {
                continue;
            }
            let vector: Vec<f32> = value[12..vector_end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let payload_len = u32::from_le_bytes(value[vector_end..vector_end + 4].try_into().unwrap()) as usize;
            let payload = if payload_len > 0 {
                if value.len() < vector_end + 4 + payload_len {
                    continue;
                }
                Some(value[vector_end + 4..vector_end + 4 + payload_len].to_vec())
            } else {
                None
            };

            let key_str = String::from_utf8(key.to_vec()).map_err(|_| RekhaError::Internal {
                detail: "invalid hint key utf8".into(),
            })?;
            let mut parts = key_str.splitn(3, '\0');
            let target = parts.next().unwrap_or_default().to_string();
            let collection = parts.next().unwrap_or_default().to_string();
            let id_part = parts.next().unwrap_or_default();
            let id_bytes = id_part.as_bytes();
            let id = if id_bytes.len() >= 8 {
                u64::from_be_bytes(id_bytes[id_bytes.len() - 8..].try_into().unwrap())
            } else {
                continue;
            };

            results.push(HintEntry {
                target_node_id: target,
                collection,
                id,
                vector,
                payload,
                timestamp,
            });
        }
        Ok(results)
    }

    pub fn delete_hint(&self, target_node_id: &str, collection: &str, id: u64) -> Result<(), RekhaError> {
        let mut key = Vec::new();
        key.extend_from_slice(target_node_id.as_bytes());
        key.push(0);
        key.extend_from_slice(collection.as_bytes());
        key.push(0);
        key.extend_from_slice(&id.to_be_bytes());

        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_HINTS.into(),
                msg: "handle not found".into(),
            }
        })?;
        self.db.delete_cf(&cf, key).map_err(|e| RekhaError::Internal {
            detail: format!("hint delete failed: {e}"),
        })
    }

    pub fn put_collection_hint(
        &self, target_node_id: &str, collection: &str,
        config_bytes: &[u8], timestamp: u64, op: u8,
    ) -> Result<(), RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| StorageError::ColumnFamily {
            name: CF_HINTS.into(), msg: "handle not found".into(),
        })?;
        let mut key = Vec::new();
        key.extend_from_slice(HINT_PREFIX_COLLECTION.as_bytes());
        key.extend_from_slice(target_node_id.as_bytes());
        key.push(0);
        key.extend_from_slice(collection.as_bytes());

        let mut value = Vec::with_capacity(9 + 4 + config_bytes.len());
        value.extend_from_slice(&timestamp.to_le_bytes());
        value.push(op);
        value.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        value.extend_from_slice(config_bytes);

        self.db.put_cf(&cf, key, value).map_err(|e| StorageError::Write { msg: e.to_string() }.into())
    }

    pub fn iter_collection_hints_for_node(&self, target_node_id: &str) -> Result<Vec<(String, u64, u8, Vec<u8>)>, RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| StorageError::ColumnFamily {
            name: CF_HINTS.into(), msg: "handle not found".into(),
        })?;
        let prefix = format!("{}{}\0", HINT_PREFIX_COLLECTION, target_node_id);
        let iter = self.db.iterator_cf(&cf, IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Internal { detail: format!("iteration error: {e}") })?;
            if !key.starts_with(prefix.as_bytes()) { break; }
            if value.len() < 13 { continue; }
            let timestamp = u64::from_le_bytes(value[0..8].try_into().unwrap());
            let op = value[8];
            let config_len = u32::from_le_bytes(value[9..13].try_into().unwrap()) as usize;
            let config_bytes = if value.len() >= 13 + config_len { value[13..13 + config_len].to_vec() } else { continue };
            let key_str = String::from_utf8(key.to_vec()).unwrap_or_default();
            let collection = key_str.split('\0').nth(1).unwrap_or("").to_string();
            results.push((collection, timestamp, op, config_bytes));
        }
        Ok(results)
    }

    pub fn delete_collection_hint(&self, target_node_id: &str, collection: &str) -> Result<(), RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| StorageError::ColumnFamily {
            name: CF_HINTS.into(), msg: "handle not found".into(),
        })?;
        let mut key = Vec::new();
        key.extend_from_slice(HINT_PREFIX_COLLECTION.as_bytes());
        key.extend_from_slice(target_node_id.as_bytes());
        key.push(0);
        key.extend_from_slice(collection.as_bytes());
        self.db.delete_cf(&cf, key).map_err(|e| RekhaError::Internal { detail: format!("delete hint: {e}") })
    }

    pub fn delete_expired_hints(&self, max_age_secs: u64) -> Result<u64, RekhaError> {
        let now = now_micros();
        let cutoff = now.saturating_sub(max_age_secs * 1_000_000);

        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_HINTS.into(),
                msg: "handle not found".into(),
            }
        })?;

        let mut count = 0u64;
        let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Internal {
                detail: format!("hints iteration failed: {e}"),
            })?;
            if value.len() < 8 {
                continue;
            }
            let ts = u64::from_le_bytes(value[0..8].try_into().unwrap());
            if ts < cutoff {
                self.db.delete_cf(&cf, &key).map_err(|e| RekhaError::Internal {
                    detail: format!("hint delete failed: {e}"),
                })?;
                count += 1;
            }
        }

        let prefix = HINT_PREFIX_COLLECTION.as_bytes();
        let iter2 = self.db.iterator_cf(&cf, IteratorMode::From(prefix, rocksdb::Direction::Forward));
        for item in iter2 {
            let (key, value) = item.map_err(|e| RekhaError::Internal {
                detail: format!("collection hints iteration failed: {e}"),
            })?;
            if !key.starts_with(prefix) {
                break;
            }
            if value.len() < 8 {
                continue;
            }
            let ts = u64::from_le_bytes(value[0..8].try_into().unwrap());
            if ts < cutoff {
                self.db.delete_cf(&cf, &key).map_err(|e| RekhaError::Internal {
                    detail: format!("collection hint delete failed: {e}"),
                })?;
                count += 1;
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::RocksVectorStore;

    fn setup_store(name: &str) -> (RocksVectorStore, HintStore) {
        let id = std::sync::atomic::AtomicU64::new(0).fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("{}_{}", name, id));
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let hints = HintStore::new(store.db().clone());
        (store, hints)
    }

    #[test]
    fn test_hint_roundtrip() {
        let (_store, hints) = setup_store("rekha_test_hint_rt");
        hints.put_hint("node2", "col1", 1, &[1.0, 2.0], Some(b"payload"), 100).unwrap();

        let results = hints.iter_hints_for_node("node2").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].target_node_id, "node2");
        assert_eq!(results[0].collection, "col1");
        assert_eq!(results[0].id, 1);
        assert_eq!(results[0].vector, vec![1.0, 2.0]);
        assert_eq!(results[0].payload, Some(b"payload".to_vec()));
        assert_eq!(results[0].timestamp, 100);

        hints.delete_hint("node2", "col1", 1).unwrap();
        let results = hints.iter_hints_for_node("node2").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_scavenge_expired_hints() {
        let (_store, hints) = setup_store("rekha_test_scavenge");
        let old_ts = 1000u64;
        let recent_ts = 9_999_999_999_999_999u64;

        hints.put_hint("node2", "col1", 1, &[1.0], None, old_ts).unwrap();
        hints.put_hint("node2", "col1", 2, &[2.0], None, recent_ts).unwrap();

        let deleted = hints.delete_expired_hints(100_000_000).unwrap();
        assert_eq!(deleted, 1);

        let results = hints.iter_hints_for_node("node2").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 2);
    }

    #[test]
    fn test_hint_isolation() {
        let (_store, hints) = setup_store("rekha_test_hint_iso");
        hints.put_hint("node-a", "c1", 1, &[1.0], None, 10).unwrap();
        hints.put_hint("node-b", "c1", 2, &[2.0], None, 20).unwrap();

        let hints_a = hints.iter_hints_for_node("node-a").unwrap();
        assert_eq!(hints_a.len(), 1);
        assert_eq!(hints_a[0].id, 1);

        let hints_b = hints.iter_hints_for_node("node-b").unwrap();
        assert_eq!(hints_b.len(), 1);
        assert_eq!(hints_b[0].id, 2);
    }

    #[test]
    fn test_collection_hint_roundtrip() {
        let (_store, hints) = setup_store("rekha_test_coll_hint_rt");
        hints.put_collection_hint("node2", "images", b"{\"dim\":256}", 100, 0).unwrap();

        let results = hints.iter_collection_hints_for_node("node2").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "images");
        assert_eq!(results[0].1, 100);
        assert_eq!(results[0].2, 0);
        assert_eq!(results[0].3, b"{\"dim\":256}");

        hints.delete_collection_hint("node2", "images").unwrap();
        let results = hints.iter_collection_hints_for_node("node2").unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_collection_hint_isolation() {
        let (_store, hints) = setup_store("rekha_test_coll_hint_iso");
        hints.put_collection_hint("node-a", "c1", b"config1", 10, 0).unwrap();
        hints.put_collection_hint("node-b", "c2", b"config2", 20, 1).unwrap();

        let hints_a = hints.iter_collection_hints_for_node("node-a").unwrap();
        assert_eq!(hints_a.len(), 1);
        assert_eq!(hints_a[0].0, "c1");
        assert_eq!(hints_a[0].2, 0);

        let hints_b = hints.iter_collection_hints_for_node("node-b").unwrap();
        assert_eq!(hints_b.len(), 1);
        assert_eq!(hints_b[0].0, "c2");
        assert_eq!(hints_b[0].2, 1);
    }
}
