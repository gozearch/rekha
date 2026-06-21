use rekha_core::{RekhaError, StorageError, VectorStoreBackend};
use rocksdb::{
    ColumnFamilyDescriptor, DBWithThreadMode, IteratorMode, MultiThreaded, Options,
};
use std::path::Path;
use std::sync::Arc;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";

#[derive(Clone)]
pub struct RocksVectorStore {
    db: Arc<DBWithThreadMode<MultiThreaded>>,
    namespace: Option<String>,
    max_payload_size: usize,
}

impl RocksVectorStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RekhaError> {
        let path = path.as_ref();
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let required: Vec<&str> = vec![CF_VECTORS, CF_PAYLOADS, CF_METADATA];
        let existing_list = DBWithThreadMode::<MultiThreaded>::list_cf(&opts, path)
            .unwrap_or_default();

        let mut all_cf_names = required.clone();
        if !existing_list.is_empty() {
            for name in &existing_list {
                if !all_cf_names.contains(&name.as_str()) {
                    all_cf_names.push(name.as_str());
                }
            }
        }

        let cf_descriptors: Vec<ColumnFamilyDescriptor> =
            all_cf_names.iter().map(|name| {
                let mut cf_opts = Options::default();
                cf_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
                ColumnFamilyDescriptor::new(*name, cf_opts)
            }).collect();

        let db =
            DBWithThreadMode::<MultiThreaded>::open_cf_descriptors(&opts, path, cf_descriptors)
                .map_err(|e| StorageError::DbOpen {
                    path: path.display().to_string(),
                    source: e.to_string(),
                })?;

        Ok(Self {
            db: Arc::new(db),
            namespace: None,
            max_payload_size: 1024 * 1024,
        })
    }

    pub fn from_db(db: Arc<DBWithThreadMode<MultiThreaded>>, namespace: Option<String>) -> Self {
        Self {
            db,
            namespace,
            max_payload_size: 1024 * 1024,
        }
    }

    pub fn with_namespace(mut self, namespace: String) -> Self {
        self.namespace = Some(namespace);
        self
    }

    pub fn get_namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    pub fn with_max_payload_size(mut self, max: usize) -> Self {
        self.max_payload_size = max;
        self
    }

    pub fn db(&self) -> &Arc<DBWithThreadMode<MultiThreaded>> {
        &self.db
    }

    fn encode_key(&self, id: u64) -> Vec<u8> {
        let mut key = Vec::with_capacity(8);
        if let Some(ref ns) = self.namespace {
            key.extend_from_slice(ns.as_bytes());
            key.push(0);
        }
        key.extend_from_slice(&id.to_be_bytes());
        key
    }

    fn decode_id(&self, key: &[u8]) -> Option<u64> {
        if let Some(ref ns) = self.namespace {
            let prefix_len = ns.len() + 1;
            if key.len() < prefix_len + 8 || key[..prefix_len - 1] != ns.as_bytes()[..]
                || key[prefix_len - 1] != 0
            {
                return None;
            }
            let id_bytes = &key[key.len() - 8..];
            Some(u64::from_be_bytes(id_bytes.try_into().ok()?))
        } else if key.len() >= 8 {
            let id_bytes = &key[key.len() - 8..];
            Some(u64::from_be_bytes(id_bytes.try_into().ok()?))
        } else {
            None
        }
    }

    fn namespace_prefix(&self) -> Option<Vec<u8>> {
        self.namespace.as_ref().map(|ns| {
            let mut prefix = ns.as_bytes().to_vec();
            prefix.push(0);
            prefix
        })
    }
}

impl Drop for RocksVectorStore {
    fn drop(&mut self) {
        let _ = self.db.flush_wal(true);
    }
}

impl RocksVectorStore {
    pub fn delete_all_in_namespace(&self) -> Result<u64, RekhaError> {
        let prefix = self
            .namespace_prefix()
            .ok_or_else(|| RekhaError::Internal {
                detail: "delete_all_in_namespace requires a namespace".into(),
            })?;
        let mut count = 0u64;
        for cf_name in &[CF_VECTORS, CF_PAYLOADS] {
            let cf = self
                .db
                .cf_handle(cf_name)
                .ok_or_else(|| StorageError::ColumnFamily {
                    name: cf_name.to_string(),
                    source: "handle not found".into(),
                })?;
            let mut batch = rocksdb::WriteBatch::default();
            let iter = self.db.iterator_cf(&cf, IteratorMode::Start);
            for result in iter {
                let (key, _) = result.map_err(|e| RekhaError::Internal {
                    detail: format!("db iteration error: {e}"),
                })?;
                if key.len() < prefix.len() || key[..prefix.len()] != prefix[..] {
                    break;
                }
                batch.delete_cf(&cf, &key);
                count += 1;
            }
            self.db.write(batch).map_err(|e| RekhaError::Internal {
                detail: format!("failed to delete namespace keys: {e}"),
            })?;
        }
        Ok(count)
    }

    pub fn get_storage_estimate(&self) -> Result<u64, RekhaError> {
        self.iter_ids().map(|ids| ids.len() as u64)
    }

    pub fn put_metadata(&self, key: &str, value: &[u8]) -> Result<(), RekhaError> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_METADATA.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.put_cf(&cf, key, value).map_err(|e| {
            RekhaError::Internal {
                detail: format!("metadata write failed: {e}"),
            }
        })
    }

    pub fn get_metadata(&self, key: &str) -> Result<Option<Vec<u8>>, RekhaError> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_METADATA.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.get_cf(&cf, key).map_err(|e| RekhaError::Internal {
            detail: format!("metadata read failed: {e}"),
        })
    }

    pub fn delete_metadata(&self, key: &str) -> Result<(), RekhaError> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_METADATA.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.delete_cf(&cf, key).map_err(|e| RekhaError::Internal {
            detail: format!("metadata delete failed: {e}"),
        })
    }

    pub fn iter_metadata_prefix(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>, RekhaError> {
        let cf = self.db.cf_handle(CF_METADATA).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_METADATA.into(),
                source: "handle not found".into(),
            }
        })?;
        let mut results = Vec::new();
        let iter = self
            .db
            .iterator_cf(&cf, IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Internal {
                detail: format!("metadata iteration failed: {e}"),
            })?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            let key_str = String::from_utf8(key.to_vec()).unwrap_or_default();
            results.push((key_str, value.to_vec()));
        }
        Ok(results)
    }
}

impl VectorStoreBackend for RocksVectorStore {
    fn put_vector(&self, id: u64, data: &[f32]) -> Result<(), RekhaError> {
        let key = self.encode_key(id);
        let value = vector_to_bytes(data);
        let cf = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.put_cf(&cf, key, value).map_err(|e| {
            StorageError::Write {
                source: e.to_string(),
            }
            .into()
        })
    }

    fn get_vector(&self, id: u64) -> Result<Option<Vec<f32>>, RekhaError> {
        let key = self.encode_key(id);
        let cf = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;
        match self.db.get_cf(&cf, key) {
            Ok(Some(bytes)) => Ok(Some(bytes_to_vector(&bytes))),
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::Read {
                key: id.to_be_bytes().to_vec(),
                source: e.to_string(),
            }
            .into()),
        }
    }

    fn put_payload(&self, id: u64, payload: &[u8]) -> Result<(), RekhaError> {
        if payload.len() > self.max_payload_size {
            return Err(StorageError::PayloadTooLarge {
                size: payload.len(),
                max: self.max_payload_size,
            }
            .into());
        }
        let key = self.encode_key(id);
        let cf = self.db.cf_handle(CF_PAYLOADS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.put_cf(&cf, key, payload).map_err(|e| {
            StorageError::Write {
                source: e.to_string(),
            }
            .into()
        })
    }

    fn get_payload(&self, id: u64) -> Result<Option<Vec<u8>>, RekhaError> {
        let key = self.encode_key(id);
        let cf = self.db.cf_handle(CF_PAYLOADS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            }
        })?;
        self.db.get_cf(&cf, key).map_err(|e| StorageError::Read {
            key: id.to_be_bytes().to_vec(),
            source: e.to_string(),
        }
        .into())
    }

    fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError> {
        let mut wb = crate::batch::WriteBatch::new(self);
        for id in ids {
            wb = wb.delete(*id);
        }
        wb.commit()?;
        Ok(ids.len() as u64)
    }

    fn iter_ids(&self) -> Result<Vec<u64>, RekhaError> {
        let cf = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;
        let mut ids = Vec::new();

        let prefix = self.namespace_prefix();
        let iter_mode = match &prefix {
            Some(p) => IteratorMode::From(p, rocksdb::Direction::Forward),
            None => IteratorMode::Start,
        };
        let prefix_len = prefix.as_ref().map(|p| p.len());

        let iter = self.db.iterator_cf(&cf, iter_mode);
        for result in iter {
            let (key, _) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if let Some(plen) = prefix_len {
                if key.len() < plen || &key[..plen] != prefix.as_ref().unwrap() {
                    break;
                }
            }
            if let Some(id) = self.decode_id(&key) {
                ids.push(id);
            }
        }

        ids.sort();
        ids.dedup();
        Ok(ids)
    }
}

fn vector_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|&v| v.to_le_bytes())
        .collect()
}

fn bytes_to_vector(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_store1");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let v = vec![1.0, 2.0, 3.0, 4.0];
        store.put_vector(42, &v).unwrap();
        let retrieved = store.get_vector(42).unwrap().unwrap();
        assert_eq!(v, retrieved);
    }

    #[test]
    fn payload_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_store2");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let payload = b"hello world".to_vec();
        store.put_payload(42, &payload).unwrap();
        let retrieved = store.get_payload(42).unwrap().unwrap();
        assert_eq!(retrieved, payload);
    }

    #[test]
    fn delete_vector() {
        let dir = std::env::temp_dir().join("rekha_test_delete");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_vector(1, &[1.0, 2.0]).unwrap();
        store.put_vector(2, &[3.0, 4.0]).unwrap();
        let deleted = store.delete(&[1, 2]).unwrap();
        assert_eq!(deleted, 2);
        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_vector(2).unwrap().is_none());
    }

    #[test]
    fn get_nonexistent_vector() {
        let dir = std::env::temp_dir().join("rekha_test_nonexist");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let result = store.get_vector(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn iter_ids_empty() {
        let dir = std::env::temp_dir().join("rekha_test_iter_empty");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        let ids = store.iter_ids().unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn iter_ids_after_inserts() {
        let dir = std::env::temp_dir().join("rekha_test_iter_ids");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();
        let store_ns = store.clone().with_namespace("col".into());

        store.put_vector(10, &[0.1]).unwrap();
        store_ns.put_vector(20, &[0.2]).unwrap();
        store_ns.put_vector(30, &[0.3]).unwrap();

        let mut all = store.iter_ids().unwrap();
        all.sort();
        assert_eq!(all, vec![10, 20, 30]);

        let mut ns_ids = store_ns.iter_ids().unwrap();
        ns_ids.sort();
        assert_eq!(ns_ids, vec![20, 30]);
    }

    #[test]
    fn metadata_roundtrip() {
        let dir = std::env::temp_dir().join("rekha_test_meta");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_metadata("my_key", b"my_value").unwrap();
        let val = store.get_metadata("my_key").unwrap().unwrap();
        assert_eq!(val, b"my_value");

        let missing = store.get_metadata("nonexistent").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn metadata_delete() {
        let dir = std::env::temp_dir().join("rekha_test_meta_del");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_metadata("del_key", b"data").unwrap();
        assert!(store.get_metadata("del_key").unwrap().is_some());
        store.delete_metadata("del_key").unwrap();
        assert!(store.get_metadata("del_key").unwrap().is_none());
    }

    #[test]
    fn metadata_iter_prefix() {
        let dir = std::env::temp_dir().join("rekha_test_meta_iter");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_metadata("collection:a", b"config_a").unwrap();
        store.put_metadata("collection:b", b"config_b").unwrap();
        store.put_metadata("other", b"other_data").unwrap();

        let collections = store.iter_metadata_prefix("collection:").unwrap();
        assert_eq!(collections.len(), 2);
        let keys: Vec<String> = collections.into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&"collection:a".to_string()));
        assert!(keys.contains(&"collection:b".to_string()));
    }

    #[test]
    fn metadata_overwrite() {
        let dir = std::env::temp_dir().join("rekha_test_meta_ovw");
        let _ = std::fs::remove_dir_all(&dir);
        let store = RocksVectorStore::open(&dir).unwrap();

        store.put_metadata("key", b"v1").unwrap();
        store.put_metadata("key", b"v2").unwrap();
        let val = store.get_metadata("key").unwrap().unwrap();
        assert_eq!(val, b"v2");
    }
}
