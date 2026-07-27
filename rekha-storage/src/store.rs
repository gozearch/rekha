use std::io::Cursor;
use std::sync::Arc;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use rekha_core::{IvfConfig, RekhaError, VectorRecord};
use rocksdb::{IteratorMode, DB};

use crate::hint_store::HintStore;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";
const CF_INVERTED_LISTS: &str = "inverted_lists";
const CF_HINTS: &str = "hints";

const ALL_CFS: &[&str] = &[
    CF_VECTORS,
    CF_PAYLOADS,
    CF_METADATA,
    CF_INVERTED_LISTS,
    CF_HINTS,
];

pub struct RekhaStore {
    db: Arc<DB>,
}

impl RekhaStore {
    pub fn open(path: &str) -> Result<Self, RekhaError> {
        let existing = DB::list_cf(&rocksdb::Options::default(), path).unwrap_or_default();

        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cfs: Vec<rocksdb::ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|name| rocksdb::ColumnFamilyDescriptor::new(*name, rocksdb::Options::default()))
            .collect();

        let mut db = DB::open_cf_descriptors(&opts, path, cfs)
            .map_err(|e| RekhaError::Storage(e.to_string()))?;

        if !existing.is_empty() {
            for name in ALL_CFS {
                if !existing.contains(&name.to_string()) {
                    db.create_cf(name, &rocksdb::Options::default())
                        .map_err(|e| RekhaError::Storage(e.to_string()))?;
                }
            }
        }

        Ok(RekhaStore { db: Arc::new(db) })
    }

    pub fn db(&self) -> &Arc<DB> {
        &self.db
    }

    pub fn hint_store(&self) -> HintStore {
        HintStore::new(self.db.clone())
    }

    fn cf(&self, name: &str) -> Result<&rocksdb::ColumnFamily, RekhaError> {
        self.db
            .cf_handle(name)
            .ok_or_else(|| RekhaError::Internal(format!("column family {} not found", name)))
    }

    fn vector_key(collection: &str, id: u64) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(collection.as_bytes());
        key.push(0);
        key.write_u64::<BigEndian>(id).unwrap();
        key
    }

    pub fn put_vector(
        &self,
        collection: &str,
        id: u64,
        vector: &[f32],
        timestamp: i64,
        is_tombstone: bool,
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_VECTORS)?;
        let key = Self::vector_key(collection, id);
        let mut value = Vec::new();
        value.write_i64::<LittleEndian>(timestamp).unwrap();
        value.push(if is_tombstone { 1u8 } else { 0u8 });
        for v in vector {
            value.write_f32::<LittleEndian>(*v).unwrap();
        }
        self.db
            .put_cf(cf, key, value)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn get_vector(
        &self,
        collection: &str,
        id: u64,
    ) -> Result<Option<VectorRecord>, RekhaError> {
        let cf = self.cf(CF_VECTORS)?;
        let key = Self::vector_key(collection, id);
        let opt = self
            .db
            .get_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        match opt {
            Some(data) => {
                let mut cursor = Cursor::new(&data);
                let timestamp = cursor
                    .read_i64::<LittleEndian>()
                    .map_err(|e| RekhaError::Storage(e.to_string()))?;
                let flags = cursor
                    .read_u8()
                    .map_err(|e| RekhaError::Storage(e.to_string()))?;
                let is_tombstone = flags != 0;
                let remaining = data.len() - cursor.position() as usize;
                let elem_count = remaining / 4;
                let mut vec_data = Vec::with_capacity(elem_count);
                for _ in 0..elem_count {
                    let v = cursor
                        .read_f32::<LittleEndian>()
                        .map_err(|e| RekhaError::Storage(e.to_string()))?;
                    vec_data.push(v);
                }
                Ok(Some(VectorRecord {
                    id,
                    data: vec_data,
                    timestamp,
                    is_tombstone,
                }))
            }
            None => Ok(None),
        }
    }

    pub fn delete_vector(&self, collection: &str, id: u64) -> Result<(), RekhaError> {
        let cf = self.cf(CF_VECTORS)?;
        let key = Self::vector_key(collection, id);
        self.db
            .delete_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn put_payload(&self, collection: &str, id: u64, payload: &[u8]) -> Result<(), RekhaError> {
        let cf = self.cf(CF_PAYLOADS)?;
        let key = Self::vector_key(collection, id);
        self.db
            .put_cf(cf, key, payload)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn get_payload(&self, collection: &str, id: u64) -> Result<Option<Vec<u8>>, RekhaError> {
        let cf = self.cf(CF_PAYLOADS)?;
        let key = Self::vector_key(collection, id);
        let opt = self
            .db
            .get_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        Ok(opt.map(|v| v.to_vec()))
    }

    pub fn delete_payload(&self, collection: &str, id: u64) -> Result<(), RekhaError> {
        let cf = self.cf(CF_PAYLOADS)?;
        let key = Self::vector_key(collection, id);
        self.db
            .delete_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    fn inverted_list_key(collection: &str, centroid_id: u32, id: u64) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(collection.as_bytes());
        key.push(0);
        key.write_u32::<BigEndian>(centroid_id).unwrap();
        key.write_u64::<BigEndian>(id).unwrap();
        key
    }

    fn inverted_list_prefix(collection: &str, centroid_id: u32) -> Vec<u8> {
        let mut key = Vec::new();
        key.extend_from_slice(collection.as_bytes());
        key.push(0);
        key.write_u32::<BigEndian>(centroid_id).unwrap();
        key
    }

    pub fn inverted_list_append(
        &self,
        collection: &str,
        centroid_id: u32,
        id: u64,
        pq_code: &[u8],
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_INVERTED_LISTS)?;
        let key = Self::inverted_list_key(collection, centroid_id, id);
        self.db
            .put_cf(cf, key, pq_code)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn inverted_list_scan(
        &self,
        collection: &str,
        centroid_id: u32,
    ) -> Result<Vec<(u64, Vec<u8>)>, RekhaError> {
        let cf = self.cf(CF_INVERTED_LISTS)?;
        let prefix = Self::inverted_list_prefix(collection, centroid_id);
        let iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(&prefix, rocksdb::Direction::Forward));
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(&prefix) {
                break;
            }
            let id = (&key[key.len() - 8..])
                .read_u64::<BigEndian>()
                .map_err(|e| RekhaError::Storage(e.to_string()))?;
            results.push((id, value.to_vec()));
        }
        Ok(results)
    }

    pub fn inverted_list_remove(
        &self,
        collection: &str,
        centroid_id: u32,
        id: u64,
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_INVERTED_LISTS)?;
        let key = Self::inverted_list_key(collection, centroid_id, id);
        self.db
            .delete_cf(cf, key)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn store_assignment(
        &self,
        collection: &str,
        id: u64,
        centroid_id: u32,
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:assign:{}", collection, id);
        let data = centroid_id.to_le_bytes();
        self.db
            .put_cf(cf, key.as_bytes(), data)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn load_assignment(
        &self,
        collection: &str,
        id: u64,
    ) -> Result<Option<u32>, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:assign:{}", collection, id);
        let opt = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        match opt {
            Some(data) if data.len() >= 4 => {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&data[..4]);
                Ok(Some(u32::from_le_bytes(buf)))
            }
            _ => Ok(None),
        }
    }

    pub fn delete_assignment(
        &self,
        collection: &str,
        id: u64,
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:assign:{}", collection, id);
        self.db
            .delete_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn store_centroids(
        &self,
        collection: &str,
        centroids: &[Vec<f32>],
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:centroids", collection);
        let data =
            bincode::serialize(centroids).map_err(|e| RekhaError::Serialization(e.to_string()))?;
        self.db
            .put_cf(cf, key.as_bytes(), data)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn load_centroids(&self, collection: &str) -> Result<Vec<Vec<f32>>, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:centroids", collection);
        let opt = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        match opt {
            Some(data) => {
                bincode::deserialize(&data).map_err(|e| RekhaError::Serialization(e.to_string()))
            }
            None => Err(RekhaError::NotFound("centroids not found".into())),
        }
    }

    pub fn store_pq_codebook(
        &self,
        collection: &str,
        m: usize,
        k: usize,
        sub_dim: usize,
        codebooks: &[Vec<Vec<f32>>],
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:pq_codebook", collection);
        let data = bincode::serialize(&(m, k, sub_dim, codebooks))
            .map_err(|e| RekhaError::Serialization(e.to_string()))?;
        self.db
            .put_cf(cf, key.as_bytes(), data)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    #[allow(clippy::type_complexity)]
    pub fn load_pq_codebook(
        &self,
        collection: &str,
    ) -> Result<(usize, usize, usize, Vec<Vec<Vec<f32>>>), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:pq_codebook", collection);
        let opt = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        match opt {
            Some(data) => {
                bincode::deserialize(&data).map_err(|e| RekhaError::Serialization(e.to_string()))
            }
            None => Err(RekhaError::NotFound("pq codebook not found".into())),
        }
    }

    pub fn store_collection_config(
        &self,
        collection: &str,
        config: &IvfConfig,
    ) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("collection:{}", collection);
        let data =
            serde_json::to_vec(config).map_err(|e| RekhaError::Serialization(e.to_string()))?;
        self.db
            .put_cf(cf, key.as_bytes(), data)
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }

    pub fn load_collection_config(&self, collection: &str) -> Result<IvfConfig, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("collection:{}", collection);
        let opt = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        match opt {
            Some(data) => {
                serde_json::from_slice(&data).map_err(|e| RekhaError::Serialization(e.to_string()))
            }
            None => Err(RekhaError::NotFound(format!(
                "collection {} not found",
                collection
            ))),
        }
    }

    pub fn list_collections(&self) -> Result<Vec<String>, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let prefix = b"collection:";
        let iter = self
            .db
            .iterator_cf(cf, IteratorMode::From(prefix, rocksdb::Direction::Forward));
        let mut collections = Vec::new();
        for item in iter {
            let (key, _) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix) {
                break;
            }
            if let Ok(name) = String::from_utf8(key[prefix.len()..].to_vec()) {
                collections.push(name);
            }
        }
        Ok(collections)
    }

    pub fn delete_collection_metadata(&self, collection: &str) -> Result<(), RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let prefix = format!("collection:{}", collection);
        let keys_to_delete: Vec<Vec<u8>> = self
            .db
            .iterator_cf(cf, rocksdb::IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ))
            .filter_map(|item| {
                let (key, _) = item.ok()?;
                if key.starts_with(prefix.as_bytes()) {
                    Some(key.to_vec())
                } else {
                    None
                }
            })
            .collect();
        for key in keys_to_delete {
            self.db.delete_cf(cf, key).map_err(|e| RekhaError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    pub fn increment_vector_count(&self, collection: &str) -> Result<u64, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:vector_count", collection);
        let current = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        let count = match current {
            Some(data) => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&data[..8.min(data.len())]);
                u64::from_le_bytes(buf) + 1
            }
            None => 1,
        };
        self.db
            .put_cf(cf, key.as_bytes(), count.to_le_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        Ok(count)
    }

    pub fn decrement_vector_count(&self, collection: &str) -> Result<u64, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:vector_count", collection);
        let current = self
            .db
            .get_cf(cf, key.as_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        let count = match current {
            Some(data) => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&data[..8.min(data.len())]);
                let c = u64::from_le_bytes(buf);
                if c > 0 { c - 1 } else { 0 }
            }
            None => 0,
        };
        self.db
            .put_cf(cf, key.as_bytes(), count.to_le_bytes())
            .map_err(|e| RekhaError::Storage(e.to_string()))?;
        Ok(count)
    }

    pub fn get_vector_count(&self, collection: &str) -> Result<u64, RekhaError> {
        let cf = self.cf(CF_METADATA)?;
        let key = format!("{}:vector_count", collection);
        match self.db.get_cf(cf, key.as_bytes()) {
            Ok(Some(data)) if data.len() >= 8 => {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(&data[..8]);
                Ok(u64::from_le_bytes(buf))
            }
            Ok(_) => Ok(0),
            Err(e) => Err(RekhaError::Storage(e.to_string())),
        }
    }

    pub fn count_vectors(&self, collection: &str) -> Result<u64, RekhaError> {
        let cf = self.cf(CF_VECTORS)?;
        let prefix = format!("{}\0", collection);
        let iter = self.db.iterator_cf(
            cf,
            IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward),
        );
        let mut count = 0u64;
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if value.len() >= 9 && value[8] == 0 {
                count += 1;
            }
        }
        Ok(count)
    }

    pub fn flush(&self) -> Result<(), RekhaError> {
        self.db
            .flush()
            .map_err(|e| RekhaError::Storage(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::DistanceMetric;
    use tempfile::TempDir;

    fn setup() -> (TempDir, RekhaStore) {
        let dir = TempDir::new().unwrap();
        let store = RekhaStore::open(dir.path().to_str().unwrap()).unwrap();
        (dir, store)
    }

    #[test]
    fn test_open_and_store() {
        let (_dir, store) = setup();
        store
            .put_vector("test", 1, &[0.1, 0.2, 0.3], 1000, false)
            .unwrap();
        let record = store.get_vector("test", 1).unwrap().unwrap();
        assert_eq!(record.id, 1);
        assert_eq!(record.data.len(), 3);
        assert!(!record.is_tombstone);
        assert_eq!(record.timestamp, 1000);
    }

    #[test]
    fn test_vector_tombstone() {
        let (_dir, store) = setup();
        store
            .put_vector("test", 1, &[0.1, 0.2], 1000, true)
            .unwrap();
        let record = store.get_vector("test", 1).unwrap().unwrap();
        assert!(record.is_tombstone);
    }

    #[test]
    fn test_vector_not_found() {
        let (_dir, store) = setup();
        let result = store.get_vector("test", 999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_vector() {
        let (_dir, store) = setup();
        store.put_vector("test", 1, &[0.1], 1000, false).unwrap();
        store.delete_vector("test", 1).unwrap();
        let result = store.get_vector("test", 1).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_payload_crud() {
        let (_dir, store) = setup();
        store.put_payload("test", 1, b"payload-data").unwrap();
        let payload = store.get_payload("test", 1).unwrap().unwrap();
        assert_eq!(payload, b"payload-data");

        store.delete_payload("test", 1).unwrap();
        let payload = store.get_payload("test", 1).unwrap();
        assert!(payload.is_none());
    }

    #[test]
    fn test_inverted_list() {
        let (_dir, store) = setup();
        store
            .inverted_list_append("test", 0, 1, b"\x01\x02")
            .unwrap();
        store
            .inverted_list_append("test", 0, 2, b"\x03\x04")
            .unwrap();

        let entries = store.inverted_list_scan("test", 0).unwrap();
        assert_eq!(entries.len(), 2);

        store.inverted_list_remove("test", 0, 1).unwrap();
        let entries = store.inverted_list_scan("test", 0).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, 2);
    }

    #[test]
    fn test_centroids() {
        let (_dir, store) = setup();
        let centroids = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        store.store_centroids("test", &centroids).unwrap();
        let loaded = store.load_centroids("test").unwrap();
        assert_eq!(loaded.len(), 2);
        assert!((loaded[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_collection_config() {
        let (_dir, store) = setup();
        let config = IvfConfig {
            dim: 128,
            nlist: 1024,
            nprobe: 32,
            pq_m: 16,
            pq_k: 256,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        store.store_collection_config("test", &config).unwrap();
        let loaded = store.load_collection_config("test").unwrap();
        assert_eq!(loaded.dim, 128);
        assert_eq!(loaded.nlist, 1024);
    }

    #[test]
    fn test_list_collections() {
        let (_dir, store) = setup();
        let cfg = IvfConfig::default();
        store.store_collection_config("a", &cfg).unwrap();
        store.store_collection_config("b", &cfg).unwrap();
        let names = store.list_collections().unwrap();
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
    }

    #[test]
    fn test_pq_codebook() {
        let (_dir, store) = setup();
        let codebooks = vec![vec![vec![0.1f32, 0.2f32], vec![0.3f32, 0.4f32]]];
        store
            .store_pq_codebook("test", 1, 2, 2, &codebooks)
            .unwrap();
        let (m, k, sub_dim, loaded) = store.load_pq_codebook("test").unwrap();
        assert_eq!(m, 1);
        assert_eq!(k, 2);
        assert_eq!(sub_dim, 2);
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_payload_not_found() {
        let (_dir, store) = setup();
        let result = store.get_payload("test", 999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_delete_collection_metadata() {
        let (_dir, store) = setup();
        let cfg = IvfConfig::default();
        store.store_collection_config("test", &cfg).unwrap();
        store.delete_collection_metadata("test").unwrap();
        assert!(store.load_collection_config("test").is_err());
    }
}
