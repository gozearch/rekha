use rekha_core::{RekhaError, StorageError, VectorRecord, VectorStoreBackend, now_micros};
use rocksdb::{
    ColumnFamilyDescriptor, DBWithThreadMode, IteratorMode, MultiThreaded, Options,
};
use std::path::Path;
use std::sync::Arc;

const CF_VECTORS: &str = "vectors";
const CF_PAYLOADS: &str = "payloads";
const CF_METADATA: &str = "metadata";
const CF_HINTS: &str = "hints";
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

        let required: Vec<&str> = vec![CF_VECTORS, CF_PAYLOADS, CF_METADATA, CF_HINTS];
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

    fn decode_vector_value(value: &[u8]) -> (u64, u8, &[u8]) {
        if value.len() < 9 {
            return (0, 0xFF, value);
        }
        let timestamp = u64::from_le_bytes(value[0..8].try_into().unwrap());
        let flag = value[8];
        (timestamp, flag, &value[9..])
    }

    fn encode_vector_value(timestamp: u64, flag: u8, data: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(9 + data.len() * 4);
        buf.extend_from_slice(&timestamp.to_le_bytes());
        buf.push(flag);
        for &v in data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
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

    pub fn iter_hints_for_node(&self, target_node_id: &str) -> Result<Vec<HintEntry>, RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_HINTS.into(),
                source: "handle not found".into(),
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
                source: "handle not found".into(),
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
            name: CF_HINTS.into(), source: "handle not found".into(),
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

        self.db.put_cf(&cf, key, value).map_err(|e| StorageError::Write { source: e.to_string() }.into())
    }

    pub fn iter_collection_hints_for_node(&self, target_node_id: &str) -> Result<Vec<(String, u64, u8, Vec<u8>)>, RekhaError> {
        let cf = self.db.cf_handle(CF_HINTS).ok_or_else(|| StorageError::ColumnFamily {
            name: CF_HINTS.into(), source: "handle not found".into(),
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
            name: CF_HINTS.into(), source: "handle not found".into(),
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
                source: "handle not found".into(),
            }
        })?;

        let mut count = 0u64;
        // Expire vector hints
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

        // Expire collection hints
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

    pub fn scan_tombstones(&self) -> Result<Vec<(u64, u64)>, RekhaError> {
        let cf = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;

        let mut results = Vec::new();
        let prefix = self.namespace_prefix();
        let iter_mode = match &prefix {
            Some(p) => IteratorMode::From(p, rocksdb::Direction::Forward),
            None => IteratorMode::Start,
        };
        let prefix_len = prefix.as_ref().map(|p| p.len());

        let iter = self.db.iterator_cf(&cf, iter_mode);
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if let Some(plen) = prefix_len {
                if key.len() < plen || &key[..plen] != prefix.as_ref().unwrap() {
                    break;
                }
            }
            let (ts, flag, _) = Self::decode_vector_value(&value);
            if flag == 0x01 {
                if let Some(id) = self.decode_id(&key) {
                    results.push((id, ts));
                }
            }
        }
        Ok(results)
    }

    pub fn physically_delete_vectors(&self, ids: &[u64]) -> Result<(), RekhaError> {
        let cf_v = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;
        let cf_p = self.db.cf_handle(CF_PAYLOADS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            }
        })?;
        for &id in ids {
            let key = self.encode_key(id);
            self.db.delete_cf(&cf_v, &key).map_err(|e| RekhaError::Internal {
                detail: format!("physical vector delete failed: {e}"),
            })?;
            self.db.delete_cf(&cf_p, &key).map_err(|e| RekhaError::Internal {
                detail: format!("physical payload delete failed: {e}"),
            })?;
        }
        Ok(())
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
    fn put_vector(&self, id: u64, data: &[f32], timestamp: u64) -> Result<(), RekhaError> {
        let key = self.encode_key(id);
        let value = Self::encode_vector_value(timestamp, 0x00, data);
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
            Ok(Some(bytes)) => {
                let (_ts, flag, rest) = Self::decode_vector_value(&bytes);
                if flag == 0x01 || flag == 0xFF {
                    Ok(None)
                } else {
                    let v: Vec<f32> = rest
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    Ok(Some(v))
                }
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::Read {
                key: id.to_be_bytes().to_vec(),
                source: e.to_string(),
            }
            .into()),
        }
    }

    fn get_vector_record(&self, id: u64) -> Result<Option<VectorRecord>, RekhaError> {
        let key = self.encode_key(id);
        let cf = self.db.cf_handle(CF_VECTORS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_VECTORS.into(),
                source: "handle not found".into(),
            }
        })?;
        match self.db.get_cf(&cf, key) {
            Ok(Some(bytes)) => {
                let (timestamp, flag, rest) = Self::decode_vector_value(&bytes);
                if flag == 0x01 {
                    Ok(Some(VectorRecord {
                        id,
                        timestamp,
                        data: None,
                        is_tombstone: true,
                    }))
                } else if flag == 0x00 {
                    let data: Vec<f32> = rest
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    Ok(Some(VectorRecord {
                        id,
                        timestamp,
                        data: Some(data),
                        is_tombstone: false,
                    }))
                } else {
                    Ok(None)
                }
            }
            Ok(None) => Ok(None),
            Err(e) => Err(StorageError::Read {
                key: id.to_be_bytes().to_vec(),
                source: e.to_string(),
            }
            .into()),
        }
    }

    fn put_tombstone(&self, id: u64, timestamp: u64) -> Result<(), RekhaError> {
        let key = self.encode_key(id);
        let value = Self::encode_vector_value(timestamp, 0x01, &[]);
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
        let timestamp = now_micros();
        let cf_p = self.db.cf_handle(CF_PAYLOADS).ok_or_else(|| {
            StorageError::ColumnFamily {
                name: CF_PAYLOADS.into(),
                source: "handle not found".into(),
            }
        })?;
        for &id in ids {
            self.put_tombstone(id, timestamp)?;
            let key = self.encode_key(id);
            self.db.delete_cf(&cf_p, &key).map_err(|e| {
                RekhaError::Internal {
                    detail: format!("payload delete failed: {e}"),
                }
            })?;
        }
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
            let (key, value) = result.map_err(|e| RekhaError::Internal {
                detail: format!("db iteration error: {e}"),
            })?;
            if let Some(plen) = prefix_len {
                if key.len() < plen || &key[..plen] != prefix.as_ref().unwrap() {
                    break;
                }
            }
            let (_ts, flag, _) = Self::decode_vector_value(&value);
            if flag == 0x01 {
                continue;
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn setup_store(name: &str) -> RocksVectorStore {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("{}_{}", name, id));
        let _ = std::fs::remove_dir_all(&dir);
        RocksVectorStore::open(&dir).unwrap()
    }

    #[test]
    fn vector_roundtrip() {
        let store = setup_store("rekha_test_store1");
        let v = vec![1.0, 2.0, 3.0, 4.0];
        store.put_vector(42, &v, 100).unwrap();
        let retrieved = store.get_vector(42).unwrap().unwrap();
        assert_eq!(v, retrieved);
    }

    #[test]
    fn payload_roundtrip() {
        let store = setup_store("rekha_test_store2");
        let payload = b"hello world".to_vec();
        store.put_payload(42, &payload).unwrap();
        let retrieved = store.get_payload(42).unwrap().unwrap();
        assert_eq!(retrieved, payload);
    }

    #[test]
    fn delete_vector() {
        let store = setup_store("rekha_test_delete");
        store.put_vector(1, &[1.0, 2.0], 100).unwrap();
        store.put_vector(2, &[3.0, 4.0], 100).unwrap();
        let deleted = store.delete(&[1, 2]).unwrap();
        assert_eq!(deleted, 2);
        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_vector(2).unwrap().is_none());
    }

    #[test]
    fn get_nonexistent_vector() {
        let store = setup_store("rekha_test_nonexist");
        let result = store.get_vector(999).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn iter_ids_empty() {
        let store = setup_store("rekha_test_iter_empty");
        let ids = store.iter_ids().unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn iter_ids_after_inserts() {
        let store = setup_store("rekha_test_iter_ids");
        let store_ns = store.clone().with_namespace("col".into());

        store.put_vector(10, &[0.1], 100).unwrap();
        store_ns.put_vector(20, &[0.2], 100).unwrap();
        store_ns.put_vector(30, &[0.3], 100).unwrap();

        let mut all = store.iter_ids().unwrap();
        all.sort();
        assert_eq!(all, vec![10, 20, 30]);

        let mut ns_ids = store_ns.iter_ids().unwrap();
        ns_ids.sort();
        assert_eq!(ns_ids, vec![20, 30]);
    }

    #[test]
    fn metadata_roundtrip() {
        let store = setup_store("rekha_test_meta");
        store.put_metadata("my_key", b"my_value").unwrap();
        let val = store.get_metadata("my_key").unwrap().unwrap();
        assert_eq!(val, b"my_value");

        let missing = store.get_metadata("nonexistent").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn metadata_delete() {
        let store = setup_store("rekha_test_meta_del");
        store.put_metadata("del_key", b"data").unwrap();
        assert!(store.get_metadata("del_key").unwrap().is_some());
        store.delete_metadata("del_key").unwrap();
        assert!(store.get_metadata("del_key").unwrap().is_none());
    }

    #[test]
    fn metadata_iter_prefix() {
        let store = setup_store("rekha_test_meta_iter");

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
        let store = setup_store("rekha_test_meta_ovw");
        store.put_metadata("key", b"v1").unwrap();
        store.put_metadata("key", b"v2").unwrap();
        let val = store.get_metadata("key").unwrap().unwrap();
        assert_eq!(val, b"v2");
    }

    #[test]
    fn test_tombstone_write_and_read() {
        let store = setup_store("rekha_test_tombstone_rw");
        store.put_vector(1, &[1.0, 2.0, 3.0], 100).unwrap();
        store.put_tombstone(1, 200).unwrap();
        assert!(store.get_vector(1).unwrap().is_none());
        let rec = store.get_vector_record(1).unwrap().unwrap();
        assert!(rec.is_tombstone);
        assert!(rec.data.is_none());
        assert_eq!(rec.timestamp, 200);
    }

    #[test]
    fn test_timestamped_vector() {
        let store = setup_store("rekha_test_ts_vec");
        store.put_vector(1, &[1.0, 2.0], 42).unwrap();
        let rec = store.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 42);
        assert!(!rec.is_tombstone);
        assert_eq!(rec.data, Some(vec![1.0, 2.0]));
    }

    #[test]
    fn test_timestamp_lww() {
        let store = setup_store("rekha_test_ts_lww");
        store.put_vector(1, &[1.0, 2.0], 100).unwrap();
        store.put_vector(1, &[3.0, 4.0], 50).unwrap();
        let rec = store.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 50);
        assert_eq!(rec.data, Some(vec![3.0, 4.0]));
    }

    #[test]
    fn test_iter_ids_skips_tombstones() {
        let store = setup_store("rekha_test_iter_skip_ts");
        store.put_vector(1, &[1.0], 100).unwrap();
        store.put_vector(2, &[2.0], 200).unwrap();
        store.put_tombstone(2, 300).unwrap();
        store.put_vector(3, &[3.0], 400).unwrap();
        store.put_tombstone(1, 500).unwrap();
        let mut ids = store.iter_ids().unwrap();
        ids.sort();
        assert_eq!(ids, vec![3]);
    }

    #[test]
    fn test_hint_roundtrip() {
        let store = setup_store("rekha_test_hint_rt");
        store.put_hint("node2", "col1", 1, &[1.0, 2.0], Some(b"payload"), 100).unwrap();

        let hints = store.iter_hints_for_node("node2").unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].target_node_id, "node2");
        assert_eq!(hints[0].collection, "col1");
        assert_eq!(hints[0].id, 1);
        assert_eq!(hints[0].vector, vec![1.0, 2.0]);
        assert_eq!(hints[0].payload, Some(b"payload".to_vec()));
        assert_eq!(hints[0].timestamp, 100);

        store.delete_hint("node2", "col1", 1).unwrap();
        let hints = store.iter_hints_for_node("node2").unwrap();
        assert!(hints.is_empty());
    }

    #[test]
    fn test_scavenge_expired_hints() {
        let store = setup_store("rekha_test_scavenge");
        let old_ts = 1000u64;
        let recent_ts = 9_999_999_999_999_999u64;

        store.put_hint("node2", "col1", 1, &[1.0], None, old_ts).unwrap();
        store.put_hint("node2", "col1", 2, &[2.0], None, recent_ts).unwrap();

        let deleted = store.delete_expired_hints(100_000_000).unwrap();
        assert_eq!(deleted, 1);

        let hints = store.iter_hints_for_node("node2").unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].id, 2);
    }

    #[test]
    fn test_scan_tombstones() {
        let store = setup_store("rekha_test_scan_ts");
        store.put_vector(1, &[1.0], 100).unwrap();
        store.put_tombstone(2, 200).unwrap();
        store.put_vector(3, &[3.0], 300).unwrap();
        store.put_tombstone(4, 400).unwrap();

        let tombstones = store.scan_tombstones().unwrap();
        let mut pairs: Vec<(u64, u64)> = tombstones;
        pairs.sort();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (2, 200));
        assert_eq!(pairs[1], (4, 400));
    }

    #[test]
    fn test_physically_delete_vectors() {
        let store = setup_store("rekha_test_phys_del");
        store.put_vector(1, &[1.0], 100).unwrap();
        store.put_payload(1, b"data").unwrap();
        store.physically_delete_vectors(&[1]).unwrap();
        assert!(store.get_vector(1).unwrap().is_none());
        assert!(store.get_payload(1).unwrap().is_none());
    }

    #[test]
    fn test_decode_vector_value_short() {
        let (_ts, flag, _rest) = RocksVectorStore::decode_vector_value(&[0u8; 4]);
        assert_eq!(flag, 0xFF);
        let (_ts2, flag2, _rest2) = RocksVectorStore::decode_vector_value(&[0u8; 9]);
        assert_eq!(flag2, 0x00);
    }

    #[test]
    fn test_hint_isolation() {
        let store = setup_store("rekha_test_hint_iso");
        store.put_hint("node-a", "c1", 1, &[1.0], None, 10).unwrap();
        store.put_hint("node-b", "c1", 2, &[2.0], None, 20).unwrap();

        let hints_a = store.iter_hints_for_node("node-a").unwrap();
        assert_eq!(hints_a.len(), 1);
        assert_eq!(hints_a[0].id, 1);

        let hints_b = store.iter_hints_for_node("node-b").unwrap();
        assert_eq!(hints_b.len(), 1);
        assert_eq!(hints_b[0].id, 2);
    }

    #[test]
    fn test_collection_hint_roundtrip() {
        let store = setup_store("rekha_test_coll_hint_rt");
        store.put_collection_hint("node2", "images", b"{\"dim\":256}", 100, 0).unwrap();

        let hints = store.iter_collection_hints_for_node("node2").unwrap();
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].0, "images");
        assert_eq!(hints[0].1, 100);
        assert_eq!(hints[0].2, 0);
        assert_eq!(hints[0].3, b"{\"dim\":256}");

        store.delete_collection_hint("node2", "images").unwrap();
        let hints = store.iter_collection_hints_for_node("node2").unwrap();
        assert!(hints.is_empty());
    }

    #[test]
    fn test_collection_hint_isolation() {
        let store = setup_store("rekha_test_coll_hint_iso");
        store.put_collection_hint("node-a", "c1", b"config1", 10, 0).unwrap();
        store.put_collection_hint("node-b", "c2", b"config2", 20, 1).unwrap();

        let hints_a = store.iter_collection_hints_for_node("node-a").unwrap();
        assert_eq!(hints_a.len(), 1);
        assert_eq!(hints_a[0].0, "c1");
        assert_eq!(hints_a[0].2, 0);

        let hints_b = store.iter_collection_hints_for_node("node-b").unwrap();
        assert_eq!(hints_b.len(), 1);
        assert_eq!(hints_b[0].0, "c2");
        assert_eq!(hints_b[0].2, 1);
    }
}
