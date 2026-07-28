use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use rekha_cluster::chord::ChordNode;
use rekha_cluster::hash_to_chord_id;
use rekha_cluster::Membership;
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, RekhaError, ScoredPoint, SearchParams};
use rekha_index::DiskIvfIndex;
use rekha_proto::proto;
use rekha_replication::{ConsistencyGate, HintedHandoff, LwwTimestamp};
use rekha_storage::RekhaStore;
use tokio::sync::{RwLock, Semaphore};

use crate::peer_pool::PeerPool;

pub enum IndexState {
    Pending { config: IvfConfig },
    Trained(DiskIvfIndex),
}

pub struct Coordinator {
    pub store: Arc<RekhaStore>,
    pub indexes: Arc<RwLock<HashMap<String, IndexState>>>,
    pub membership: Arc<RwLock<Membership>>,
    pub hinted_handoff: HintedHandoff,
    pub node_id: u64,
    pub default_write_consistency: ConsistencyLevel,
    pub default_rf: u32,
    pub node_id_str: String,
    pub chord: Arc<ChordNode>,
    pub peer_pool: Arc<PeerPool>,
    pub gc_grace_seconds: i64,
    pub concurrency_limit: Arc<Semaphore>,
}

impl Coordinator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: Arc<RekhaStore>,
        membership: Arc<RwLock<Membership>>,
        node_id: u64,
        node_id_str: String,
        hh_enabled: bool,
        max_hint_window: i64,
        default_write_consistency: ConsistencyLevel,
        default_rf: u32,
        chord: Arc<ChordNode>,
        peer_pool: Arc<PeerPool>,
        gc_grace_seconds: i64,
    ) -> Self {
        let hint_store = store.hint_store();
        let hinted_handoff = HintedHandoff::new(hint_store, hh_enabled, max_hint_window);

        Coordinator {
            store,
            indexes: Arc::new(RwLock::new(HashMap::new())),
            membership,
            hinted_handoff,
            node_id,
            default_write_consistency,
            default_rf,
            node_id_str,
            chord,
            peer_pool,
            gc_grace_seconds,
            concurrency_limit: Arc::new(Semaphore::new(1024)),
        }
    }

    pub async fn initialize(&self) -> Result<(), RekhaError> {
        let collection_names = self.store.list_collections()?;
        if collection_names.is_empty() {
            let default_config = IvfConfig {
                dim: 8,
                nlist: 256,
                nprobe: 16,
                pq_m: 4,
                pq_k: 256,
                replication_factor: 3,
                distance_metric: DistanceMetric::L2,
            };
            self.store
                .store_collection_config("default", &default_config)?;
            let mut indexes = self.indexes.write().await;
            indexes.insert(
                "default".to_string(),
                IndexState::Pending {
                    config: default_config,
                },
            );
        } else {
            let mut indexes = self.indexes.write().await;
            for name in &collection_names {
                let config = self.store.load_collection_config(name)?;
                let mut index = DiskIvfIndex::new(name, self.store.clone(), config);
                match index.load_from_store() {
                    Ok(_) => {
                        indexes.insert(name.clone(), IndexState::Trained(index));
                    }
                    Err(_) => {
                        indexes.insert(
                            name.clone(),
                            IndexState::Pending { config },
                        );
                    }
                }
            }
        }
        Ok(())
    }

    pub async fn create_collection(
        &self,
        name: &str,
        config: IvfConfig,
        _origin_node_id: &str,
        timestamp: i64,
        _consistency: ConsistencyLevel,
        is_replication: bool,
    ) -> Result<(), RekhaError> {
        {
            let mut indexes = self.indexes.write().await;
            if indexes.contains_key(name) {
                return Err(RekhaError::InvalidArgument(format!(
                    "collection {} already exists",
                    name
                )));
            }
            if is_replication {
                if let Ok(_existing_meta) = self.store.load_collection_config(name) {
                    return Ok(());
                }
            }
            self.store.store_collection_config(name, &config)?;
            indexes.insert(name.to_string(), IndexState::Pending { config });
        }

        if !is_replication {
            let name_hash = hash_to_chord_id(name.as_bytes());
            let replicas = self.chord.replicas_for_chord_id(name_hash, self.default_rf as usize).await;
            for replica in &replicas {
                if replica.address == self.chord.address { continue; }
                if replica.address.is_empty() { continue; }
                let req = proto::CreateCollectionRequest {
                    name: name.to_string(),
                    config: Some(config.into()),
                    is_replication: true,
                    timestamp: timestamp as u64,
                    consistency: proto::ConsistencyLevel::Quorum as i32,
                    origin_node_id: self.node_id_str.clone(),
                };
                let _ = self.peer_pool.replica_create_collection(&replica.address, req).await;
            }
        }

        Ok(())
    }

    pub async fn auto_train(&self, collection: &str) -> Result<bool, RekhaError> {
        let config = {
            let indexes = self.indexes.read().await;
            match indexes.get(collection) {
                Some(IndexState::Pending { config }) => *config,
                _ => return Ok(false),
            }
        };

        let total = self.store.get_vector_count(collection)?;
        let threshold = (config.nlist * 2) as u64;

        if total < threshold {
            return Ok(false);
        }

        let mut sample = Vec::new();
        let cf = self
            .store
            .db()
            .cf_handle("vectors")
            .ok_or_else(|| RekhaError::Internal("vectors cf not found".into()))?;

        let prefix = format!("{}\0", collection);
        use rocksdb::IteratorMode;
        let iter = self
            .store
            .db()
            .iterator_cf(cf, IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if value.len() < 9 {
                continue;
            }
            if value[8] != 0 {
                continue;
            }
            let f32_count = (value.len() - 9) / 4;
            let mut vec_data = Vec::with_capacity(f32_count);
            for i in 0..f32_count {
                let start = 9 + i * 4;
                let bytes: [u8; 4] = [
                    value[start],
                    value[start + 1],
                    value[start + 2],
                    value[start + 3],
                ];
                vec_data.push(f32::from_le_bytes(bytes));
            }
            sample.push(vec_data);
            if sample.len() >= threshold as usize {
                break;
            }
        }

        if sample.len() < config.nlist as usize {
            return Ok(false);
        }

        let mut index = DiskIvfIndex::new(collection, self.store.clone(), config);
        index.build(&sample)?;

        let mut indexes = self.indexes.write().await;
        if let Some(IndexState::Pending { .. }) = indexes.get(collection) {
            indexes.insert(collection.to_string(), IndexState::Trained(index));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn drop_collection(
        &self,
        name: &str,
        _origin_node_id: &str,
        timestamp: i64,
        _consistency: ConsistencyLevel,
        is_replication: bool,
    ) -> Result<(), RekhaError> {
        {
            let mut indexes = self.indexes.write().await;
            indexes.remove(name);
            self.store.delete_collection_metadata(name)?;
        }

        if !is_replication {
            let name_hash = hash_to_chord_id(name.as_bytes());
            let replicas = self.chord.replicas_for_chord_id(name_hash, self.default_rf as usize).await;
            for replica in &replicas {
                if replica.address == self.chord.address { continue; }
                if replica.address.is_empty() { continue; }
                let req = proto::DropCollectionRequest {
                    name: name.to_string(),
                    is_replication: true,
                    timestamp: timestamp as u64,
                    consistency: proto::ConsistencyLevel::Quorum as i32,
                    origin_node_id: self.node_id_str.clone(),
                };
                let _ = self.peer_pool.replica_drop_collection(&replica.address, req).await;
            }
        }

        Ok(())
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, RekhaError> {
        self.store.list_collections()
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool, RekhaError> {
        let indexes = self.indexes.read().await;
        Ok(indexes.contains_key(name))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn insert(
        &self,
        collection: &str,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
        timestamp: i64,
        origin_node_id: &str,
        consistency: ConsistencyLevel,
        is_replication: bool,
    ) -> Result<u64, RekhaError> {
        let _permit = self.concurrency_limit.clone().acquire_owned().await.map_err(|_| RekhaError::Internal("concurrency limit closed".into()))?;
        let dim = {
            let indexes = self.indexes.read().await;
            match indexes.get(collection) {
                Some(IndexState::Pending { config }) => config.dim,
                Some(IndexState::Trained(idx)) => idx.config().dim,
                None => {
                    return Err(RekhaError::NotFound(format!(
                        "collection {} not found",
                        collection
                    )))
                }
            }
        };

        if vector.len() != dim as usize {
            return Err(RekhaError::InvalidArgument(format!(
                "expected dimension {}, got {}",
                dim,
                vector.len()
            )));
        }

        // 1. LWW check: don't overwrite if existing record has newer timestamp
        let origin_hash = hash_to_chord_id(origin_node_id.as_bytes()) as u64;
        if let Ok(Some(existing)) = self.store.get_vector(collection, id) {
            let keep_local = LwwTimestamp::should_keep_local(
                timestamp,
                existing.timestamp,
                origin_hash,
                existing.id,
            );
            if !keep_local {
                return Ok(0u64);
            }
        }

        // 2. Write local
        self.store
            .put_vector(collection, id, &vector, timestamp, false)?;
        self.store.increment_vector_count(collection)?;
        if let Some(p) = &payload {
            self.store.put_payload(collection, id, p)?;
        }
        {
            let indexes = self.indexes.read().await;
            if let Some(IndexState::Trained(idx)) = indexes.get(collection) {
                let _ = idx.add(id, &vector);
            }
        }

        // 3. If client-originated, forward to replicas
        if !is_replication {
            let rf = self.default_rf as usize;
            let vector_hash = hash_to_chord_id(&id.to_le_bytes());
            let replicas = self.chord.replicas_for_chord_id(vector_hash, rf).await;

            let mut acks = 1u64;
            let needed = ConsistencyGate::required_acks(rf.clamp(1, 3), consistency) as u64;

            for replica in &replicas {
                if replica.address == self.chord.address {
                    continue;
                }
                let req = proto::InsertRequest {
                    id,
                    vector: vector.clone(),
                    payload: payload.as_ref().map(|p| proto::Payload {
                        content_type: "application/octet-stream".into(),
                        data: p.clone(),
                    }),
                    collection_name: collection.to_string(),
                    is_replication: true,
                    timestamp: timestamp as u64,
                    consistency: proto::ConsistencyLevel::Quorum as i32,
                    origin_node_id: self.node_id_str.clone(),
                };
                match self.peer_pool.replica_insert(&replica.address, req).await {
                    Ok(resp) if resp.success => acks += 1,
                    Ok(_) => {
                        let hint_data = serde_json::to_vec(&(id, &vector, &payload, timestamp))
                            .map_err(|e| RekhaError::Serialization(e.to_string()))?;
                        let _ = self.hinted_handoff.store_hint(
                            &replica.address, collection, id, &hint_data, timestamp,
                        );
                    }
                    Err(_) => {
                        let hint_data = serde_json::to_vec(&(id, &vector, &payload, timestamp))
                            .map_err(|e| RekhaError::Serialization(e.to_string()))?;
                        let _ = self.hinted_handoff.store_hint(
                            &replica.address, collection, id, &hint_data, timestamp,
                        );
                    }
                }
            }

            if acks < needed {
                return Err(RekhaError::Unavailable(format!(
                    "consistency not met: needed {} acks, got {}",
                    needed, acks
                )));
            }
        }

        // 4. Auto-train if needed
        let _ = self.auto_train(collection).await;

        Ok(1u64)
    }

    fn linear_search(
        store: &RekhaStore,
        collection: &str,
        query: &[f32],
        k: u32,
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let k = k as usize;
        let mut heap: BinaryHeap<ScoredPoint> = BinaryHeap::with_capacity(k + 1);

        let cf = store
            .db()
            .cf_handle("vectors")
            .ok_or_else(|| RekhaError::Internal("vectors cf not found".into()))?;

        let prefix = format!("{}\0", collection);
        use rocksdb::IteratorMode;
        let iter = store
            .db()
            .iterator_cf(cf, IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));

        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if value.len() < 9 {
                continue;
            }
            if value[8] != 0 {
                continue;
            }
            let id_start = key.len() - 8;
            let vid = u64::from_be_bytes([
                key[id_start],
                key[id_start + 1],
                key[id_start + 2],
                key[id_start + 3],
                key[id_start + 4],
                key[id_start + 5],
                key[id_start + 6],
                key[id_start + 7],
            ]);

            let f32_count = (value.len() - 9) / 4;
            let mut vec_data = Vec::with_capacity(f32_count);
            for i in 0..f32_count {
                let start = 9 + i * 4;
                let bytes: [u8; 4] = [
                    value[start],
                    value[start + 1],
                    value[start + 2],
                    value[start + 3],
                ];
                vec_data.push(f32::from_le_bytes(bytes));
            }

            let dist: f32 = query
                .iter()
                .zip(vec_data.iter())
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();

            let payload = if include_payloads {
                store.get_payload(collection, vid).ok().flatten()
            } else {
                None
            };

            heap.push(ScoredPoint {
                id: vid,
                score: dist,
                payload,
                timestamp: 0,
            });

            if heap.len() > k {
                heap.pop();
            }
        }

        let mut results: Vec<ScoredPoint> = Vec::with_capacity(k);
        for sp in heap.into_sorted_vec() {
            results.push(sp);
            if results.len() >= k {
                break;
            }
        }

        Ok(results)
    }

    pub async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        k: u32,
        params: SearchParams,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let _permit = self.concurrency_limit.clone().acquire_owned().await.map_err(|_| RekhaError::Internal("concurrency limit closed".into()))?;
        // 1. Local search
        let mut all_results: Vec<ScoredPoint> = {
            let indexes = self.indexes.read().await;
            match indexes.get(collection) {
                Some(IndexState::Trained(idx)) => {
                    let search_params = SearchParams {
                        nprobe: if params.nprobe > 0 { params.nprobe } else { idx.config().nprobe },
                        k,
                        include_payloads: params.include_payloads,
                        pre_filter: params.pre_filter.clone(),
                        local_only: true,
                    };
                    idx.search(&query, &search_params)?
                }
                Some(IndexState::Pending { .. }) => {
                    drop(indexes);
                    Self::linear_search(&self.store, collection, &query, k, params.include_payloads)?
                }
                None => return Err(RekhaError::NotFound(format!("collection {} not found", collection))),
            }
        };

        // 2. Fan out to replicas if not local_only
        if !params.local_only {
            let rf = self.default_rf as usize;
            let query_bytes: Vec<u8> = query.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let query_hash = hash_to_chord_id(&query_bytes);
            let replicas = self.chord.replicas_for_chord_id(query_hash, rf).await;

            for replica in &replicas {
                if replica.address == self.chord.address { continue; }
                if replica.address.is_empty() { continue; }

                let req = proto::SearchRequest {
                    query_vector: query.clone(),
                    top_k: k,
                    params: Some(proto::SearchParams {
                        ef_search: 0,
                        nprobe: params.nprobe,
                        include_payloads: params.include_payloads,
                    }),
                    local_only: true,
                    collection_name: collection.to_string(),
                    consistency: proto::ConsistencyLevel::One as i32,
                };

                match self.peer_pool.remote_search(&replica.address, req).await {
                    Ok(resp) => {
                        for sp in resp.results {
                            all_results.push(ScoredPoint {
                                id: sp.id,
                                score: sp.score,
                                payload: sp.payload.map(|p| p.data),
                                timestamp: sp.timestamp as i64,
                            });
                        }
                    }
                    Err(e) => {
                        tracing::warn!("remote search to {} failed: {}", replica.address, e);
                    }
                }
            }
        }

        // 3. Merge: deduplicate by ID with LWW, keep best score
        let mut merged: HashMap<u64, ScoredPoint> = HashMap::new();
        for sp in all_results {
            match merged.get(&sp.id) {
                Some(existing) => {
                    if sp.timestamp > existing.timestamp {
                        merged.insert(sp.id, sp);
                    }
                }
                None => {
                    merged.insert(sp.id, sp);
                }
            }
        }

        // 4. Sort by score (lower is better for L2) and return top-k
        let mut results: Vec<ScoredPoint> = merged.into_values().collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k as usize);
        Ok(results)
    }

    pub async fn delete(
        &self,
        collection: &str,
        ids: &[u64],
        timestamp: i64,
        origin_node_id: &str,
        _consistency: ConsistencyLevel,
        is_replication: bool,
    ) -> Result<u64, RekhaError> {
        let mut deleted = 0u64;
        for &id in ids {
            // 1. LWW check
            let origin_hash = hash_to_chord_id(origin_node_id.as_bytes()) as u64;
            if let Ok(Some(existing)) = self.store.get_vector(collection, id) {
                let keep_local = LwwTimestamp::should_keep_local(
                    timestamp,
                    existing.timestamp,
                    origin_hash,
                    existing.id,
                );
                if !keep_local {
                    continue;
                }
            }

            // 2. Write tombstone locally
            self.store
                .put_vector(collection, id, &[], timestamp, true)?;
            let _ = self.store.decrement_vector_count(collection);
            self.store.delete_payload(collection, id)?;
            {
                let indexes = self.indexes.read().await;
                if let Some(IndexState::Trained(idx)) = indexes.get(collection) {
                    let _ = idx.remove(id);
                }
            }
            deleted += 1;

            // 3. Forward to replicas if client-originated
            if !is_replication {
                let rf = self.default_rf as usize;
                let vector_hash = hash_to_chord_id(&id.to_le_bytes());
                let replicas = self.chord.replicas_for_chord_id(vector_hash, rf).await;
            for replica in &replicas {
                if replica.node_id == self.chord.self_id_string { continue; }
                if replica.address == self.chord.address { continue; }
                if replica.address.is_empty() { continue; }
                    let req = proto::DeleteRequest {
                        ids: vec![id],
                        collection_name: collection.to_string(),
                        timestamp: timestamp as u64,
                        consistency: proto::ConsistencyLevel::Quorum as i32,
                        is_replication: true,
                        origin_node_id: self.node_id_str.clone(),
                    };
                    let _ = self.peer_pool.replica_delete(&replica.address, req).await;
                }
            }
        }
        Ok(deleted)
    }

    pub async fn fetch(
        &self,
        collection: &str,
        ids: &[u64],
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let mut results = Vec::new();
        for &id in ids {
            if let Ok(Some(record)) = self.store.get_vector(collection, id) {
                if !record.is_tombstone {
                    let payload = if include_payloads {
                        self.store.get_payload(collection, id).ok().flatten()
                    } else {
                        None
                    };
                    results.push(ScoredPoint {
                        id,
                        score: 0.0,
                        payload,
                        timestamp: record.timestamp,
                    });
                }
            }
        }
        Ok(results)
    }

    pub async fn rebuild_index(&self, collection: &str) -> Result<(), RekhaError> {
        let config = {
            let mut indexes = self.indexes.write().await;
            match indexes.remove(collection) {
                Some(IndexState::Trained(mut idx)) => {
                    idx.rebuild()?;
                    indexes.insert(collection.to_string(), IndexState::Trained(idx));
                    return Ok(());
                }
                Some(IndexState::Pending { config }) => config,
                None => {
                    return Err(RekhaError::NotFound(format!(
                        "collection {} not found",
                        collection
                    )))
                }
            }
        };

        let total = self.store.get_vector_count(collection)?;
        if total < config.nlist as u64 * 2 {
            return Err(RekhaError::Index(format!(
                "not enough vectors to build index: need {}, have {}",
                config.nlist * 2,
                total
            )));
        }

        self.auto_train(collection).await?;
        Ok(())
    }

    pub async fn gc_collection(&self, collection: &str) -> Result<u64, RekhaError> {
        let mut gc_count = 0u64;
        let cf = self
            .store
            .db()
            .cf_handle("vectors")
            .ok_or_else(|| RekhaError::Internal("vectors cf not found".into()))?;
        let prefix = format!("{}\0", collection);
        use rocksdb::IteratorMode;
        let iter = self
            .store
            .db()
            .iterator_cf(cf, IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
        let mut to_delete = Vec::new();
        let grace_secs = self.gc_grace_seconds;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if value.len() >= 9 && value[8] != 0 {
                let ts_bytes: [u8; 8] = [
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ];
                let ts = i64::from_le_bytes(ts_bytes);
                if now - ts >= grace_secs {
                    let id_start = key.len() - 8;
                    let vid = u64::from_be_bytes([
                        key[id_start],
                        key[id_start + 1],
                        key[id_start + 2],
                        key[id_start + 3],
                        key[id_start + 4],
                        key[id_start + 5],
                        key[id_start + 6],
                        key[id_start + 7],
                    ]);
                    to_delete.push(vid);
                }
            }
        }
        let indexes = self.indexes.read().await;
        for &vid in &to_delete {
            if let Some(IndexState::Trained(idx)) = indexes.get(collection) {
                let _ = idx.remove(vid);
            }
            self.store.delete_vector(collection, vid)?;
            self.store.delete_payload(collection, vid)?;
            gc_count += 1;
        }
        Ok(gc_count)
    }

    pub async fn export_collection(
        &self,
        collection: &str,
        offset: u64,
        limit: u64,
        include_vectors: bool,
        include_payloads: bool,
    ) -> Result<Vec<rekha_core::ExportedVector>, RekhaError> {
        let cf = self.store.db().cf_handle("vectors")
            .ok_or_else(|| RekhaError::Internal("vectors cf not found".into()))?;
        let prefix = format!("{}\0", collection);
        use rocksdb::IteratorMode;
        let iter = self.store.db().iterator_cf(cf, IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        let mut results = Vec::new();
        let mut current_offset = 0u64;

        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) {
                break;
            }
            if value.len() < 9 {
                continue;
            }

            if value[8] != 0 {
                continue;
            }

            if current_offset < offset {
                current_offset += 1;
                continue;
            }

            let id_start = key.len() - 8;
            let vid = u64::from_be_bytes([
                key[id_start], key[id_start+1], key[id_start+2], key[id_start+3],
                key[id_start+4], key[id_start+5], key[id_start+6], key[id_start+7],
            ]);

            let ts_bytes: [u8; 8] = [
                value[0], value[1], value[2], value[3],
                value[4], value[5], value[6], value[7],
            ];
            let timestamp = i64::from_le_bytes(ts_bytes);

            let vector = if include_vectors {
                let f32_count = (value.len() - 9) / 4;
                let mut vec_data = Vec::with_capacity(f32_count);
                for i in 0..f32_count {
                    let start = 9 + i * 4;
                    let bytes: [u8; 4] = [
                        value[start], value[start+1], value[start+2], value[start+3],
                    ];
                    vec_data.push(f32::from_le_bytes(bytes));
                }
                vec_data
            } else {
                Vec::new()
            };

            let payload = if include_payloads {
                self.store.get_payload(collection, vid).ok().flatten()
            } else {
                None
            };

            results.push(rekha_core::ExportedVector {
                id: vid,
                vector,
                payload,
                timestamp,
            });

            if results.len() >= limit as usize {
                break;
            }
        }

        Ok(results)
    }

    pub fn export_stream(
        &self,
        collection: &str,
        offset: u64,
        limit: u64,
        include_vectors: bool,
        include_payloads: bool,
        batch_size: usize,
    ) -> tokio::sync::mpsc::Receiver<Result<Vec<rekha_core::ExportedVector>, RekhaError>> {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let store_ref = self.store.clone();
        let collection = collection.to_string();

        tokio::task::spawn_blocking(move || {
            let cf = match store_ref.db().cf_handle("vectors") {
                Some(cf) => cf,
                None => {
                    let _ = tx.blocking_send(Err(RekhaError::Internal("vectors cf not found".into())));
                    return;
                }
            };
            let prefix = format!("{}\0", collection);
            use rocksdb::IteratorMode;
            let iter = store_ref.db().iterator_cf(cf, IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            let mut batch = Vec::new();
            let mut current_offset = 0u64;
            let mut total_sent = 0u64;

            for item in iter {
                let (key, value) = match item {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(RekhaError::Storage(e.to_string())));
                        return;
                    }
                };
                if !key.starts_with(prefix.as_bytes()) { break; }
                if value.len() < 9 || value[8] != 0 { continue; }
                if current_offset < offset {
                    current_offset += 1;
                    continue;
                }

                let id_start = key.len() - 8;
                let vid = u64::from_be_bytes([
                    key[id_start], key[id_start+1], key[id_start+2], key[id_start+3],
                    key[id_start+4], key[id_start+5], key[id_start+6], key[id_start+7],
                ]);

                let ts_bytes: [u8; 8] = [
                    value[0], value[1], value[2], value[3],
                    value[4], value[5], value[6], value[7],
                ];
                let timestamp = i64::from_le_bytes(ts_bytes);

                let vector = if include_vectors {
                    let f32_count = (value.len() - 9) / 4;
                    let mut vec_data = Vec::with_capacity(f32_count);
                    for i in 0..f32_count {
                        let start = 9 + i * 4;
                        let bytes: [u8; 4] = [
                            value[start], value[start+1], value[start+2], value[start+3],
                        ];
                        vec_data.push(f32::from_le_bytes(bytes));
                    }
                    vec_data
                } else {
                    Vec::new()
                };

                let payload = if include_payloads {
                    store_ref.get_payload(&collection, vid).ok().flatten()
                } else {
                    None
                };

                batch.push(rekha_core::ExportedVector { id: vid, vector, payload, timestamp });
                total_sent += 1;

                if batch.len() >= batch_size
                    && tx.blocking_send(Ok(std::mem::take(&mut batch))).is_err() {
                    return;
                }

                if total_sent >= limit {
                    break;
                }
            }

            if !batch.is_empty() {
                let _ = tx.blocking_send(Ok(batch));
            }
        });

        rx
    }

    pub async fn transfer_shard_out(
        &self,
        collection: &str,
        batch_size: usize,
    ) -> Result<Vec<proto::TransferShardChunk>, RekhaError> {
        let config = self.store.load_collection_config(collection)?;
        let mut chunks = Vec::new();

        let centroids: Vec<proto::Vector> = if let Ok(cs) = self.store.load_centroids(collection) {
            cs.into_iter().enumerate().map(|(i, c)| proto::Vector {
                id: i as u64,
                data: c,
                timestamp: 0,
            }).collect()
        } else {
            Vec::new()
        };

        let cf = self.store.db().cf_handle("vectors")
            .ok_or_else(|| RekhaError::Internal("vectors cf not found".into()))?;
        let prefix = format!("{}\0", collection);

        let mut current_batch = Vec::new();
        let iter = self.store.db().iterator_cf(cf, rocksdb::IteratorMode::From(
            prefix.as_bytes(),
            rocksdb::Direction::Forward,
        ));

        for item in iter {
            let (key, value) = item.map_err(|e| RekhaError::Storage(e.to_string()))?;
            if !key.starts_with(prefix.as_bytes()) { break; }
            if value.len() < 9 || value[8] != 0 { continue; }

            let id_start = key.len() - 8;
            let vid = u64::from_be_bytes([
                key[id_start], key[id_start+1], key[id_start+2], key[id_start+3],
                key[id_start+4], key[id_start+5], key[id_start+6], key[id_start+7],
            ]);

            let f32_count = (value.len() - 9) / 4;
            let mut vec_data = Vec::with_capacity(f32_count);
            for i in 0..f32_count {
                let start = 9 + i * 4;
                let bytes: [u8; 4] = [
                    value[start], value[start+1], value[start+2], value[start+3],
                ];
                vec_data.push(f32::from_le_bytes(bytes));
            }

            let cluster_id = self.store.load_assignment(collection, vid)?
                .unwrap_or(0);

            let payload = self.store.get_payload(collection, vid).ok().flatten();

            current_batch.push(proto::VectorWithCluster {
                id: vid,
                data: vec_data,
                cluster_id,
                payload,
            });

            if current_batch.len() >= batch_size {
                if chunks.is_empty() {
                    chunks.push(proto::TransferShardChunk {
                        centroids: centroids.clone(),
                        nlist: config.nlist,
                        nprobe: config.nprobe,
                        total_dim: config.dim,
                        vector_batches: vec![proto::VectorBatch {
                            vectors: std::mem::take(&mut current_batch),
                        }],
                        final_chunk: false,
                    });
                } else {
                    chunks.push(proto::TransferShardChunk {
                        centroids: vec![],
                        nlist: 0, nprobe: 0, total_dim: 0,
                        vector_batches: vec![proto::VectorBatch {
                            vectors: std::mem::take(&mut current_batch),
                        }],
                        final_chunk: false,
                    });
                }
            }
        }

        if !current_batch.is_empty() || chunks.is_empty() {
            chunks.push(proto::TransferShardChunk {
                centroids: if chunks.is_empty() { centroids } else { vec![] },
                nlist: if chunks.is_empty() { config.nlist } else { 0 },
                nprobe: if chunks.is_empty() { config.nprobe } else { 0 },
                total_dim: if chunks.is_empty() { config.dim } else { 0 },
                vector_batches: vec![proto::VectorBatch {
                    vectors: std::mem::take(&mut current_batch),
                }],
                final_chunk: false,
            });
        }

        if let Some(last) = chunks.last_mut() {
            last.final_chunk = true;
        }

        Ok(chunks)
    }

    pub fn transfer_shard_stream(
        &self,
        collection: &str,
        batch_size: usize,
    ) -> tokio::sync::mpsc::Receiver<Result<proto::TransferShardChunk, RekhaError>> {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let store_ref = self.store.clone();
        let collection = collection.to_string();

        tokio::task::spawn_blocking(move || {
            let config = match store_ref.load_collection_config(&collection) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
            };

            let centroids: Vec<proto::Vector> = if let Ok(cs) = store_ref.load_centroids(&collection) {
                cs.into_iter().enumerate().map(|(i, c)| proto::Vector {
                    id: i as u64, data: c, timestamp: 0,
                }).collect()
            } else {
                Vec::new()
            };

            let cf = match store_ref.db().cf_handle("vectors") {
                Some(cf) => cf,
                None => {
                    let _ = tx.blocking_send(Err(RekhaError::Internal("vectors cf not found".into())));
                    return;
                }
            };
            let prefix = format!("{}\0", collection);
            use rocksdb::IteratorMode;
            let iter = store_ref.db().iterator_cf(cf, IteratorMode::From(
                prefix.as_bytes(),
                rocksdb::Direction::Forward,
            ));

            let mut current_batch = Vec::new();
            let mut needs_metadata = true;

            for item in iter {
                let (key, value) = match item {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(RekhaError::Storage(e.to_string())));
                        return;
                    }
                };
                if !key.starts_with(prefix.as_bytes()) { break; }
                if value.len() < 9 || value[8] != 0 { continue; }

                let id_start = key.len() - 8;
                let vid = u64::from_be_bytes([
                    key[id_start], key[id_start+1], key[id_start+2], key[id_start+3],
                    key[id_start+4], key[id_start+5], key[id_start+6], key[id_start+7],
                ]);

                let f32_count = (value.len() - 9) / 4;
                let mut vec_data = Vec::with_capacity(f32_count);
                for i in 0..f32_count {
                    let start = 9 + i * 4;
                    let bytes: [u8; 4] = [
                        value[start], value[start+1], value[start+2], value[start+3],
                    ];
                    vec_data.push(f32::from_le_bytes(bytes));
                }

                let cluster_id = match store_ref.load_assignment(&collection, vid) {
                    Ok(Some(c)) => c,
                    _ => 0,
                };

                let payload = store_ref.get_payload(&collection, vid).ok().flatten();

                let mut chunk_vectors = Vec::new();
                chunk_vectors.push(proto::VectorWithCluster {
                    id: vid, data: vec_data, cluster_id, payload,
                });

                if needs_metadata {
                    needs_metadata = false;
                    if tx.blocking_send(Ok(proto::TransferShardChunk {
                        centroids: centroids.clone(),
                        nlist: config.nlist,
                        nprobe: config.nprobe,
                        total_dim: config.dim,
                        vector_batches: vec![proto::VectorBatch { vectors: chunk_vectors }],
                        final_chunk: false,
                    })).is_err() { return; }
                } else {
                    current_batch.push(chunk_vectors.into_iter().next().unwrap());
                    if current_batch.len() >= batch_size
                        && tx.blocking_send(Ok(proto::TransferShardChunk {
                            centroids: vec![], nlist: 0, nprobe: 0, total_dim: 0,
                            vector_batches: vec![proto::VectorBatch {
                                vectors: std::mem::take(&mut current_batch),
                            }],
                            final_chunk: false,
                        })).is_err() { return; }
                }
            }

            if !current_batch.is_empty() {
                let _ = tx.blocking_send(Ok(proto::TransferShardChunk {
                    centroids: vec![], nlist: 0, nprobe: 0, total_dim: 0,
                    vector_batches: vec![proto::VectorBatch {
                        vectors: std::mem::take(&mut current_batch),
                    }],
                    final_chunk: true,
                }));
            } else {
                let _ = tx.blocking_send(Ok(proto::TransferShardChunk {
                    centroids: if needs_metadata { centroids } else { vec![] },
                    nlist: if needs_metadata { config.nlist } else { 0 },
                    nprobe: if needs_metadata { config.nprobe } else { 0 },
                    total_dim: if needs_metadata { config.dim } else { 0 },
                    vector_batches: vec![],
                    final_chunk: true,
                }));
            }
        });

        rx
    }

    pub async fn repair_collection(
        &self,
        collection: &str,
    ) -> Result<proto::RepairProgress, RekhaError> {
        let local_count = self.store.get_vector_count(collection)?;
        Ok(proto::RepairProgress {
            repaired: 0,
            total: local_count,
            current_node: self.node_id_str.clone(),
        })
    }

    pub fn replay_hints(&self) -> Result<u64, RekhaError> {
        let hint_store = self.store.hint_store();
        let all = hint_store.iter_all()?;
        let mut replayed = 0u64;
        let rt = tokio::runtime::Handle::current();

        for (key, value) in &all {
            let key_str = String::from_utf8_lossy(key);
            let last_colon = match key_str.rfind(':') {
                Some(pos) => pos,
                None => continue,
            };
            let second_last_colon = match key_str[..last_colon].rfind(':') {
                Some(pos) => pos,
                None => continue,
            };

            let _id: u64 = match key_str[last_colon + 1..].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let collection = &key_str[second_last_colon + 1..last_colon];
            let target = &key_str[..second_last_colon];

            let hint_data: (u64, Vec<f32>, Option<Vec<u8>>, i64) =
                match serde_json::from_slice(value) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

            let req = proto::InsertRequest {
                id: hint_data.0,
                vector: hint_data.1,
                payload: hint_data.2.as_ref().map(|p| proto::Payload {
                    content_type: "application/octet-stream".into(),
                    data: p.clone(),
                }),
                collection_name: collection.to_string(),
                is_replication: true,
                timestamp: hint_data.3 as u64,
                consistency: proto::ConsistencyLevel::One as i32,
                origin_node_id: self.node_id_str.clone(),
            };

            match rt.block_on(async { self.peer_pool.replica_insert(target, req).await }) {
                Ok(resp) if resp.success => {
                    if let Err(e) = hint_store.delete_hint(key) {
                        tracing::warn!("failed to delete hint: {}", e);
                    }
                    replayed += 1;
                }
                _ => {}
            }
        }

        Ok(replayed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_pool::PeerPool;
    use rekha_cluster::Membership;
    use rekha_cluster::chord::ChordNode;
    use tempfile::TempDir;

    async fn setup_coordinator() -> (TempDir, Coordinator) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
        let membership = Arc::new(RwLock::new(Membership::new("node1", 5000)));
        let chord_id = rekha_cluster::hash_to_chord_id(b"node1");
        let chord = Arc::new(ChordNode::new(chord_id, "127.0.0.1:5001"));
        let coord = Coordinator::new(store, membership, 1, "node1".to_string(), true, 3600, ConsistencyLevel::Quorum, 3, chord, Arc::new(PeerPool::new()), 86400);
        (dir, coord)
    }

    #[tokio::test]
    async fn test_initialize_creates_default() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();
        let exists = coord.collection_exists("default").await.unwrap();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_create_collection() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();

        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        coord.create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false).await.unwrap();
        let exists = coord.collection_exists("test").await.unwrap();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_insert_and_search() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();

        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        coord.create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false).await.unwrap();

        coord
            .insert(
                "test",
                1,
                vec![0.1, 0.2, 0.3, 0.4],
                None,
                1000,
                "node1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();

        let results = coord
            .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 1);
    }

    #[tokio::test]
    async fn test_drop_collection() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();
        coord.drop_collection("default", "node1", 0, ConsistencyLevel::Quorum, false).await.unwrap();
        let exists = coord.collection_exists("default").await.unwrap();
        assert!(!exists);
    }

    #[tokio::test]
    async fn test_delete() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();
        coord
            .insert(
                "default",
                1,
                vec![0.1; 8],
                None,
                1000,
                "node1",
                ConsistencyLevel::One,
                false,
            )
            .await
            .unwrap();
        let deleted = coord
            .delete("default", &[1], 1001, "node1", ConsistencyLevel::One, false)
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        let results = coord
            .search("default", vec![0.1; 8], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_duplicate_collection_error() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();
        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        coord.create_collection("test", config.clone(), "node1", 0, ConsistencyLevel::Quorum, false).await.unwrap();
        let result = coord.create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_linear_search_untrained() {
        let (_dir, coord) = setup_coordinator().await;
        coord.initialize().await.unwrap();

        let config = IvfConfig {
            dim: 4,
            nlist: 2,
            nprobe: 2,
            pq_m: 2,
            pq_k: 4,
            replication_factor: 3,
            distance_metric: DistanceMetric::L2,
        };
        coord.create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false).await.unwrap();

        coord
            .insert("test", 1, vec![0.1, 0.2, 0.3, 0.4], None, 1000, "node1", ConsistencyLevel::One, false)
            .await
            .unwrap();
        coord
            .insert("test", 2, vec![0.9, 0.8, 0.7, 0.6], None, 1000, "node1", ConsistencyLevel::One, false)
            .await
            .unwrap();

        let results = coord
            .search("test", vec![0.1, 0.2, 0.3, 0.4], 5, SearchParams::default())
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 1);
    }

    #[tokio::test]
    async fn test_create_collection_with_self_only_replica_no_deadlock() {
        let (_dir, coord) = setup_coordinator().await;
        let chord = coord.chord.clone();
        let self_addr = chord.address.clone();
        chord.set_successor("self", &self_addr);
        chord.successor_list.write().await.push("self".to_string());
        chord.successor_addresses.write().await.push(self_addr);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            coord.create_collection("deadlock_test", IvfConfig::default(), "tester", 1000, ConsistencyLevel::One, false),
        ).await;
        assert!(result.is_ok(), "create_collection should not deadlock with self as only replica");
    }

    #[tokio::test]
    async fn test_create_collection_with_empty_address_replica() {
        let (_dir, coord) = setup_coordinator().await;
        let chord = coord.chord.clone();
        chord.successor_list.write().await.push("ghost".to_string());
        chord.successor_addresses.write().await.push(String::new());

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            coord.create_collection("empty_addr_test", IvfConfig::default(), "tester", 1000, ConsistencyLevel::One, false),
        ).await;
        assert!(result.is_ok(), "create_collection should skip replicas with empty addresses");
    }
}
