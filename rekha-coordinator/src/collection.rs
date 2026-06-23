use rekha_core::{CollectionConfig, CollectionMeta, ConsistencyLevel, RekhaError};
use rekha_replication::{ConsistencyGate, LwwResolver};

use crate::Coordinator;

impl Coordinator {
    pub(super) fn read_collection_config(&self, name: &str) -> Option<CollectionConfig> {
        let key = format!("collection:{name}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                if !meta.is_deleted { return Some(meta.config); }
                return None;
            }
            serde_json::from_slice(&data).ok()
        } else { None }
    }

    pub(super) fn update_vector_count(&self, collection: &str, delta: i64) {
        let key = format!("collection:{collection}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            if let Ok(mut meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                if delta > 0 { meta.vector_count = meta.vector_count.saturating_add(delta as u64); }
                else { meta.vector_count = meta.vector_count.saturating_sub((-delta) as u64); }
                if let Ok(json) = serde_json::to_vec(&meta) { let _ = self.store.put_metadata(&key, &json); }
            }
        }
    }

    pub async fn create_collection(
        &self, name: &str, dim: u32, nlist: u32, nprobe: u32, rf: u64,
        timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let timestamp = LwwResolver::resolve_timestamp(timestamp);
        let key = format!("collection:{name}");
        if let Some(data) = self.store.get_metadata(&key)? {
            if let Ok(existing) = serde_json::from_slice::<CollectionMeta>(&data) {
                if !existing.is_deleted && existing.timestamp >= timestamp { return Ok(false); }
            } else { return Ok(false); }
        }
        let cfg = CollectionConfig {
            dim, nlist, nprobe, num_vector_shards: 6, replication_factor: rf,
            num_dim_groups: 4, dim_group_size: dim / 4,
            pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
        };
        let meta = CollectionMeta { config: cfg.clone(), timestamp, is_deleted: false, vector_count: 0 };
        let json = serde_json::to_vec(&meta).map_err(|e| RekhaError::InvalidArgument(format!("serialize config: {e}")))?;
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection(name, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
        }
        drop(idx);

        let proto_cfg = rekha_proto::proto::CollectionConfig {
            dim, nlist, nprobe, num_vector_shards: 6, replication_factor: rf,
            num_dim_groups: 4, dim_group_size: dim / 4,
            pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
        };
        let peer_ids: Vec<String> = { let pool = self.peer_pool.read().await; pool.clients.keys().cloned().collect() };
        let required = ConsistencyGate::required(consistency, rf as usize);
        let mut acks = 1u64;
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                match client.try_remote_create_collection(name, &proto_cfg, timestamp).await {
                    Ok(true) => acks += 1,
                    Ok(false) => {}
                    Err(_) => {
                        self.handoff.store_collection_hint(&self.store.hint_store(), node_id, name, &[], timestamp, 0);
                    }
                }
            }
        }
        if (acks as usize) >= required { Ok(true) }
        else { Err(RekhaError::Unavailable { detail: format!("consistency level not met: got {acks}/{required}") }) }
    }

    pub async fn drop_collection(
        &self, name: &str, timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let timestamp = LwwResolver::resolve_timestamp(timestamp);
        let key = format!("collection:{name}");
        let existing = match self.store.get_metadata(&key)? {
            Some(data) => {
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) { meta }
                else {
                    CollectionMeta { config: serde_json::from_slice(&data).unwrap_or_default(), timestamp: 0, is_deleted: false, vector_count: 0 }
                }
            }
            None => return Ok(false),
        };
        if existing.timestamp >= timestamp { return Ok(false); }
        let drop_rf = existing.config.replication_factor as usize;
        let meta = CollectionMeta { config: existing.config, timestamp, is_deleted: true, vector_count: existing.vector_count };
        let json = serde_json::to_vec(&meta).map_err(|e| RekhaError::InvalidArgument(format!("serialize config: {e}")))?;
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx { let _ = index.drop_collection(name); }
        drop(idx);

        let required = ConsistencyGate::required(consistency, drop_rf);
        let peer_ids: Vec<String> = { let pool = self.peer_pool.read().await; pool.clients.keys().cloned().collect() };
        let mut acks = 1u64;
        for node_id in &peer_ids {
            let mut pool = self.peer_pool.write().await;
            if let Some(client) = pool.clients.get_mut(node_id) {
                match client.try_remote_drop_collection(name, timestamp).await {
                    Ok(true) => acks += 1,
                    Ok(false) => {}
                    Err(_) => {
                        self.handoff.store_collection_hint(&self.store.hint_store(), node_id, name, &[], timestamp, 1);
                    }
                }
            }
        }
        if (acks as usize) >= required { Ok(true) }
        else { Err(RekhaError::Unavailable { detail: format!("consistency level not met for collection drop: got {acks}/{required} acknowledgments") }) }
    }

    pub async fn replicate_collection(
        &self, name: &str, proto_cfg: &rekha_proto::proto::CollectionConfig, timestamp: u64,
    ) -> Result<bool, RekhaError> {
        let cfg: CollectionConfig = proto_cfg.clone().into();
        let key = format!("collection:{name}");
        if let Ok(Some(data)) = self.store.get_metadata(&key) {
            if let Ok(existing) = serde_json::from_slice::<CollectionMeta>(&data) {
                if existing.timestamp > timestamp { return Ok(false); }
            }
        }
        let meta = CollectionMeta { config: cfg.clone(), timestamp, is_deleted: false, vector_count: 0 };
        let json = serde_json::to_vec(&meta).map_err(|e| RekhaError::InvalidArgument(format!("serialize config: {e}")))?;
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            let _ = index.create_collection(name, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
        }
        Ok(true)
    }

    pub async fn replicate_drop_collection(&self, name: &str, timestamp: u64) -> Result<bool, RekhaError> {
        let key = format!("collection:{name}");
        let existing = match self.store.get_metadata(&key)? {
            Some(data) => {
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) { meta }
                else {
                    CollectionMeta { config: serde_json::from_slice(&data).unwrap_or_default(), timestamp: 0, is_deleted: false, vector_count: 0 }
                }
            }
            None => return Ok(false),
        };
        if existing.timestamp > timestamp && !existing.is_deleted { return Ok(false); }
        let meta = CollectionMeta { config: existing.config, timestamp, is_deleted: true, vector_count: existing.vector_count };
        let json = serde_json::to_vec(&meta).map_err(|e| RekhaError::InvalidArgument(format!("serialize config: {e}")))?;
        self.store.put_metadata(&key, &json)?;
        let idx = self.index.read().await;
        if let Some(ref index) = *idx { let _ = index.drop_collection(name); }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use crate::coordinator::tests::test_coordinator;
    use rekha_core::CollectionMeta;
    use rekha_index::RekhaIndex;

    #[tokio::test]
    async fn test_create_collection_with_explicit_timestamp() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("ts_test", 8, 16, 4, 1, 777, rekha_core::ConsistencyLevel::One).await.unwrap();
        let key = "collection:ts_test";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.timestamp, 777);
        assert!(!meta.is_deleted);
    }

    #[tokio::test]
    async fn test_create_collection_duplicate_returns_false() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        assert!(coord.create_collection("dup_test", 8, 16, 4, 1, 100, rekha_core::ConsistencyLevel::One).await.unwrap());
        assert!(!coord.create_collection("dup_test", 8, 16, 4, 1, 100, rekha_core::ConsistencyLevel::One).await.unwrap());
        assert!(coord.create_collection("dup_test", 8, 16, 4, 1, 200, rekha_core::ConsistencyLevel::One).await.unwrap());
    }

    #[tokio::test]
    async fn test_drop_collection_writes_tombstone() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("dropme", 8, 16, 4, 1, 100, rekha_core::ConsistencyLevel::One).await.unwrap();
        coord.drop_collection("dropme", 200, rekha_core::ConsistencyLevel::One).await.unwrap();
        let key = "collection:dropme";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert!(meta.is_deleted);
        assert_eq!(meta.timestamp, 200);
    }

    #[tokio::test]
    async fn test_collection_config_timestamp_stored() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        let ts = 999888777;
        coord.create_collection("ts_stored", 16, 32, 8, 2, ts, rekha_core::ConsistencyLevel::One).await.unwrap();
        let key = "collection:ts_stored";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.timestamp, ts);
    }

    #[tokio::test]
    async fn test_replicate_collection_lww_skips_stale() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        let proto_cfg = rekha_proto::proto::CollectionConfig {
            dim: 8, nlist: 16, nprobe: 4, num_vector_shards: 6,
            replication_factor: 1, num_dim_groups: 4, dim_group_size: 2,
            pq_num_sub_vectors: 4, pq_num_centroids: 256, re_rank_k: 256,
        };
        coord.replicate_collection("lww_coll", &proto_cfg, 500).await.unwrap();
        let mut stale_cfg = proto_cfg.clone();
        stale_cfg.nlist = 999;
        coord.replicate_collection("lww_coll", &stale_cfg, 300).await.unwrap();
        let key = "collection:lww_coll";
        let data = coord.store().get_metadata(key).unwrap().unwrap();
        let meta: CollectionMeta = serde_json::from_slice(&data).unwrap();
        assert_eq!(meta.config.nlist, 16);
        assert_eq!(meta.timestamp, 500);
    }

    #[tokio::test]
    async fn test_multi_collection_different_dims() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("dim4", 4, 8, 2, 1, 100, rekha_core::ConsistencyLevel::One).await.unwrap();
        coord.create_collection("dim16", 16, 16, 4, 1, 100, rekha_core::ConsistencyLevel::One).await.unwrap();
        coord.insert("dim4", 1, vec![0.5; 4], None, 0, rekha_core::ConsistencyLevel::One).await.unwrap();
        coord.insert("dim16", 1, vec![0.5; 16], None, 0, rekha_core::ConsistencyLevel::One).await.unwrap();
        let params = rekha_core::SearchParams { ef_search: 64, nprobe: 4, include_payloads: false, local_only: true };
        let (r4, _) = coord.search("dim4", vec![0.5; 4], 5, params.clone(), rekha_core::ConsistencyLevel::One).await.unwrap();
        assert!(!r4.is_empty());
        let (r16, _) = coord.search("dim16", vec![0.5; 16], 5, params, rekha_core::ConsistencyLevel::One).await.unwrap();
        assert!(!r16.is_empty());
    }
}
