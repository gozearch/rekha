use rekha_core::{ConsistencyLevel, Payload, RekhaError, VectorRecord, VectorStoreBackend};
use rekha_replication::{ConsistencyGate, LwwResolver};
use std::sync::atomic::Ordering;

use crate::Coordinator;

impl Coordinator {
    pub async fn replica_insert(
        &self, collection: &str, id: u64, vector: &[f32], payload: &Option<Payload>, timestamp: u64,
    ) -> Result<u64, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        if let Ok(Some(record)) = ns.get_vector_record(id) {
            if record.timestamp > timestamp { return Ok(id); }
        }
        let idx = self.index.read().await;
        if let Some(ref index) = *idx {
            if let Err(e) = index.insert(collection, id, timestamp, vector) {
                if matches!(&e, RekhaError::NotFound(_)) {
                    if let Some(cfg) = self.read_collection_config(collection) {
                        let _ = index.create_collection(collection, cfg.dim as usize, cfg.nlist as usize, cfg.nprobe as usize);
                        let _ = index.insert(collection, id, timestamp, vector);
                    }
                }
            }
        }
        drop(idx);
        let is_new = match ns.get_vector_record(id) {
            Ok(Some(ref rec)) => rec.is_tombstone,
            _ => true,
        };
        ns.put_vector(id, vector, timestamp)?;
        if let Some(ref p) = payload { ns.put_payload(id, &p.data)?; }
        if is_new { self.update_vector_count(collection, 1i64); }
        Ok(id)
    }

    pub async fn insert(
        &self, collection: &str, id: u64, vector: Vec<f32>, payload: Option<Payload>,
        timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let timestamp = LwwResolver::resolve_timestamp(timestamp);
        let id = if id == 0 { self.next_auto_id.fetch_add(1, Ordering::SeqCst) } else { id };
        self.replica_insert(collection, id, &vector, &payload, timestamp).await?;

        let mut acks = 1u64;
        let pdata = payload.as_ref().map(|p| p.data.clone());

        if let Some(cfg) = self.read_collection_config(collection) {
            let rf = cfg.replication_factor as usize;
            let shard = id % cfg.num_vector_shards;
            let replicas = self.membership.read().await.replicas_for(shard, rf);
            for replica in &replicas {
                if replica.node_id == self.config.node_id { continue; }
                let mut pool = self.peer_pool.write().await;
                if let Some(client) = pool.clients.get_mut(&replica.node_id) {
                    match client.try_remote_insert(collection, id, &vector, &pdata, timestamp).await {
                        Ok(_) => acks += 1,
                        Err(_) => {
                            self.handoff.store_hint(&self.store.hint_store(), &replica.node_id, collection, id, &vector, pdata.as_deref(), timestamp);
                        }
                    }
                } else {
                    self.handoff.store_hint(&self.store.hint_store(), &replica.node_id, collection, id, &vector, pdata.as_deref(), timestamp);
                }
            }
        }

        let rf = self.read_collection_config(collection).map(|c| c.replication_factor as usize).unwrap_or(1);
        let required = ConsistencyGate::required(consistency, rf);
        if acks >= required as u64 { Ok(id) }
        else { Err(RekhaError::Unavailable { detail: format!("consistency level not met: got {acks}/{required} acknowledgments") }) }
    }

    pub async fn replica_delete(&self, collection: &str, ids: &[u64], timestamp: u64) -> Result<u64, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        let mut removed = 0u64;
        for id in ids {
            let was_live = matches!(ns.get_vector_record(*id), Ok(Some(ref r)) if !r.is_tombstone);
            ns.put_tombstone(*id, timestamp)?;
            if was_live { removed += 1; }
        }
        if removed > 0 { self.update_vector_count(collection, -(removed as i64)); }
        Ok(ids.len() as u64)
    }

    pub async fn delete(
        &self, collection: &str, ids: &[u64], timestamp: u64, consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let timestamp = LwwResolver::resolve_timestamp(timestamp);
        self.replica_delete(collection, ids, timestamp).await?;
        let mut acks = 1u64;
        let mut hints: Vec<(String, u64, u64)> = Vec::new();

        if let Some(cfg) = self.read_collection_config(collection) {
            for id in ids {
                let shard = id % cfg.num_vector_shards;
                let replicas = self.membership.read().await.replicas_for(shard, cfg.replication_factor as usize);
                for replica in &replicas {
                    if replica.node_id == self.config.node_id { continue; }
                    let mut pool = self.peer_pool.write().await;
                    if let Some(client) = pool.clients.get_mut(&replica.node_id) {
                        match client.try_remote_delete(collection, &[*id], timestamp).await {
                            Ok(_) => acks += 1,
                            Err(_) => { if self.handoff.is_enabled() { hints.push((replica.node_id.clone(), *id, timestamp)); } }
                        }
                    } else if self.handoff.is_enabled() { hints.push((replica.node_id.clone(), *id, timestamp)); }
                }
            }
        }

        let rf = self.read_collection_config(collection).map(|c| c.replication_factor as usize).unwrap_or(1);
        let required = ConsistencyGate::required(consistency, rf);
        if acks < required as u64 {
            return Err(RekhaError::Unavailable { detail: format!("consistency level not met for delete: got {acks}/{required}") });
        }
        Ok(ids.len() as u64)
    }

    pub async fn fetch(&self, collection: &str, ids: &[u64], _consistency: ConsistencyLevel) -> Result<Vec<VectorRecord>, RekhaError> {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        let mut results = Vec::new();
        for id in ids {
            if let Ok(Some(record)) = ns.get_vector_record(*id) {
                if !record.is_tombstone { results.push(record); }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::coordinator::tests::test_coordinator;
    use rekha_core::{ConsistencyLevel, Payload, VectorStoreBackend};
    use rekha_index::RekhaIndex;

    #[tokio::test]
    async fn test_coordinator_insert() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.insert("default", 42, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], None, 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let v = ns.get_vector(42).unwrap().unwrap();
        assert!((v[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_insert_with_payload() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        let payload = Payload::from_text("test data");
        coord.insert("default", 7, vec![0.5; 8], Some(payload), 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let stored_payload = ns.get_payload(7).unwrap().unwrap();
        assert_eq!(stored_payload, b"test data");
    }

    #[tokio::test]
    async fn test_replica_insert_lww_skips_stale() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.replica_insert("default", 1, &[0.1; 8], &None, 100).await.unwrap();
        coord.replica_insert("default", 1, &[0.9; 8], &None, 50).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 100);
        assert!((rec.data.unwrap()[0] - 0.1).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_coordinator_delete_writes_tombstone() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.insert("default", 1, vec![0.1; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        assert!(ns.get_vector(1).unwrap().is_some());
        coord.delete("default", &[1], 0, ConsistencyLevel::One).await.unwrap();
        assert!(ns.get_vector(1).unwrap().is_none());
        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert!(rec.is_tombstone);
    }

    #[tokio::test]
    async fn test_coordinator_insert_auto_id() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        let id1 = coord.insert("default", 0, vec![0.1; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        let id2 = coord.insert("default", 0, vec![0.2; 8], None, 0, ConsistencyLevel::One).await.unwrap();
        assert_eq!(id2, id1 + 1);
    }

    #[tokio::test]
    async fn test_insert_quorum_rf1_succeeds_locally() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        assert!(coord.insert("default", 99, vec![0.5; 8], None, 0, ConsistencyLevel::Quorum).await.is_ok());
    }

    #[tokio::test]
    async fn test_lww_newer_overwrites() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.replica_insert("default", 1, &[0.1; 8], &None, 100).await.unwrap();
        coord.replica_insert("default", 1, &[0.9; 8], &None, 200).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 200);
        assert!((rec.data.unwrap()[0] - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_lww_equal_timestamp_applies() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.replica_insert("default", 1, &[0.1; 8], &None, 100).await.unwrap();
        coord.replica_insert("default", 1, &[0.9; 8], &None, 100).await.unwrap();
        let ns = coord.store().as_ref().clone().with_namespace("default".into());
        let rec = ns.get_vector_record(1).unwrap().unwrap();
        assert_eq!(rec.timestamp, 100);
        assert!((rec.data.unwrap()[0] - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_consistency_one_succeeds_no_peers() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("one_test", 8, 16, 4, 3, 100, ConsistencyLevel::One).await.unwrap();
        assert!(coord.insert("one_test", 1, vec![0.5; 8], None, 0, ConsistencyLevel::One).await.is_ok());
    }

    #[tokio::test]
    async fn test_consistency_quorum_fails_no_peers() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("quorum_test", 8, 16, 4, 3, 100, ConsistencyLevel::One).await.unwrap();
        let result = coord.insert("quorum_test", 1, vec![0.5; 8], None, 0, ConsistencyLevel::Quorum).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_consistency_all_fails_no_peers() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("all_test", 8, 16, 4, 3, 100, ConsistencyLevel::One).await.unwrap();
        let result = coord.insert("all_test", 1, vec![0.5; 8], None, 0, ConsistencyLevel::All).await;
        assert!(result.is_err());
    }
}
