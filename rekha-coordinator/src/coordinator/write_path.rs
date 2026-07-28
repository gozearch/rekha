use rekha_cluster::hash_to_chord_id;
use rekha_core::{ConsistencyLevel, RekhaError};
use rekha_proto::proto;
use rekha_replication::{ConsistencyGate, LwwTimestamp};

use crate::coordinator::{Coordinator, IndexState};

impl Coordinator {
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
        let _permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RekhaError::Internal("concurrency limit closed".into()))?;
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
                    Ok(_) | Err(_) => {
                        let hint_data = serde_json::to_vec(&(id, &vector, &payload, timestamp))
                            .map_err(|e| RekhaError::Serialization(e.to_string()))?;
                        let _ = self.hinted_handoff.store_hint(
                            &replica.address,
                            collection,
                            id,
                            &hint_data,
                            timestamp,
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

        let _ = self.auto_train(collection).await;

        Ok(1u64)
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

            if !is_replication {
                let rf = self.default_rf as usize;
                let vector_hash = hash_to_chord_id(&id.to_le_bytes());
                let replicas = self.chord.replicas_for_chord_id(vector_hash, rf).await;
                for replica in &replicas {
                    if replica.node_id == self.chord.self_id_string {
                        continue;
                    }
                    if replica.address == self.chord.address {
                        continue;
                    }
                    if replica.address.is_empty() {
                        continue;
                    }
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
}
