use rekha_cluster::hash_to_chord_id;
use rekha_core::{ConsistencyLevel, IvfConfig, RekhaError};
use rekha_index::DiskIvfIndex;
use rekha_proto::proto;

use crate::coordinator::{Coordinator, IndexState};

impl Coordinator {
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
            let replicas = self
                .chord
                .replicas_for_chord_id(name_hash, self.default_rf as usize)
                .await;
            for replica in &replicas {
                if replica.address == self.chord.address {
                    continue;
                }
                if replica.address.is_empty() {
                    continue;
                }
                let req = proto::CreateCollectionRequest {
                    name: name.to_string(),
                    config: Some(config.into()),
                    is_replication: true,
                    timestamp: timestamp as u64,
                    consistency: proto::ConsistencyLevel::Quorum as i32,
                    origin_node_id: self.node_id_str.clone(),
                };
                let _ = self
                    .peer_pool
                    .replica_create_collection(&replica.address, req)
                    .await;
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

        let vectors = self.store.iterate_vectors(collection)?;
        let sample: Vec<Vec<f32>> = vectors
            .into_iter()
            .take(threshold as usize)
            .map(|(_, v, _)| v)
            .collect();

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
            let replicas = self
                .chord
                .replicas_for_chord_id(name_hash, self.default_rf as usize)
                .await;
            for replica in &replicas {
                if replica.address == self.chord.address {
                    continue;
                }
                if replica.address.is_empty() {
                    continue;
                }
                let req = proto::DropCollectionRequest {
                    name: name.to_string(),
                    is_replication: true,
                    timestamp: timestamp as u64,
                    consistency: proto::ConsistencyLevel::Quorum as i32,
                    origin_node_id: self.node_id_str.clone(),
                };
                let _ = self
                    .peer_pool
                    .replica_drop_collection(&replica.address, req)
                    .await;
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
        let grace_secs = self.gc_grace_seconds;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let tombstones = self.store.iterate_tombstones(collection)?;
        let to_delete: Vec<u64> = tombstones
            .into_iter()
            .filter(|(_, ts)| now - ts >= grace_secs)
            .map(|(id, _)| id)
            .collect();
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
}
