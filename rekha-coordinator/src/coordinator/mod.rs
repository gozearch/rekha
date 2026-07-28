pub mod collection;
pub mod read_path;
pub mod transfer;
pub mod write_path;

use std::collections::HashMap;
use std::sync::Arc;

use rekha_cluster::chord::ChordNode;
use rekha_cluster::Membership;
use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, RekhaError};
use rekha_index::DiskIvfIndex;
use rekha_proto::proto;
use rekha_replication::HintedHandoff;
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
                        indexes.insert(name.clone(), IndexState::Pending { config });
                    }
                }
            }
        }
        Ok(())
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
    use rekha_cluster::chord::ChordNode;
    use rekha_cluster::Membership;
    use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig, SearchParams};
    use tempfile::TempDir;

    async fn setup_coordinator() -> (TempDir, Coordinator) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
        let membership = Arc::new(RwLock::new(Membership::new("node1", 5000)));
        let chord_id = rekha_cluster::hash_to_chord_id(b"node1");
        let chord = Arc::new(ChordNode::new(chord_id, "127.0.0.1:5001"));
        let coord = Coordinator::new(
            store,
            membership,
            1,
            "node1".to_string(),
            true,
            3600,
            ConsistencyLevel::Quorum,
            3,
            chord,
            Arc::new(PeerPool::new()),
            86400,
        );
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
        coord
            .create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();
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
        coord
            .create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();

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
        coord
            .drop_collection("default", "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();
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
        coord
            .create_collection(
                "test",
                config.clone(),
                "node1",
                0,
                ConsistencyLevel::Quorum,
                false,
            )
            .await
            .unwrap();
        let result = coord
            .create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await;
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
        coord
            .create_collection("test", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();

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
        coord
            .insert(
                "test",
                2,
                vec![0.9, 0.8, 0.7, 0.6],
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
    async fn test_create_collection_with_self_only_replica_no_deadlock() {
        let (_dir, coord) = setup_coordinator().await;
        let chord = coord.chord.clone();
        let self_addr = chord.address.clone();
        chord.set_successor("self", &self_addr);
        chord.successor_list.write().await.push("self".to_string());
        chord.successor_addresses.write().await.push(self_addr);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            coord.create_collection(
                "deadlock_test",
                IvfConfig::default(),
                "tester",
                1000,
                ConsistencyLevel::One,
                false,
            ),
        )
        .await;
        assert!(
            result.is_ok(),
            "create_collection should not deadlock with self as only replica"
        );
    }

    #[tokio::test]
    async fn test_create_collection_with_empty_address_replica() {
        let (_dir, coord) = setup_coordinator().await;
        let chord = coord.chord.clone();
        chord.successor_list.write().await.push("ghost".to_string());
        chord.successor_addresses.write().await.push(String::new());

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            coord.create_collection(
                "empty_addr_test",
                IvfConfig::default(),
                "tester",
                1000,
                ConsistencyLevel::One,
                false,
            ),
        )
        .await;
        assert!(
            result.is_ok(),
            "create_collection should skip replicas with empty addresses"
        );
    }
}
