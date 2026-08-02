use rekha_core::RekhaError;
use rekha_proto::proto;

use crate::coordinator::Coordinator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::peer_pool::PeerPool;
    use rekha_cluster::chord::ChordNode;
    use rekha_cluster::Membership;
    use rekha_core::{ConsistencyLevel, DistanceMetric, IvfConfig};
    use rekha_storage::RekhaStore;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::RwLock;

    async fn setup_coordinator_with_collection() -> (TempDir, Coordinator, String) {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(RekhaStore::open(dir.path().to_str().unwrap()).unwrap());
        let membership = Arc::new(RwLock::new(Membership::new("node1", 5000)));
        let chord_id = rekha_cluster::hash_to_chord_id(b"node1");
        let chord = Arc::new(ChordNode::new(chord_id, "127.0.0.1:5001"));
        let coord = Coordinator::new(
            store.clone(),
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

        for i in 1..=10 {
            coord
                .insert(
                    "test",
                    i,
                    vec![i as f32 * 0.1; 4],
                    Some(vec![i as u8; 4]),
                    1000 + i as i64,
                    "node1",
                    ConsistencyLevel::One,
                    false,
                )
                .await
                .unwrap();
        }

        (dir, coord, "test".to_string())
    }

    #[tokio::test]
    async fn test_export_collection_basic() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let exported = coord
            .export_collection(&collection, 0, 10, true, true)
            .await
            .unwrap();

        assert_eq!(exported.len(), 10);
        for (idx, ev) in exported.iter().enumerate() {
            assert_eq!(ev.id, (idx + 1) as u64);
            assert_eq!(ev.vector.len(), 4);
            assert_eq!(ev.payload, Some(vec![(idx + 1) as u8; 4]));
            assert_eq!(ev.timestamp, 1000 + (idx + 1) as i64);
        }
    }

    #[tokio::test]
    async fn test_export_collection_without_vectors() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let exported = coord
            .export_collection(&collection, 0, 10, false, true)
            .await
            .unwrap();

        assert_eq!(exported.len(), 10);
        for ev in exported {
            assert!(ev.vector.is_empty());
            assert!(ev.payload.is_some());
        }
    }

    #[tokio::test]
    async fn test_export_collection_without_payloads() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let exported = coord
            .export_collection(&collection, 0, 10, true, false)
            .await
            .unwrap();

        assert_eq!(exported.len(), 10);
        for ev in exported {
            assert!(!ev.vector.is_empty());
            assert!(ev.payload.is_none());
        }
    }

    #[tokio::test]
    async fn test_export_collection_with_offset_and_limit() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let exported = coord
            .export_collection(&collection, 2, 5, true, true)
            .await
            .unwrap();

        assert_eq!(exported.len(), 5);
        assert_eq!(exported[0].id, 3);
        assert_eq!(exported[1].id, 4);
        assert_eq!(exported[4].id, 7);
    }

    #[tokio::test]
    async fn test_export_collection_limit_exceeds_data() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let exported = coord
            .export_collection(&collection, 0, 100, true, true)
            .await
            .unwrap();

        assert_eq!(exported.len(), 10);
    }

    #[tokio::test]
    async fn test_export_stream_basic() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.export_stream(&collection, 0, 10, true, true, 3);
        let mut total = 0;
        while let Some(batch_result) = rx.recv().await {
            let batch = batch_result.unwrap();
            total += batch.len();
            assert!(batch.len() <= 3);
            for ev in batch {
                assert!(!ev.vector.is_empty());
                assert!(ev.payload.is_some());
            }
        }
        assert_eq!(total, 10);
    }

    #[tokio::test]
    async fn test_export_stream_with_offset_and_limit() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.export_stream(&collection, 2, 5, true, true, 2);
        let mut total = 0;
        while let Some(batch_result) = rx.recv().await {
            let batch = batch_result.unwrap();
            total += batch.len();
        }
        assert_eq!(total, 5);
    }

    #[tokio::test]
    async fn test_export_stream_without_vectors() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.export_stream(&collection, 0, 10, false, true, 5);
        let mut total = 0;
        while let Some(batch_result) = rx.recv().await {
            let batch = batch_result.unwrap();
            total += batch.len();
            for ev in batch {
                assert!(ev.vector.is_empty());
                assert!(ev.payload.is_some());
            }
        }
        assert_eq!(total, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_out_basic() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let chunks = coord.transfer_shard_out(&collection, 5).await.unwrap();

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().final_chunk);

        let first = &chunks[0];
        assert!(!first.centroids.is_empty());
        assert_eq!(first.nlist, 2);
        assert_eq!(first.nprobe, 2);
        assert_eq!(first.total_dim, 4);

        let mut total_vectors = 0;
        for chunk in &chunks {
            for batch in &chunk.vector_batches {
                total_vectors += batch.vectors.len();
            }
        }
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_out_single_batch() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let chunks = coord.transfer_shard_out(&collection, 20).await.unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].final_chunk);
        assert!(!chunks[0].centroids.is_empty());

        let total_vectors: usize = chunks[0]
            .vector_batches
            .iter()
            .map(|b| b.vectors.len())
            .sum();
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_out_batch_boundaries() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let chunks = coord.transfer_shard_out(&collection, 3).await.unwrap();

        assert!(chunks.len() >= 3);
        assert!(chunks.last().unwrap().final_chunk);

        let mut total_vectors = 0;
        for chunk in &chunks {
            for batch in &chunk.vector_batches {
                total_vectors += batch.vectors.len();
            }
        }
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_out_no_vectors() {
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
            .create_collection("empty", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();

        let chunks = coord.transfer_shard_out("empty", 5).await.unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].final_chunk);
        assert!(chunks[0].vector_batches[0].vectors.is_empty());
    }

    #[tokio::test]
    async fn test_transfer_shard_stream_basic() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.transfer_shard_stream(&collection, 5);
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk.unwrap());
        }

        assert!(!chunks.is_empty());
        assert!(chunks.last().unwrap().final_chunk);
        assert!(!chunks[0].centroids.is_empty());
        assert_eq!(chunks[0].nlist, 2);

        let mut total_vectors = 0;
        for chunk in &chunks {
            for batch in &chunk.vector_batches {
                total_vectors += batch.vectors.len();
            }
        }
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_stream_single_batch() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.transfer_shard_stream(&collection, 20);
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk.unwrap());
        }

        // First chunk has metadata (centroids), second has the vectors
        assert_eq!(chunks.len(), 2);
        assert!(chunks.last().unwrap().final_chunk);
        assert!(!chunks[0].centroids.is_empty());
        assert!(chunks[1].centroids.is_empty());

        let total_vectors: usize = chunks
            .iter()
            .flat_map(|c| &c.vector_batches)
            .map(|b| b.vectors.len())
            .sum();
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_stream_batch_boundaries() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let mut rx = coord.transfer_shard_stream(&collection, 3);
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk.unwrap());
        }

        assert!(chunks.len() >= 3);
        assert!(chunks.last().unwrap().final_chunk);

        let mut total_vectors = 0;
        for chunk in &chunks {
            for batch in &chunk.vector_batches {
                total_vectors += batch.vectors.len();
            }
        }
        assert_eq!(total_vectors, 10);
    }

    #[tokio::test]
    async fn test_transfer_shard_stream_no_vectors() {
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
            .create_collection("empty", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();

        let mut rx = coord.transfer_shard_stream("empty", 5);
        let mut chunks = Vec::new();
        while let Some(chunk) = rx.recv().await {
            chunks.push(chunk.unwrap());
        }

        // With no vectors, stream sends 1 final chunk with metadata and empty vectors
        assert_eq!(chunks.len(), 1);
        let chunk = &chunks[0];
        assert!(chunk.final_chunk);
        assert!(chunk.vector_batches.is_empty() || chunk.vector_batches[0].vectors.is_empty());
    }

    #[tokio::test]
    async fn test_repair_collection_basic() {
        let (_dir, coord, collection) = setup_coordinator_with_collection().await;

        let progress = coord.repair_collection(&collection).await.unwrap();

        assert_eq!(progress.total, 10);
        assert_eq!(progress.repaired, 0);
        assert_eq!(progress.current_node, "node1");
    }

    #[tokio::test]
    async fn test_repair_collection_empty() {
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
            .create_collection("empty", config, "node1", 0, ConsistencyLevel::Quorum, false)
            .await
            .unwrap();

        let progress = coord.repair_collection("empty").await.unwrap();

        assert_eq!(progress.total, 0);
        assert_eq!(progress.repaired, 0);
        assert_eq!(progress.current_node, "node1");
    }
}

impl Coordinator {
    pub async fn export_collection(
        &self,
        collection: &str,
        offset: u64,
        limit: u64,
        include_vectors: bool,
        include_payloads: bool,
    ) -> Result<Vec<rekha_core::ExportedVector>, RekhaError> {
        let vectors = self.store.iterate_vectors(collection)?;
        let results: Vec<rekha_core::ExportedVector> = vectors
            .into_iter()
            .skip(offset as usize)
            .take(limit as usize)
            .map(|(id, data, timestamp)| {
                let vector = if include_vectors { data } else { Vec::new() };
                let payload = if include_payloads {
                    self.store.get_payload(collection, id).ok().flatten()
                } else {
                    None
                };
                rekha_core::ExportedVector {
                    id,
                    vector,
                    payload,
                    timestamp,
                }
            })
            .collect();

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
            let vectors = match store_ref.iterate_vectors(&collection) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
            };

            let mut batch = Vec::new();
            let mut total_sent = 0u64;

            for (vid, data, timestamp) in vectors.into_iter().skip(offset as usize) {
                let vector = if include_vectors { data } else { Vec::new() };
                let payload = if include_payloads {
                    store_ref.get_payload(&collection, vid).ok().flatten()
                } else {
                    None
                };

                batch.push(rekha_core::ExportedVector {
                    id: vid,
                    vector,
                    payload,
                    timestamp,
                });
                total_sent += 1;

                if batch.len() >= batch_size
                    && tx.blocking_send(Ok(std::mem::take(&mut batch))).is_err()
                {
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
            cs.into_iter()
                .enumerate()
                .map(|(i, c)| proto::Vector {
                    id: i as u64,
                    data: c,
                    timestamp: 0,
                })
                .collect()
        } else {
            Vec::new()
        };

        let vectors = self.store.iterate_vectors(collection)?;
        let mut current_batch = Vec::new();

        for (vid, vec_data, _) in &vectors {
            let cluster_id = self.store.load_assignment(collection, *vid)?.unwrap_or(0);

            let payload = self.store.get_payload(collection, *vid).ok().flatten();

            current_batch.push(proto::VectorWithCluster {
                id: *vid,
                data: vec_data.clone(),
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
                        nlist: 0,
                        nprobe: 0,
                        total_dim: 0,
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

            let centroids: Vec<proto::Vector> =
                if let Ok(cs) = store_ref.load_centroids(&collection) {
                    cs.into_iter()
                        .enumerate()
                        .map(|(i, c)| proto::Vector {
                            id: i as u64,
                            data: c,
                            timestamp: 0,
                        })
                        .collect()
                } else {
                    Vec::new()
                };

            let vectors = match store_ref.iterate_vectors(&collection) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.blocking_send(Err(e));
                    return;
                }
            };

            let mut current_batch = Vec::new();
            let mut needs_metadata = true;

            for (vid, vec_data, _) in &vectors {
                let cluster_id = match store_ref.load_assignment(&collection, *vid) {
                    Ok(Some(c)) => c,
                    _ => 0,
                };

                let payload = store_ref.get_payload(&collection, *vid).ok().flatten();

                let chunk_vector = proto::VectorWithCluster {
                    id: *vid,
                    data: vec_data.clone(),
                    cluster_id,
                    payload,
                };

                if needs_metadata {
                    needs_metadata = false;
                    if tx
                        .blocking_send(Ok(proto::TransferShardChunk {
                            centroids: centroids.clone(),
                            nlist: config.nlist,
                            nprobe: config.nprobe,
                            total_dim: config.dim,
                            vector_batches: vec![proto::VectorBatch {
                                vectors: vec![chunk_vector],
                            }],
                            final_chunk: false,
                        }))
                        .is_err()
                    {
                        return;
                    }
                } else {
                    current_batch.push(chunk_vector);
                    if current_batch.len() >= batch_size
                        && tx
                            .blocking_send(Ok(proto::TransferShardChunk {
                                centroids: vec![],
                                nlist: 0,
                                nprobe: 0,
                                total_dim: 0,
                                vector_batches: vec![proto::VectorBatch {
                                    vectors: std::mem::take(&mut current_batch),
                                }],
                                final_chunk: false,
                            }))
                            .is_err()
                    {
                        return;
                    }
                }
            }

            if !current_batch.is_empty() {
                let _ = tx.blocking_send(Ok(proto::TransferShardChunk {
                    centroids: vec![],
                    nlist: 0,
                    nprobe: 0,
                    total_dim: 0,
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
}
