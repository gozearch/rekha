use rekha_core::RekhaError;
use rekha_proto::proto;

use crate::coordinator::Coordinator;

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
