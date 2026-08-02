use std::collections::{BinaryHeap, HashMap};

use rekha_cluster::hash_to_chord_id;
use rekha_core::{RekhaError, ScoredPoint, SearchParams};
use rekha_proto::proto;
use rekha_storage::RekhaStore;

use crate::coordinator::{Coordinator, IndexState};

impl Coordinator {
    fn linear_search(
        store: &RekhaStore,
        collection: &str,
        query: &[f32],
        k: u32,
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let k = k as usize;
        let mut heap: BinaryHeap<ScoredPoint> = BinaryHeap::with_capacity(k + 1);

        let vectors = store.iterate_vectors(collection)?;
        for (id, vec_data, _ts) in &vectors {
            let dist: f32 = query
                .iter()
                .zip(vec_data.iter())
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();

            let payload = if include_payloads {
                store.get_payload(collection, *id).ok().flatten()
            } else {
                None
            };

            heap.push(ScoredPoint {
                id: *id,
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
        let _permit = self
            .concurrency_limit
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RekhaError::Internal("concurrency limit closed".into()))?;
        let mut all_results: Vec<ScoredPoint> = {
            let indexes = self.indexes.read().await;
            match indexes.get(collection) {
                Some(IndexState::Trained(idx)) => {
                    let search_params = SearchParams {
                        nprobe: if params.nprobe > 0 {
                            params.nprobe
                        } else {
                            idx.config().nprobe
                        },
                        k,
                        include_payloads: params.include_payloads,
                        pre_filter: params.pre_filter.clone(),
                        local_only: true,
                    };
                    idx.search(&query, &search_params)?
                }
                Some(IndexState::Pending { .. }) => {
                    drop(indexes);
                    Self::linear_search(
                        &self.store,
                        collection,
                        &query,
                        k,
                        params.include_payloads,
                    )?
                }
                None => {
                    return Err(RekhaError::NotFound(format!(
                        "collection {} not found",
                        collection
                    )))
                }
            }
        };

        if !params.local_only {
            let rf = self.default_rf as usize;
            let query_bytes: Vec<u8> = query.iter().flat_map(|f| f.to_le_bytes()).collect();
            let query_hash = hash_to_chord_id(&query_bytes);
            let replicas = self.chord.replicas_for_chord_id(query_hash, rf).await;

            for replica in &replicas {
                if replica.address == self.chord.address {
                    continue;
                }
                if replica.address.is_empty() {
                    continue;
                }

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

        let mut results: Vec<ScoredPoint> = merged.into_values().collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        results.truncate(k as usize);
        Ok(results)
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
}
