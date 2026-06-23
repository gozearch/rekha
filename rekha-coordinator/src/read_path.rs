use rekha_core::{ConsistencyLevel, DistanceMetric, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats, VectorStoreBackend};
use rekha_replication::ConsistencyGate;
use std::collections::HashMap;

use crate::Coordinator;

impl Coordinator {
    fn maybe_load_payload(&self, collection: &str, include: bool, id: u64) -> Option<Payload> {
        if include {
            let ns = self.store.as_ref().clone().with_namespace(collection.into());
            ns.get_payload(id).ok().flatten().map(Payload::from_bytes)
        } else { None }
    }

    fn maybe_load_timestamp(&self, collection: &str, id: u64) -> u64 {
        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        ns.get_vector_record(id).ok().flatten().map(|r| r.timestamp).unwrap_or(0)
    }

    pub async fn search(
        &self, collection: &str, query: Vec<f32>, k: usize,
        params: SearchParams, consistency: ConsistencyLevel,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();
        let mut stats = SearchStats::default();

        let index_guard = self.index.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal {
            detail: "index not initialized".into(),
        })?;

        if !index.has_collection(collection) {
            return Err(RekhaError::NotFound(collection.into()));
        }

        let mut candidates: Vec<ScoredPoint> = Vec::new();
        let (ids, dists) = index.search(collection, &query, k * 2, &params).map_err(|e| {
            stats.warnings.push(format!("search failed: {e}")); e
        })?;
        for (i, id) in ids.iter().enumerate() {
            let score = dists.get(i).copied().unwrap_or(f32::MAX);
            let payload = self.maybe_load_payload(collection, params.include_payloads, *id);
            let timestamp = self.maybe_load_timestamp(collection, *id);
            candidates.push(ScoredPoint { id: *id, score, payload, timestamp });
        }

        if !params.local_only {
            let peer_count = { self.peer_pool.read().await.clients.len() };
            if peer_count > 0 {
                let rf = self.read_collection_config(collection)
                    .map(|c| c.replication_factor as usize).unwrap_or(1);
                let needed_per_shard = ConsistencyGate::required(consistency, rf);

                let mut node_set: std::collections::HashSet<String> = std::collections::HashSet::new();
                if let Some(cfg) = self.read_collection_config(collection) {
                    let memb = self.membership.read().await;
                    for shard in 0..cfg.num_vector_shards {
                        let replicas = memb.replicas_for(shard, rf);
                        for replica in replicas.iter().take(needed_per_shard) {
                            if replica.node_id != self.config.node_id {
                                node_set.insert(replica.node_id.clone());
                            }
                        }
                    }
                }
                let node_ids: Vec<String> = node_set.into_iter().collect();
                let mut pool = self.peer_pool.write().await;

                if !node_ids.is_empty() {
                    let mut peer_params = params.clone();
                    peer_params.local_only = true;
                    let mut all_peer_candidates: Vec<ScoredPoint> = Vec::new();
                    let mut peer_stats = SearchStats::default();
                    for node_id in &node_ids {
                        if let Some(client) = pool.clients.get_mut(node_id) {
                            match client.try_search(&query, k, &peer_params, collection).await {
                                Ok((candidates, _)) => { all_peer_candidates.extend(candidates); client.error_count = 0; }
                                Err(_) => {
                                    if let Some(c) = pool.clients.get_mut(node_id) {
                                        c.error_count += 1;
                                        if c.error_count >= 3 { pool.clients.remove(node_id); }
                                    }
                                    peer_stats.warnings.push(format!("peer {node_id} search failed"));
                                }
                            }
                        }
                    }
                    all_peer_candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
                    all_peer_candidates.truncate(k * 2);
                    peer_stats.nodes_contacted = node_ids.len() as u32;
                    stats.nodes_contacted = 1 + peer_stats.nodes_contacted;
                    stats.warnings.extend(peer_stats.warnings);
                    candidates.extend(all_peer_candidates);
                }
            }
        }

        let mut seen: HashMap<u64, ScoredPoint> = HashMap::new();
        for c in candidates {
            let id = c.id;
            match seen.entry(id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => { if c.timestamp > entry.get().timestamp { entry.insert(c); } }
                std::collections::hash_map::Entry::Vacant(entry) => { entry.insert(c); }
            }
        }
        let mut candidates: Vec<ScoredPoint> = seen.into_values().collect();
        candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        candidates.truncate(k * 2);

        let ns = self.store.as_ref().clone().with_namespace(collection.into());
        for candidate in candidates.iter_mut().take(k * 2) {
            if let Ok(Some(full_vec)) = ns.get_vector(candidate.id) {
                candidate.score = DistanceMetric::L2.distance(&full_vec, &query);
            }
        }
        candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        candidates.truncate(k);

        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.vectors_scanned = candidates.len() as u64;
        Ok((candidates, stats))
    }
}

#[cfg(test)]
mod tests {
    use crate::coordinator::tests::test_coordinator;
    use rekha_core::{ConsistencyLevel, SearchParams};
    use rekha_index::RekhaIndex;

    #[tokio::test]
    async fn test_coordinator_search_before_init() {
        let coord = test_coordinator();
        let result = coord.search("default", vec![0.0; 8], 5, SearchParams::default(), ConsistencyLevel::One).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_uses_consistent_hashing_for_quorum() {
        let coord = test_coordinator();
        let index = RekhaIndex::new().unwrap();
        coord.initialize(index).await;
        coord.create_collection("rt4", 4, 8, 2, 2, 100, ConsistencyLevel::One).await.unwrap();
        for i in 0..20 {
            coord.insert("rt4", i, (0..4).map(|d| (i * 4 + d) as f32).collect(), None, 100, ConsistencyLevel::One).await.unwrap();
        }
        let params = SearchParams { ef_search: 64, nprobe: 4, include_payloads: false, local_only: false };
        let result = coord.search("rt4", vec![0.0; 4], 5, params, ConsistencyLevel::Quorum).await;
        assert!(result.is_ok());
    }
}
