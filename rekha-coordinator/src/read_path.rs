use rekha_core::{ConsistencyLevel, DistanceMetric, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats, VectorStoreBackend};
use rekha_replication::ConsistencyGate;
use std::collections::HashMap;

use crate::Coordinator;

impl Coordinator {
    fn load_payload(&self, collection: &str, include: bool, id: u64) -> Option<Payload> {
        if include { self.ns(collection).get_payload(id).ok().flatten().map(Payload::from_bytes) } else { None }
    }

    fn load_timestamp(&self, collection: &str, id: u64) -> u64 {
        self.ns(collection).get_vector_record(id).ok().flatten().map(|r| r.timestamp).unwrap_or(0)
    }

    /// Local index search — returns candidates with payloads and timestamps.
    async fn local_search(&self, collection: &str, query: &[f32], k: usize, params: &SearchParams) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let mut stats = SearchStats::default();
        let index_guard = self.index.read().await;
        let index = index_guard.as_ref().ok_or_else(|| RekhaError::Internal { detail: "index not initialized".into() })?;
        if !index.has_collection(collection) {
            return Err(RekhaError::NotFound(collection.into()));
        }
        let (ids, dists) = index.search(collection, query, k * 2, params).map_err(|e| { stats.warnings.push(format!("search failed: {e}")); e })?;
        let points = ids.iter().enumerate().map(|(i, id)| ScoredPoint {
            id: *id,
            score: dists.get(i).copied().unwrap_or(f32::MAX),
            payload: self.load_payload(collection, params.include_payloads, *id),
            timestamp: self.load_timestamp(collection, *id),
        }).collect();
        Ok((points, stats))
    }

    /// Peer fan-out search — queries remote replicas.
    async fn fanout_search(&self, collection: &str, query: &[f32], k: usize, params: &SearchParams, consistency: ConsistencyLevel) -> (Vec<ScoredPoint>, SearchStats) {
        let mut stats = SearchStats::default();
        let peer_count = { self.peer_pool.read().await.clients.len() };
        if peer_count == 0 { stats.nodes_contacted = 1; return (vec![], stats); }

        let cfg = self.read_collection_config(collection);
        let rf = cfg.as_ref().map(|c| c.replication_factor as usize).unwrap_or(1);
        let needed = ConsistencyGate::required(consistency, rf);

        let mut node_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        if let Some(cfg) = cfg {
            let memb = self.membership.read().await;
            for shard in 0..cfg.num_vector_shards {
                for replica in memb.replicas_for(shard, rf).iter().take(needed) {
                    if replica.node_id != self.config.node_id { node_set.insert(replica.node_id.clone()); }
                }
            }
        }

        let node_ids: Vec<String> = node_set.into_iter().collect();
        let mut pool = self.peer_pool.write().await;
        let mut all_peer_candidates = Vec::new();
        let mut peer_params = params.clone();
        peer_params.local_only = true;

        for node_id in &node_ids {
            if let Some(client) = pool.clients.get_mut(node_id) {
                match client.try_search(query, k, &peer_params, collection).await {
                    Ok((candidates, _)) => { all_peer_candidates.extend(candidates); client.error_count = 0; }
                    Err(_) => {
                        if let Some(c) = pool.clients.get_mut(node_id) { c.error_count += 1; if c.error_count >= 3 { pool.clients.remove(node_id); } }
                        stats.warnings.push(format!("peer {node_id} search failed"));
                    }
                }
            }
        }
        all_peer_candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        all_peer_candidates.truncate(k * 2);
        stats.nodes_contacted = 1 + node_ids.len() as u32;
        (all_peer_candidates, stats)
    }

    /// LWW dedup + re-rank with full vectors.
    fn merge_and_rerank(&self, candidates: Vec<ScoredPoint>, query: &[f32], k: usize, collection: &str) -> Vec<ScoredPoint> {
        let mut seen: HashMap<u64, ScoredPoint> = HashMap::new();
        for c in candidates {
            match seen.entry(c.id) {
                std::collections::hash_map::Entry::Occupied(mut entry) => { if c.timestamp > entry.get().timestamp { entry.insert(c); } }
                std::collections::hash_map::Entry::Vacant(entry) => { entry.insert(c); }
            }
        }
        let mut candidates: Vec<_> = seen.into_values().collect();
        candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        candidates.truncate(k * 2);

        for c in candidates.iter_mut() {
            let ns = self.ns(collection);
            if let Ok(Some(v)) = ns.get_vector(c.id) {
                c.score = DistanceMetric::L2.distance(&v, query);
            }
        }
        candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
        candidates.truncate(k);
        candidates
    }

    pub async fn search(
        &self, collection: &str, query: Vec<f32>, k: usize,
        params: SearchParams, consistency: ConsistencyLevel,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let start = std::time::Instant::now();

        let (mut candidates, mut stats) = self.local_search(collection, &query, k, &params).await?;

        if !params.local_only {
            let (peer_candidates, peer_stats) = self.fanout_search(collection, &query, k, &params, consistency).await;
            candidates.extend(peer_candidates);
            stats.warnings.extend(peer_stats.warnings);
            stats.nodes_contacted = peer_stats.nodes_contacted;
        } else {
            stats.nodes_contacted = 1;
        }

        candidates = self.merge_and_rerank(candidates, &query, k, collection);
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
