use rekha_client::RekhaClient as PeerRekhaClient;
use rekha_core::{NodeInfo, RekhaError, ScoredPoint, SearchParams, SearchStats};
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

#[derive(Debug, Clone)]
pub(crate) struct PeerState {
    pub info: NodeInfo,
    pub last_seen: Instant,
}

struct PeerClient {
    client: PeerRekhaClient,
    last_used: Instant,
    error_count: u64,
    collection_name: String,
}

impl PeerClient {
    async fn connect(info: &NodeInfo, collection_name: &str) -> Result<Self, RekhaError> {
        let seeds = vec![info.address.clone()];
        let client = PeerRekhaClient::connect(&seeds).await?;
        Ok(Self {
            client,
            last_used: Instant::now(),
            error_count: 0,
            collection_name: collection_name.to_string(),
        })
    }

    async fn try_search(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        self.last_used = Instant::now();
        self.client
            .search_with_params(query.to_vec(), &self.collection_name, k, params.clone())
            .await
    }
}

pub(crate) struct PeerPool {
    clients: HashMap<String, PeerClient>,
    collection_name: String,
}

impl PeerPool {
    pub fn new(collection_name: &str) -> Self {
        Self {
            clients: HashMap::new(),
            collection_name: collection_name.to_string(),
        }
    }

    pub async fn refresh(&mut self, peers: &[NodeInfo]) {
        let active: std::collections::HashSet<String> =
            peers.iter().map(|p| p.node_id.clone()).collect();
        self.clients.retain(|node_id, _| active.contains(node_id));
        for info in peers {
            if !self.clients.contains_key(&info.node_id) {
                match PeerClient::connect(info, &self.collection_name).await {
                    Ok(client) => {
                        info!("Connected to peer {} at {}", info.node_id, info.address);
                        self.clients.insert(info.node_id.clone(), client);
                    }
                    Err(e) => info!("Failed to connect to peer {}: {}", info.node_id, e),
                }
            }
        }
    }

    pub async fn search_fan_out(
        &mut self,
        query: &[f32],
        k: usize,
        params: &SearchParams,
    ) -> (Vec<ScoredPoint>, SearchStats) {
        let mut peer_params = params.clone();
        peer_params.local_only = true;
        let mut all_candidates: Vec<ScoredPoint> = Vec::new();
        let mut stats = SearchStats::default();
        let mut nodes_contacted = 0u32;

        for node_id in self.clients.keys().cloned().collect::<Vec<_>>() {
            if let Some(client) = self.clients.get_mut(&node_id) {
                match client.try_search(query, k, &peer_params).await {
                    Ok((candidates, _peer_stats)) => {
                        nodes_contacted += 1;
                        all_candidates.extend(candidates);
                        client.error_count = 0;
                    }
                    Err(_) => {
                        client.error_count += 1;
                        if client.error_count >= 3 {
                            info!("Dropping peer {} after 3 errors", node_id);
                            self.clients.remove(&node_id);
                        }
                        stats.warnings.push(format!("peer {node_id} search failed"));
                    }
                }
            }
        }

        all_candidates.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        all_candidates.truncate(k * 2);
        stats.nodes_contacted = nodes_contacted;
        (all_candidates, stats)
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }

    pub fn len(&self) -> usize {
        self.clients.len()
    }
}
