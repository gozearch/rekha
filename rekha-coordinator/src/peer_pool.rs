use rekha_core::{ConsistencyLevel, NodeInfo, RekhaError, ScoredPoint, SearchParams, SearchStats};
use rekha_client::RekhaClient as PeerRekhaClient;
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

pub(super) struct PeerClient {
    #[allow(dead_code)]
    pub(super) info: NodeInfo,
    pub(super) client: PeerRekhaClient,
    pub(super) last_used: Instant,
    pub(super) error_count: u64,
}

impl PeerClient {
    pub(super) async fn connect(info: &NodeInfo) -> Result<Self, RekhaError> {
        let seeds = vec![info.address.clone()];
        let client = PeerRekhaClient::connect(&seeds).await?;
        Ok(Self { info: info.clone(), client, last_used: Instant::now(), error_count: 0 })
    }

    pub(super) async fn try_search(
        &mut self, query: &[f32], k: usize, params: &SearchParams, collection: &str,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        self.last_used = Instant::now();
        self.client.search_with_params(query.to_vec(), collection, k, params.clone(), ConsistencyLevel::One).await
    }

    pub(super) async fn try_remote_insert(
        &mut self, collection: &str, id: u64, vector: &[f32], payload: &Option<Vec<u8>>, timestamp: u64,
    ) -> Result<(), RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_insert(id, vector.to_vec(), collection, payload.clone(), timestamp).await?;
        Ok(())
    }

    pub(super) async fn try_remote_create_collection(
        &mut self, name: &str, config: &rekha_proto::proto::CollectionConfig, timestamp: u64,
    ) -> Result<bool, RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_create_collection(name, config.clone(), timestamp).await
    }

    pub(super) async fn try_remote_drop_collection(&mut self, name: &str, timestamp: u64) -> Result<bool, RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_drop_collection(name, timestamp).await
    }

    pub(super) async fn try_remote_delete(
        &mut self, collection: &str, ids: &[u64], timestamp: u64,
    ) -> Result<(), RekhaError> {
        self.last_used = Instant::now();
        self.client.replica_delete(ids, collection, timestamp).await?;
        Ok(())
    }
}

pub(super) struct PeerPool {
    pub(super) clients: HashMap<String, PeerClient>,
}

impl PeerPool {
    pub(super) fn new() -> Self { Self { clients: HashMap::new() } }

    pub(super) async fn refresh(&mut self, peers: &[NodeInfo]) {
        let active: std::collections::HashSet<String> = peers.iter().map(|p| p.node_id.clone()).collect();
        self.clients.retain(|node_id, _| active.contains(node_id));
        for info in peers {
            if !self.clients.contains_key(&info.node_id) {
                match PeerClient::connect(info).await {
                    Ok(client) => {
                        info!("Connected to peer {} at {}", info.node_id, info.address);
                        self.clients.insert(info.node_id.clone(), client);
                    }
                    Err(e) => info!("Failed to connect to peer {}: {}", info.node_id, e),
                }
            }
        }
    }
}
