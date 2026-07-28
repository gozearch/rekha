use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rekha_core::RekhaError;
use rekha_proto::proto::{self, rekha_client::RekhaClient};
use tokio::sync::RwLock;
use tonic::transport::Channel;

struct PeerConnection {
    client: RekhaClient<Channel>,
    #[allow(dead_code)]
    last_used: Instant,
    error_count: u64,
}

pub struct PeerPool {
    peers: Arc<RwLock<HashMap<String, PeerConnection>>>,
}

impl Default for PeerPool {
    fn default() -> Self {
        Self::new()
    }
}

impl PeerPool {
    pub fn new() -> Self {
        PeerPool {
            peers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn connect(&self, address: &str) -> Result<RekhaClient<Channel>, RekhaError> {
        let addr = if !address.starts_with("http://") && !address.starts_with("https://") {
            format!("http://{}", address)
        } else {
            address.to_string()
        };
        let channel = Channel::from_shared(addr)
            .map_err(|e| RekhaError::InvalidArgument(e.to_string()))?
            .connect_timeout(Duration::from_secs(5))
            .connect()
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?;
        Ok(RekhaClient::new(channel))
    }

    pub async fn get(&self, address: &str) -> Result<RekhaClient<Channel>, RekhaError> {
        {
            let peers = self.peers.read().await;
            if let Some(conn) = peers.get(address) {
                return Ok(conn.client.clone());
            }
        }
        let client = self.connect(address).await?;
        let mut peers = self.peers.write().await;
        peers.insert(
            address.to_string(),
            PeerConnection {
                client: client.clone(),
                last_used: Instant::now(),
                error_count: 0,
            },
        );
        Ok(client)
    }

    pub async fn replica_insert(
        &self,
        address: &str,
        req: proto::InsertRequest,
    ) -> Result<proto::InsertResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .insert(tonic::Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn replica_delete(
        &self,
        address: &str,
        req: proto::DeleteRequest,
    ) -> Result<proto::DeleteResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .delete(tonic::Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn replica_create_collection(
        &self,
        address: &str,
        req: proto::CreateCollectionRequest,
    ) -> Result<proto::CreateCollectionResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .create_collection(tonic::Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn replica_drop_collection(
        &self,
        address: &str,
        req: proto::DropCollectionRequest,
    ) -> Result<proto::DropCollectionResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .drop_collection(tonic::Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn remote_search(
        &self,
        address: &str,
        req: proto::SearchRequest,
    ) -> Result<proto::SearchResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .search(tonic::Request::new(req))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn find_successor(
        &self,
        address: &str,
        id: u128,
    ) -> Result<proto::FindSuccessorResponse, RekhaError> {
        let mut client = self.get(address).await?;
        client
            .find_successor(tonic::Request::new(proto::FindSuccessorRequest {
                id: id.to_le_bytes().to_vec(),
            }))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))
    }

    pub async fn get_predecessor(
        &self,
        address: &str,
    ) -> Result<Option<proto::NodeInfo>, RekhaError> {
        let mut client = self.get(address).await?;
        let resp = client
            .get_predecessor(tonic::Request::new(proto::GetPredecessorRequest {}))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?;
        Ok(resp.predecessor)
    }

    pub async fn notify_chord(
        &self,
        address: &str,
        node: proto::NodeInfo,
    ) -> Result<bool, RekhaError> {
        let mut client = self.get(address).await?;
        let resp = client
            .notify_chord(tonic::Request::new(proto::NotifyChordRequest {
                node: Some(node),
            }))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?;
        Ok(resp.success)
    }

    pub async fn evict(&self, address: &str) {
        let mut peers = self.peers.write().await;
        peers.remove(address);
    }

    pub async fn peer_addresses(&self) -> Vec<String> {
        let peers = self.peers.read().await;
        peers.keys().cloned().collect()
    }

    pub async fn record_error(&self, address: &str) -> u64 {
        let mut peers = self.peers.write().await;
        if let Some(conn) = peers.get_mut(address) {
            conn.error_count += 1;
            conn.error_count
        } else {
            0
        }
    }
}
