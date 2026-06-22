use rekha_core::{ConsistencyLevel, NodeInfo, RekhaError};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::proto::{
    rekha_client::RekhaClient as GrpcClient, CollectionExistsRequest, CreateCollectionRequest,
    DeleteRequest, DropCollectionRequest, InsertRequest, ListCollectionsRequest, SearchRequest,
    SearchParams as ProtoSearchParams,
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub seeds: Vec<String>,
    pub max_retries: u32,
    pub request_timeout: Duration,
    pub connect_timeout: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            seeds: vec!["127.0.0.1:50051".into()],
            max_retries: 3,
            request_timeout: DEFAULT_TIMEOUT,
            connect_timeout: CONNECT_TIMEOUT,
        }
    }
}

pub struct RekhaClient {
    config: ClientConfig,
    channel: Arc<RwLock<tonic::transport::Channel>>,
}

impl RekhaClient {
    pub async fn connect(seeds: &[String]) -> Result<Self, RekhaError> {
        let config = ClientConfig {
            seeds: seeds.to_vec(),
            ..Default::default()
        };
        Self::connect_with_config(config).await
    }

    pub async fn connect_with_config(config: ClientConfig) -> Result<Self, RekhaError> {
        let endpoint_string = format!("http://{}", config.seeds[0]);
        let channel = tonic::transport::Channel::from_shared(endpoint_string.clone())
            .map_err(|e| RekhaError::Unavailable {
                detail: format!("invalid endpoint {endpoint_string}: {e}"),
            })?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout)
            .connect()
            .await
            .map_err(|e| RekhaError::Unavailable {
                detail: format!("failed to connect to {}: {e}", config.seeds[0]),
            })?;

        Ok(Self {
            config,
            channel: Arc::new(RwLock::new(channel)),
        })
    }

    async fn with_retry<F, Fut, T>(&self, _op_name: &str, f: F) -> Result<T, RekhaError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, tonic::Status>>,
    {
        let max_attempts = self.config.max_retries + 1;
        let mut last_err = None;

        for attempt in 0..max_attempts {
            match f().await {
                Ok(val) => return Ok(val),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 < max_attempts {
                        tokio::time::sleep(Duration::from_millis(50 * (attempt as u64 + 1))).await;
                    }
                }
            }
        }

        Err(RekhaError::Unavailable {
            detail: format!("operation failed after {max_attempts} attempts: {:?}", last_err),
        })
    }

    async fn resolve_connect(&self, addr: &str) -> Result<tonic::transport::Channel, RekhaError> {
        let endpoint_string = format!("http://{addr}");
        tonic::transport::Channel::from_shared(endpoint_string.clone())
            .map_err(|e| RekhaError::Unavailable {
                detail: format!("invalid endpoint {endpoint_string}: {e}"),
            })?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .connect()
            .await
            .map_err(|e| RekhaError::Unavailable {
                detail: format!("failed to connect to {addr}: {e}"),
            })
    }

    /// Insert a vector with optional payload.
    /// Automatically follows leader redirects.
    pub async fn replica_insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        collection_name: &str,
        payload: Option<Vec<u8>>,
        timestamp: u64,
    ) -> Result<u64, RekhaError> {
        let ch = self.channel.read().await.clone();
        let v = vector.clone();
        let p = payload.clone();
        let cn = collection_name.to_string();
        self.with_retry("replica_insert", move || {
            let request = tonic::Request::new(InsertRequest {
                id,
                vector: v.clone(),
                collection_name: cn.clone(),
                payload: p.clone().map(|data| crate::proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
                is_replication: true,
                timestamp,
                consistency: 0,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.insert(request).await?.into_inner();
                if resp.success { Ok(resp.id) }
                else { Err(tonic::Status::internal(resp.error)) }
            }
        }).await
    }

    pub async fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        collection_name: &str,
        payload: Option<Vec<u8>>,
        consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let max_attempts = self.config.max_retries + 1;
        let mut last_err = None;
        let cl_val = consistency.to_i32();

        for _ in 0..max_attempts {
            let ch = self.channel.read().await.clone();
            let v = vector.clone();
            let p = payload.clone();
            let cn = collection_name.to_string();

            let request = tonic::Request::new(InsertRequest {
                id,
                vector: v,
                collection_name: cn,
                payload: p.map(|data| crate::proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
                is_replication: false,
                timestamp: 0,
                consistency: consistency.to_i32(),
            });
            let mut client = GrpcClient::new(ch);

            match client.insert(request).await {
                Ok(resp) => {
                    let inner = resp.into_inner();
                    if inner.success {
                        return Ok(inner.id);
                    }
                    return Err(RekhaError::Internal {
                        detail: inner.error,
                    });
                }
                Err(e) => {
                    last_err = Some(e);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }

        Err(RekhaError::Unavailable {
            detail: format!("insert failed after {max_attempts} attempts: {:?}", last_err),
        })
    }

    pub async fn replica_create_collection(
        &self, name: &str, config: crate::proto::CollectionConfig, timestamp: u64,
    ) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let cn = name.to_string();
        self.with_retry("replica_create_collection", move || {
            let request = tonic::Request::new(CreateCollectionRequest {
                name: cn.clone(),
                config: Some(config.clone()),
                is_replication: true,
                timestamp,
                consistency: 0,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.create_collection(request).await?.into_inner();
                Ok(resp.success)
            }
        }).await
    }

    pub async fn replica_drop_collection(&self, name: &str, timestamp: u64) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let cn = name.to_string();
        self.with_retry("replica_drop_collection", move || {
            let request = tonic::Request::new(DropCollectionRequest {
                name: cn.clone(),
                is_replication: true,
                timestamp,
                consistency: 0,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.drop_collection(request).await?.into_inner();
                Ok(resp.success)
            }
        }).await
    }

    pub async fn search_with_params(
        &self,
        query: Vec<f32>,
        collection_name: &str,
        k: usize,
        params: rekha_core::SearchParams,
        consistency: ConsistencyLevel,
    ) -> Result<(Vec<rekha_core::ScoredPoint>, rekha_core::SearchStats), RekhaError> {
        let ch = self.channel.read().await.clone();
        let q = query.clone();
        let cn = collection_name.to_string();
        let proto_params = ProtoSearchParams {
            ef_search: params.ef_search as u32,
            nprobe: params.nprobe as u32,
            include_payloads: params.include_payloads,
        };

        self.with_retry("search", move || {
            let request = tonic::Request::new(SearchRequest {
                query_vector: q.clone(),
                top_k: k as u32,
                collection_name: cn.clone(),
                local_only: params.local_only,
                params: Some(proto_params.clone()),
                consistency: 0,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.search(request).await?.into_inner();
                let results: Vec<rekha_core::ScoredPoint> = resp
                    .results
                    .into_iter()
                    .map(|r| rekha_core::ScoredPoint {
                        id: r.id,
                        score: r.score,
                        payload: r.payload.map(|p| rekha_core::Payload {
                            content_type: match p.content_type.as_str() {
                                "json" => rekha_core::PayloadType::Json,
                                "text" => rekha_core::PayloadType::Text,
                                _ => rekha_core::PayloadType::Raw,
                            },
                            data: p.data,
                        }),
                        timestamp: r.timestamp,
                    })
                    .collect();
                let stats = resp.stats.unwrap_or_default();
                Ok((
                    results,
                    rekha_core::SearchStats {
                        total_ms: stats.total_ms,
                        nodes_contacted: stats.nodes_contacted,
                        vectors_scanned: stats.vectors_scanned,
                        warnings: stats.warnings,
                    },
                ))
            }
        }).await
    }

    pub async fn delete(&self, ids: &[u64], collection_name: &str, consistency: ConsistencyLevel) -> Result<u64, RekhaError> {
        let ch = self.channel.read().await.clone();
        let ids_vec = ids.to_vec();
        let cn = collection_name.to_string();
        self.with_retry("delete", move || {
            let request = tonic::Request::new(DeleteRequest {
                ids: ids_vec.clone(),
                collection_name: cn.clone(),
                timestamp: 0,
                consistency: consistency.to_i32(),
                is_replication: false,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.delete(request).await?.into_inner();
                Ok(resp.deleted_count)
            }
        }).await
    }

    pub async fn replica_delete(
        &self,
        ids: &[u64],
        collection_name: &str,
        timestamp: u64,
    ) -> Result<u64, RekhaError> {
        let ch = self.channel.read().await.clone();
        let ids_vec = ids.to_vec();
        let cn = collection_name.to_string();
        self.with_retry("replica_delete", move || {
            let request = tonic::Request::new(DeleteRequest {
                ids: ids_vec.clone(),
                collection_name: cn.clone(),
                timestamp,
                consistency: 0,
                is_replication: true,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.delete(request).await?.into_inner();
                Ok(resp.deleted_count)
            }
        }).await
    }

    pub async fn search(
        &self,
        query: Vec<f32>,
        collection_name: &str,
        k: usize,
        consistency: ConsistencyLevel,
    ) -> Result<Vec<rekha_core::ScoredPoint>, RekhaError> {
        let (results, _) = self
            .search_with_params(query, collection_name, k, rekha_core::SearchParams::default(), consistency)
            .await?;
        Ok(results)
    }

    pub async fn get_node_info(&self) -> Result<NodeInfo, RekhaError> {
        let ch = self.channel.read().await.clone();
        self.with_retry("get_node_info", move || {
            use crate::proto::rekha_client::RekhaClient as GrpcClientInner;
            let request = tonic::Request::new(crate::proto::HandshakeRequest {
                node_id: String::new(),
                address: String::new(),
            });
            let mut client = GrpcClientInner::new(ch.clone());
            async move {
                let resp = client.handshake(request).await?.into_inner();
                Ok(NodeInfo {
                    node_id: resp.cluster_id,
                    address: String::new(),
                    partition_id: 0,
                    dim_groups: Vec::new(),
                    is_leader: false,
                    raft_term: 0,
                    commit_index: 0,
                    storage_bytes: 0,
                    status: rekha_core::NodeStatus::Healthy,
                    last_heartbeat: 0,
                })
            }
        }).await
    }

    pub async fn cluster_info(&self) -> Result<NodeInfo, RekhaError> {
        self.get_node_info().await
    }

    pub async fn create_collection(
        &self, name: &str, dim: u32, nlist: u32, nprobe: u32, rf: u64,
        consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name_str = name.to_string();
        self.with_retry("create_collection", move || {
            let request = tonic::Request::new(CreateCollectionRequest {
                name: name_str.clone(),
                config: Some(crate::proto::CollectionConfig {
                    dim, nlist, nprobe,
                    num_vector_shards: 6,
                    replication_factor: rf,
                    num_dim_groups: 4,
                    dim_group_size: dim / 4,
                    pq_num_sub_vectors: 4,
                    pq_num_centroids: 256,
                    re_rank_k: 256,
                }),
                is_replication: false,
                timestamp: 0,
                consistency: consistency.to_i32(),
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.create_collection(request).await.map(|r| r.into_inner().success)
            }
        }).await
    }

    pub async fn drop_collection(&self, name: &str, consistency: ConsistencyLevel) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name_str = name.to_string();
        self.with_retry("drop_collection", move || {
            let request = tonic::Request::new(DropCollectionRequest {
                name: name_str.clone(),
                is_replication: false,
                timestamp: 0,
                consistency: consistency.to_i32(),
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.drop_collection(request).await.map(|r| r.into_inner().success)
            }
        }).await
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, RekhaError> {
        let ch = self.channel.read().await.clone();
        self.with_retry("list_collections", move || {
            let request = tonic::Request::new(ListCollectionsRequest {});
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.list_collections(request).await?.into_inner();
                Ok(resp.collections.into_iter().map(|c| c.name).collect())
            }
        }).await
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let cn = name.to_string();
        self.with_retry("collection_exists", move || {
            let request = tonic::Request::new(CollectionExistsRequest {
                name: cn.clone(),
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                let resp = client.collection_exists(request).await?.into_inner();
                Ok(resp.exists)
            }
        }).await
    }
}
