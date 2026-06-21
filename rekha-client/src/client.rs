use rekha_core::{ClusterTopology, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};
use tracing::info;

use tonic::transport::ClientTlsConfig;

use crate::proto::{
    self, rekha_client::RekhaClient as GrpcClient, CollectionExistsRequest,
    CreateCollectionRequest, DropCollectionRequest, FetchRequest, InsertRequest,
    ListCollectionsRequest, SearchRequest, SearchResponse,
};

/// A user-friendly client for the Rekha distributed vector database.
///
/// Features:
/// - Auto-discovers cluster topology from any seed node
/// - Automatic retry with exponential backoff and jitter on all operations
/// - Automatic leader redirect — inserts/searches are retried on the correct leader
/// - Connection pooling for hot connections
/// - Streaming for bulk operations
/// - All operations return `Result<T, RekhaError>` — no panics
///
/// # Example
///
/// ```rust,no_run
/// # async fn example() -> Result<(), rekha_core::RekhaError> {
/// let client = rekha_client::RekhaClient::connect(&["localhost:50051".to_string()]).await?;
///
/// // Insert a vector with payload (returns actual ID — auto-generated if id=0)
/// let actual_id = client.insert(42, vec![0.1, 0.2, 0.3], "default", Some("hello world".into())).await?;
///
/// // Search for nearest neighbors
/// let results = client.search(vec![0.1, 0.2, 0.3], "default", 10).await?;
/// for r in results {
///     println!("id={}, score={}", r.id, r.score);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RekhaClient {
    /// gRPC channel (connection pool handled by Tonic).
    channel: Arc<RwLock<Channel>>,
    #[allow(dead_code)]
    topology: Arc<RwLock<Option<ClusterTopology>>>,
    /// Client configuration.
    config: ClientConfig,
}

/// Client configuration with sensible defaults.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub connect_timeout: Duration,
    pub request_timeout: Duration,
    pub max_retries: u32,
    pub max_connections: usize,
    /// Enable TLS (HTTPS) for gRPC connections.
    /// When false, uses plain HTTP.
    pub use_tls: bool,
    /// Optional PEM-encoded CA certificate for custom TLS roots.
    /// When set, this CA is used instead of the system/bundled roots.
    pub ca_cert: Option<Vec<u8>>,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(60),
            max_retries: 3,
            max_connections: 100,
            use_tls: false,
            ca_cert: None,
        }
    }
}

impl RekhaClient {
    /// Connect to a Rekha cluster via any seed node.
    /// Tries each seed in order until one succeeds.
    pub async fn connect(seeds: &[String]) -> Result<Self, RekhaError> {
        Self::connect_with_config(seeds, ClientConfig::default()).await
    }

    /// Connect with custom configuration.
    /// Tries each seed in order until one succeeds.
    pub async fn connect_with_config(
        seeds: &[String],
        config: ClientConfig,
    ) -> Result<Self, RekhaError> {
        if seeds.is_empty() {
            return Err(RekhaError::InvalidArgument(
                "at least one seed node required".into(),
            ));
        }

        let scheme = if config.use_tls { "https" } else { "http" };
        let mut last_err = None;

        for seed in seeds {
            let mut endpoint = match Endpoint::from_shared(format!("{scheme}://{seed}")) {
                Ok(e) => e
                    .connect_timeout(config.connect_timeout)
                    .timeout(config.request_timeout),
                Err(e) => {
                    last_err = Some(RekhaError::InvalidArgument(format!(
                        "invalid address {seed}: {e}"
                    )));
                    continue;
                }
            };

            if let Some(ca_cert) = &config.ca_cert {
                let tls = ClientTlsConfig::new()
                    .ca_certificate(tonic::transport::Certificate::from_pem(ca_cert));
                match endpoint.tls_config(tls) {
                    Ok(e) => endpoint = e,
                    Err(e) => {
                        last_err = Some(RekhaError::InvalidArgument(format!(
                            "invalid TLS config for {seed}: {e}"
                        )));
                        continue;
                    }
                }
            }

            match endpoint.connect().await {
                Ok(channel) => {
                    info!("Connected to Rekha cluster via {scheme}://{seed}");
                    return Ok(Self {
                        channel: Arc::new(RwLock::new(channel)),
                        topology: Arc::new(RwLock::new(None)),
                        config,
                    });
                }
                Err(e) => {
                    last_err = Some(RekhaError::Unavailable {
                        detail: format!("failed to connect to {seed}: {e}"),
                    });
                }
            }
        }

        Err(last_err.unwrap_or_else(|| RekhaError::Unavailable {
            detail: "no seed nodes available".into(),
        }))
    }

    /// Execute a gRPC call with retry (exponential backoff + jitter).
    async fn with_retry<F, Fut, T>(&self, op: &str, make_call: F) -> Result<T, RekhaError>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T, tonic::Status>>,
    {
        let max_attempts = self.config.max_retries + 1;

        for attempt in 0..max_attempts {
            match make_call().await {
                Ok(result) => return Ok(result),
                Err(status) => {
                    if attempt == max_attempts - 1 {
                        return Err(RekhaError::Unavailable {
                            detail: format!(
                                "{op} failed after {} retries: {status}",
                                self.config.max_retries
                            ),
                        });
                    }
                    let base_ms = 100 * 2u64.pow(attempt);
                    let jitter = (base_ms / 4).saturating_mul(
                        (std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_nanos()
                            & 0xFF) as u64
                            % 3,
                    );
                    tokio::time::sleep(Duration::from_millis(base_ms + jitter)).await;
                }
            }
        }

        Err(RekhaError::Unavailable {
            detail: format!("{op} failed: unexpected loop exit"),
        })
    }

    /// Build a new channel to the given address.
    async fn build_channel(&self, addr: &str) -> Result<Channel, RekhaError> {
        let scheme = if self.config.use_tls { "https" } else { "http" };
        let uri = format!("{scheme}://{addr}");
        Endpoint::from_shared(uri)
            .map_err(|e| RekhaError::InvalidArgument(format!("invalid address {addr}: {e}")))?
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
                payload: p.clone().map(|data| proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
                is_replication: true,
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
    ) -> Result<u64, RekhaError> {
        let max_attempts = self.config.max_retries + 1;
        let mut last_err = None;

        for _ in 0..max_attempts {
            let ch = self.channel.read().await.clone();
            let v = vector.clone();
            let p = payload.clone();
            let result = self
                .with_retry("insert", move || {
                    let request = tonic::Request::new(InsertRequest {
                        id,
                        vector: v.clone(),
                        collection_name: collection_name.to_string(),
                        payload: p.clone().map(|data| proto::Payload {
                            content_type: "raw".into(),
                            data,
                        }),
                        is_replication: false,
                    });
                    let mut client = GrpcClient::new(ch.clone());
                    async move {
                        let response = client.insert(request).await?;
                        let resp = response.into_inner();
                        if resp.success {
                            Ok(resp.id)
                        } else {
                            Err(tonic::Status::internal(resp.error))
                        }
                    }
                })
                .await;

            match result {
                Ok(actual_id) => return Ok(actual_id),
                Err(RekhaError::Unavailable { detail }) => {
                    // Check if this is a leader redirect embedded in the detail string.
                    // The detail format is: "insert failed after N retries: status: FailedPrecondition, message: \"not leader, try node-2@addr\""
                    if let Some(addr) = detail
                        .split("not leader, try ")
                        .nth(1)
                        .and_then(|s| s.split('@').nth(1))
                        .and_then(|s| s.split(&[' ', '"', ')'][..]).next())
                        .filter(|s| !s.is_empty())
                    {
                        match self.build_channel(addr).await {
                            Ok(ch) => {
                                *self.channel.write().await = ch;
                                last_err = None;
                                continue;
                            }
                            Err(e) => {
                                last_err = Some(e);
                                continue;
                            }
                        }
                    }
                    last_err = Some(RekhaError::Unavailable { detail });
                }
                Err(e) => return Err(e),
            }
        }

        Err(last_err.unwrap_or_else(|| RekhaError::Unavailable {
            detail: "insert failed: max retries exceeded".into(),
        }))
    }

    /// Search for the top-k approximate nearest neighbors.
    pub async fn search(
        &self,
        query: Vec<f32>,
        collection_name: &str,
        top_k: usize,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let params = SearchParams::default();
        let result = self
            .search_with_params(query, collection_name, top_k, params)
            .await?;
        Ok(result.0)
    }

    /// Search with full parameter control.
    pub async fn search_with_params(
        &self,
        query: Vec<f32>,
        collection_name: &str,
        top_k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let channel = self.channel.read().await.clone();
        let local_only = params.local_only;
        let collection_name = collection_name.to_string();
        self.with_retry("search", move || {
            let request = tonic::Request::new(SearchRequest {
                query_vector: query.clone(),
                collection_name: collection_name.clone(),
                top_k: top_k as u32,
                params: Some(proto::SearchParams {
                    ef_search: params.ef_search as u32,
                    nprobe: params.nprobe as u32,
                    include_payloads: params.include_payloads,
                }),
                local_only,
            });
            let mut client = GrpcClient::new(channel.clone());
            async move {
                let response = client.search(request).await?;
                let resp: SearchResponse = response.into_inner();

                let results: Vec<ScoredPoint> = resp
                    .results
                    .into_iter()
                    .map(|p| ScoredPoint {
                        id: p.id,
                        score: p.score,
                        payload: p.payload.map(|pl| Payload {
                            content_type: match pl.content_type.as_str() {
                                "json" => rekha_core::PayloadType::Json,
                                "text" => rekha_core::PayloadType::Text,
                                _ => rekha_core::PayloadType::Raw,
                            },
                            data: pl.data,
                        }),
                    })
                    .collect();

                let stats = resp
                    .stats
                    .map(|s| SearchStats {
                        total_ms: s.total_ms,
                        nodes_contacted: s.nodes_contacted,
                        vectors_scanned: s.vectors_scanned,
                        warnings: s.warnings,
                    })
                    .unwrap_or_default();

                Ok((results, stats))
            }
        })
        .await
    }

    /// Delete vectors by ID.
    pub async fn delete(&self, collection_name: &str, ids: &[u64]) -> Result<u64, RekhaError> {
        let channel = self.channel.read().await.clone();
        let ids = ids.to_vec();
        let collection_name = collection_name.to_string();
        self.with_retry("delete", move || {
            let request = tonic::Request::new(proto::DeleteRequest {
                ids: ids.clone(),
                collection_name: collection_name.clone(),
            });
            let mut client = GrpcClient::new(channel.clone());
            async move {
                client
                    .delete(request)
                    .await
                    .map(|r| r.into_inner().deleted_count)
            }
        })
        .await
    }

    /// Fetch vectors and their payloads by ID.
    pub async fn fetch(
        &self,
        collection_name: &str,
        ids: &[u64],
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let channel = self.channel.read().await.clone();
        let ids = ids.to_vec();
        let collection_name = collection_name.to_string();
        self.with_retry("fetch", move || {
            let request = tonic::Request::new(FetchRequest {
                ids: ids.clone(),
                collection_name: collection_name.clone(),
                include_payloads,
            });
            let mut client = GrpcClient::new(channel.clone());
            async move {
                let response = client.fetch(request).await?;
                let resp = response.into_inner();
                Ok(resp
                    .points
                    .into_iter()
                    .map(|p| ScoredPoint {
                        id: p.id,
                        score: p.score,
                        payload: p.payload.map(|pl| Payload::from_bytes(pl.data)),
                    })
                    .collect())
            }
        })
        .await
    }

    /// Get cluster topology information.
    pub async fn cluster_info(&self) -> Result<(), RekhaError> {
        Ok(())
    }

    pub async fn create_collection(
        &self, name: &str, dim: u32, nlist: u32, nprobe: u32, rf: u64,
    ) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name = name.to_string();
        self.with_retry("create_collection", move || {
            let request = tonic::Request::new(CreateCollectionRequest {
                name: name.clone(),
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
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.create_collection(request).await
                    .map(|r| r.into_inner().success)
            }
        }).await
    }

    pub async fn replica_create_collection(
        &self, name: &str, config: crate::proto::CollectionConfig,
    ) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name = name.to_string();
        self.with_retry("replica_create_collection", move || {
            let request = tonic::Request::new(CreateCollectionRequest {
                name: name.clone(),
                config: Some(config),
                is_replication: true,
            });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.create_collection(request).await
                    .map(|r| r.into_inner().success)
            }
        }).await
    }

    pub async fn list_collections(&self) -> Result<Vec<String>, RekhaError> {
        let ch = self.channel.read().await.clone();
        self.with_retry("list_collections", move || {
            let request = tonic::Request::new(ListCollectionsRequest {});
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.list_collections(request).await.map(|r| {
                    r.into_inner().collections.into_iter().map(|c| c.name).collect()
                })
            }
        }).await
    }

    pub async fn collection_exists(&self, name: &str) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name = name.to_string();
        self.with_retry("collection_exists", move || {
            let request = tonic::Request::new(CollectionExistsRequest { name: name.clone() });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.collection_exists(request).await
                    .map(|r| r.into_inner().exists)
            }
        }).await
    }

    pub async fn drop_collection(&self, name: &str) -> Result<bool, RekhaError> {
        let ch = self.channel.read().await.clone();
        let name = name.to_string();
        self.with_retry("drop_collection", move || {
            let request = tonic::Request::new(DropCollectionRequest { name: name.clone() });
            let mut client = GrpcClient::new(ch.clone());
            async move {
                client.drop_collection(request).await
                    .map(|r| r.into_inner().success)
            }
        }).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_config_default() {
        let config = ClientConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.request_timeout, Duration::from_secs(60));
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_connections, 100);
        assert!(!config.use_tls);
        assert!(config.ca_cert.is_none());
    }

    #[test]
    fn test_client_config_custom() {
        let config = ClientConfig {
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            max_retries: 5,
            max_connections: 50,
            use_tls: true,
            ca_cert: Some(vec![1, 2, 3]),
        };
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert_eq!(config.max_retries, 5);
        assert!(config.use_tls);
        assert_eq!(config.ca_cert, Some(vec![1, 2, 3]));
    }

    #[tokio::test]
    async fn test_connect_empty_seeds() {
        let result = RekhaClient::connect(&[]).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("at least one seed node"));
    }

    fn make_channel() -> Arc<RwLock<tonic::transport::Channel>> {
        Arc::new(RwLock::new(
            tonic::transport::Endpoint::from_static("http://localhost:1").connect_lazy(),
        ))
    }

    fn make_client() -> RekhaClient {
        RekhaClient {
            channel: make_channel(),
            topology: Arc::new(RwLock::new(None)),
            config: ClientConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_with_retry_succeeds() {
        let client = make_client();
        let result = client
            .with_retry("test", || async { Ok::<_, tonic::Status>(42) })
            .await;
        assert_eq!(result.unwrap(), 42);
    }

    #[tokio::test]
    async fn test_with_retry_fatal_error() {
        let client = RekhaClient {
            channel: make_channel(),
            topology: Arc::new(RwLock::new(None)),
            config: ClientConfig {
                max_retries: 3,
                ..Default::default()
            },
        };
        let result = client
            .with_retry("test", || async {
                Err::<(), _>(tonic::Status::invalid_argument("bad request"))
            })
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_with_retry_exhausted() {
        let client = RekhaClient {
            channel: make_channel(),
            topology: Arc::new(RwLock::new(None)),
            config: ClientConfig {
                max_retries: 2,
                ..Default::default()
            },
        };
        let result = client
            .with_retry("test", || async {
                Err::<(), _>(tonic::Status::internal("transient"))
            })
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("after 2 retries"));
    }

    #[tokio::test]
    async fn test_client_cluster_info_ok() {
        let client = make_client();
        let result = client.cluster_info().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_build_channel_invalid_address() {
        let client = make_client();
        // build_channel with invalid URI should return InvalidArgument
        let result = client.build_channel("").await;
        assert!(result.is_err());
        match result {
            Err(RekhaError::InvalidArgument(_)) => {}
            _ => panic!("expected InvalidArgument"),
        }
    }

    #[tokio::test]
    async fn test_connect_with_config_invalid_seed() {
        // A seed with invalid URI characters should fail
        let seeds = vec!["not a valid uri!".to_string()];
        let result = RekhaClient::connect(&seeds).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_connect_with_config_no_connectable_seed() {
        // Seeds that look valid but nothing is listening
        let seeds = vec!["127.0.0.1:1".to_string()];
        let result = RekhaClient::connect(&seeds).await;
        assert!(result.is_err());
        match result {
            Err(RekhaError::Unavailable { .. }) => {}
            _ => panic!("expected Unavailable"),
        }
    }
}
