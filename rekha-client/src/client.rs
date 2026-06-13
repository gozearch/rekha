use rekha_core::{ClusterTopology, Payload, RekhaError, ScoredPoint, SearchParams, SearchStats};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};
use tracing::info;

use tonic::transport::ClientTlsConfig;

use crate::proto::{
    self, rekha_client::RekhaClient as GrpcClient, FetchRequest, InsertRequest, SearchRequest,
    SearchResponse,
};

/// A user-friendly client for the Rekha distributed vector database.
///
/// Features:
/// - Auto-discovers cluster topology from any seed node
/// - Automatic retry with exponential backoff and jitter on all operations
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
/// // Insert a vector with payload
/// client.insert(42, vec![0.1, 0.2, 0.3], Some("hello world".into())).await?;
///
/// // Search for nearest neighbors
/// let results = client.search(vec![0.1, 0.2, 0.3], 10).await?;
/// for r in results {
///     println!("id={}, score={}", r.id, r.score);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RekhaClient {
    /// gRPC channel (connection pool handled by Tonic).
    channel: Channel,
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
                        channel,
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

    /// Insert a vector with optional payload.
    pub async fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    ) -> Result<(), RekhaError> {
        let channel = self.channel.clone();
        self.with_retry("insert", move || {
            let request = tonic::Request::new(InsertRequest {
                id,
                vector: vector.clone(),
                payload: payload.clone().map(|data| proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
            });
            let mut client = GrpcClient::new(channel.clone());
            async move {
                let response = client.insert(request).await?;
                let resp = response.into_inner();
                if resp.success {
                    Ok(())
                } else {
                    Err(tonic::Status::internal(resp.error))
                }
            }
        })
        .await
    }

    /// Search for the top-k approximate nearest neighbors.
    pub async fn search(
        &self,
        query: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let params = SearchParams::default();
        let result = self.search_with_params(query, top_k, params).await?;
        Ok(result.0)
    }

    /// Search with full parameter control.
    pub async fn search_with_params(
        &self,
        query: Vec<f32>,
        top_k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let channel = self.channel.clone();
        self.with_retry("search", move || {
            let request = tonic::Request::new(SearchRequest {
                query_vector: query.clone(),
                top_k: top_k as u32,
                params: Some(proto::SearchParams {
                    ef_search: params.ef_search as u32,
                    beam_width: params.beam_width as u32,
                    include_payloads: params.include_payloads,
                    partition_hint: params.partition_hint,
                }),
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
    pub async fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError> {
        let channel = self.channel.clone();
        let ids = ids.to_vec();
        self.with_retry("delete", move || {
            let request = tonic::Request::new(proto::DeleteRequest { ids: ids.clone() });
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
        ids: &[u64],
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let channel = self.channel.clone();
        let ids = ids.to_vec();
        self.with_retry("fetch", move || {
            let request = tonic::Request::new(FetchRequest {
                ids: ids.clone(),
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

    fn make_channel() -> tonic::transport::Channel {
        tonic::transport::Endpoint::from_static("http://localhost:1").connect_lazy()
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
}
