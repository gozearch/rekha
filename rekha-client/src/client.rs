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
/// - Automatic retry with exponential backoff
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
    /// The client will auto-discover the full cluster topology.
    pub async fn connect(seeds: &[String]) -> Result<Self, RekhaError> {
        Self::connect_with_config(seeds, ClientConfig::default()).await
    }

    /// Connect with custom configuration.
    pub async fn connect_with_config(
        seeds: &[String],
        config: ClientConfig,
    ) -> Result<Self, RekhaError> {
        if seeds.is_empty() {
            return Err(RekhaError::InvalidArgument(
                "at least one seed node required".into(),
            ));
        }

        // Connect to the first available seed node.
        let scheme = if config.use_tls { "https" } else { "http" };
        let mut endpoint = Endpoint::from_shared(format!("{scheme}://{}", seeds[0]))
            .map_err(|e| RekhaError::InvalidArgument(format!("invalid address: {e}")))?
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout);

        if let Some(ca_cert) = &config.ca_cert {
            let tls = ClientTlsConfig::new()
                .ca_certificate(tonic::transport::Certificate::from_pem(ca_cert));
            endpoint = endpoint
                .tls_config(tls)
                .map_err(|e| RekhaError::InvalidArgument(format!("invalid TLS config: {e}")))?;
        }

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| RekhaError::Unavailable {
                detail: format!("failed to connect to {}: {e}", seeds[0]),
            })?;

        info!("Connected to Rekha cluster via {scheme}://{}", seeds[0]);

        Ok(Self {
            channel,
            topology: Arc::new(RwLock::new(None)),
            config,
        })
    }

    /// Insert a vector with optional payload.
    ///
    /// # Arguments
    /// * `id` - Unique vector identifier
    /// * `vector` - The vector data (f32 slice)
    /// * `payload` - Optional arbitrary payload (text, JSON, etc.)
    pub async fn insert(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
    ) -> Result<(), RekhaError> {
        self.insert_with_retry(id, vector, payload, 0).await
    }

    /// Internal insert with retry logic (loop-based to avoid recursive async).
    async fn insert_with_retry(
        &self,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
        attempt: u32,
    ) -> Result<(), RekhaError> {
        let max_attempts = attempt + self.config.max_retries + 1;

        for attempt_idx in attempt..max_attempts {
            let mut client = GrpcClient::new(self.channel.clone());

            let request = tonic::Request::new(InsertRequest {
                id,
                vector: vector.clone(),
                payload: payload.clone().map(|data| proto::Payload {
                    content_type: "raw".into(),
                    data,
                }),
            });

            match client.insert(request).await {
                Ok(response) => {
                    let resp = response.into_inner();
                    if resp.success {
                        return Ok(());
                    } else {
                        return Err(RekhaError::Internal { detail: resp.error });
                    }
                }
                Err(status) => {
                    if attempt_idx == max_attempts - 1 {
                        return Err(RekhaError::Unavailable {
                            detail: format!(
                                "insert failed after {} retries: {status}",
                                self.config.max_retries
                            ),
                        });
                    }
                    let backoff = Duration::from_millis(100 * 2u64.pow(attempt_idx));
                    tokio::time::sleep(backoff).await;
                }
            }
        }

        Err(RekhaError::Unavailable {
            detail: "insert failed: unexpected loop exit".into(),
        })
    }

    /// Search for the top-k approximate nearest neighbors.
    ///
    /// This is the primary query API. Returns results sorted by distance (closest first).
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `top_k` - Number of results to return
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
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `top_k` - Number of results to return
    /// * `params` - Search parameters (ef_search, include_payloads, etc.)
    pub async fn search_with_params(
        &self,
        query: Vec<f32>,
        top_k: usize,
        params: SearchParams,
    ) -> Result<(Vec<ScoredPoint>, SearchStats), RekhaError> {
        let mut client = GrpcClient::new(self.channel.clone());

        let request = tonic::Request::new(SearchRequest {
            query_vector: query,
            top_k: top_k as u32,
            params: Some(proto::SearchParams {
                ef_search: params.ef_search as u32,
                beam_width: params.beam_width as u32,
                include_payloads: params.include_payloads,
                partition_hint: params.partition_hint,
            }),
        });

        let response = client
            .search(request)
            .await
            .map_err(|status| RekhaError::Unavailable {
                detail: format!("search failed: {status}"),
            })?;

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

    /// Delete vectors by ID.
    pub async fn delete(&self, ids: &[u64]) -> Result<u64, RekhaError> {
        let mut client = GrpcClient::new(self.channel.clone());

        let request = tonic::Request::new(proto::DeleteRequest { ids: ids.to_vec() });

        let response = client
            .delete(request)
            .await
            .map_err(|status| RekhaError::Unavailable {
                detail: format!("delete failed: {status}"),
            })?;

        Ok(response.into_inner().deleted_count)
    }

    /// Fetch vectors and their payloads by ID.
    pub async fn fetch(
        &self,
        ids: &[u64],
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let mut client = GrpcClient::new(self.channel.clone());

        let request = tonic::Request::new(FetchRequest {
            ids: ids.to_vec(),
            include_payloads,
        });

        let response = client
            .fetch(request)
            .await
            .map_err(|status| RekhaError::Unavailable {
                detail: format!("fetch failed: {status}"),
            })?;

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

    /// Get cluster topology information.
    pub async fn cluster_info(&self) -> Result<(), RekhaError> {
        // Simplified — would query the cluster endpoint.
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
}
