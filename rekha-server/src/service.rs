use rekha_core::{CollectionConfig, NodeInfo, NodeStatus, Payload, RekhaError, SearchParams, VectorStoreBackend};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::coordinator::Coordinator;
use crate::proto::{
    self, rekha_server::Rekha, CollectionExistsRequest, CollectionExistsResponse,
    CreateCollectionRequest, CreateCollectionResponse, DeleteRequest, DeleteResponse,
    DropCollectionRequest, DropCollectionResponse, FetchRequest, FetchResponse,
    HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse, InsertBatchResponse,
    InsertRequest, InsertResponse, ListCollectionsRequest, ListCollectionsResponse,
    ScoredPoint, SearchDimRangeRequest, SearchDimRangeResponse, SearchRequest, SearchResponse,
    TransferShardChunk, TransferShardRequest,
};
use tokio_stream::wrappers::ReceiverStream;

/// gRPC service implementation for the Rekha distributed vector database.
pub struct RekhaService {
    coordinator: Arc<Coordinator>,
}

impl RekhaService {
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        Self { coordinator }
    }

    /// Map internal RekhaError to gRPC Status.
    pub fn map_error(e: RekhaError) -> Status {
        match &e {
            RekhaError::NotFound(_) => Status::not_found(e.to_string()),
            RekhaError::InvalidArgument(_) => Status::invalid_argument(e.to_string()),
            RekhaError::IndexFull { .. } => Status::resource_exhausted(e.to_string()),
            RekhaError::InvalidDimension { .. } => Status::invalid_argument(e.to_string()),
            RekhaError::Timeout { .. } => Status::deadline_exceeded(e.to_string()),
            RekhaError::Unavailable { .. } => Status::unavailable(e.to_string()),
            _ => Status::internal(e.to_string()),
        }
    }
}

#[tonic::async_trait]
impl Rekha for RekhaService {
    // ── Insert ────────────────────────────────────────────────
    async fn insert(
        &self,
        request: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let req = request.into_inner();
        let payload = req.payload.map(|p| Payload {
            content_type: match p.content_type.as_str() {
                "json" => rekha_core::PayloadType::Json,
                "text" => rekha_core::PayloadType::Text,
                _ => rekha_core::PayloadType::Raw,
            },
            data: p.data,
        });

        let actual_id = if req.is_replication {
            self.coordinator
                .replica_insert(&req.collection_name, req.id, &req.vector, &payload)
                .await
                .map_err(Self::map_error)?
        } else {
            self.coordinator
                .insert(&req.collection_name, req.id, req.vector, payload)
                .await
                .map_err(Self::map_error)?
        };

        Ok(Response::new(InsertResponse {
            id: actual_id,
            success: true,
            error: String::new(),
        }))
    }

    async fn insert_batch(
        &self,
        request: Request<tonic::Streaming<InsertRequest>>,
    ) -> Result<Response<InsertBatchResponse>, Status> {
        let mut stream = request.into_inner();
        let mut count = 0u64;
        let mut errors = Vec::new();

        while let Some(item) = stream.message().await? {
            let payload = item.payload.map(|p| Payload {
                content_type: match p.content_type.as_str() {
                    "json" => rekha_core::PayloadType::Json,
                    "text" => rekha_core::PayloadType::Text,
                    _ => rekha_core::PayloadType::Raw,
                },
                data: p.data,
            });

            match self.coordinator.insert(&item.collection_name, item.id, item.vector, payload).await {
                Ok(_actual_id) => count += 1,
                Err(e) => errors.push(format!("id {}: {}", item.id, e)),
            }
        }

        Ok(Response::new(InsertBatchResponse {
            inserted_count: count,
            errors,
        }))
    }

    // ── Delete ────────────────────────────────────────────────
    async fn delete(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let req = request.into_inner();
        let deleted = self
            .coordinator
            .store()
            .delete(&req.ids)
            .map_err(Self::map_error)?;

        Ok(Response::new(DeleteResponse {
            deleted_count: deleted,
            error: String::new(),
        }))
    }

    // ── Fetch ─────────────────────────────────────────────────
    async fn fetch(
        &self,
        request: Request<FetchRequest>,
    ) -> Result<Response<FetchResponse>, Status> {
        let req = request.into_inner();
        let mut vectors = Vec::new();
        let mut points = Vec::new();

        for id in &req.ids {
            match self.coordinator.store().get_vector(*id) {
                Ok(Some(data)) => {
                    vectors.push(proto::Vector { id: *id, data });
                    if req.include_payloads {
                        let payload = self.coordinator.store().get_payload(*id).ok().flatten();
                        points.push(ScoredPoint {
                            id: *id,
                            score: 0.0,
                            payload: payload.map(|data| proto::Payload {
                                content_type: "raw".into(),
                                data,
                            }),
                        });
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(Status::internal(format!("fetch error: {e}")));
                }
            }
        }

        Ok(Response::new(FetchResponse {
            vectors,
            points,
            error: String::new(),
        }))
    }

    // ── Search ────────────────────────────────────────────────
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = request.into_inner();
        let params = req.params.unwrap_or_default();

        let search_params = SearchParams {
            ef_search: params.ef_search as usize,
            nprobe: params.nprobe as usize,
            include_payloads: params.include_payloads,
            local_only: req.local_only,
        };

        match self
            .coordinator
            .search(&req.collection_name, req.query_vector, req.top_k as usize, search_params)
            .await
        {
            Ok((results, stats)) => {
                let points: Vec<ScoredPoint> = results
                    .into_iter()
                    .map(|r| ScoredPoint {
                        id: r.id,
                        score: r.score,
                        payload: r.payload.map(|p| proto::Payload {
                            content_type: p.content_type.to_string(),
                            data: p.data,
                        }),
                    })
                    .collect();

                Ok(Response::new(SearchResponse {
                    results: points,
                    stats: Some(proto::SearchStats {
                        total_ms: stats.total_ms,
                        nodes_contacted: stats.nodes_contacted,
                        vectors_scanned: stats.vectors_scanned,
                        warnings: stats.warnings,
                    }),
                }))
            }
            Err(e) => Err(Self::map_error(e)),
        }
    }

    // ── Search Stream ─────────────────────────────────────────
    type SearchStreamStream = ReceiverStream<Result<ScoredPoint, Status>>;

    async fn search_stream(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<Self::SearchStreamStream>, Status> {
        let req = request.into_inner();
        let params = req.params.unwrap_or_default();

        let search_params = SearchParams {
            ef_search: params.ef_search as usize,
            nprobe: params.nprobe as usize,
            include_payloads: params.include_payloads,
            local_only: req.local_only,
        };

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let coordinator = self.coordinator.clone();

        tokio::spawn(async move {
            let coll = req.collection_name.clone();
            match coordinator
                .search(&coll, req.query_vector, req.top_k as usize, search_params.clone())
                .await
            {
                Ok((results, _stats)) => {
                    for r in results {
                        let point = ScoredPoint {
                            id: r.id,
                            score: r.score,
                            payload: r.payload.map(|p| proto::Payload {
                                content_type: p.content_type.to_string(),
                                data: p.data,
                            }),
                        };
                        if tx.send(Ok(point)).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    // ── Cluster Management ────────────────────────────────────
    async fn handshake(
        &self,
        request: Request<HandshakeRequest>,
    ) -> Result<Response<HandshakeResponse>, Status> {
        let req = request.into_inner();
        info!("Handshake from node {} at {}", req.node_id, req.address);

        // Register the requester as a peer.
        let peer_info = NodeInfo {
            node_id: req.node_id.clone(),
            address: req.address.clone(),
            partition_id: 0,
            dim_groups: (0..4).collect(),
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: 0,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        self.coordinator.register_peer(peer_info).await;

        // Return known peers list (minus the requester).
        let peers = self.coordinator.peers_for_handshake(&req.node_id).await;
        let proto_peers = peers
            .into_iter()
            .map(|n| crate::proto::NodeInfo {
                node_id: n.node_id,
                address: n.address,
                partition_id: n.partition_id,
                dim_groups: n.dim_groups,
                storage_bytes: n.storage_bytes,
                status: format!("{:?}", n.status).to_lowercase(),
            })
            .collect();

        Ok(Response::new(HandshakeResponse {
            cluster_id: self.coordinator.cluster_id().into(),
            peers: proto_peers,
            error: String::new(),
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let peer_info = NodeInfo {
            node_id: req.node_id.clone(),
            address: req.address.clone(),
            partition_id: 0,
            dim_groups: Vec::new(),
            is_leader: false,
            raft_term: 0,
            commit_index: 0,
            storage_bytes: req.storage_bytes,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        self.coordinator.register_peer(peer_info).await;

        Ok(Response::new(HeartbeatResponse { success: true }))
    }

    async fn search_dim_range(
        &self,
        request: Request<SearchDimRangeRequest>,
    ) -> Result<Response<SearchDimRangeResponse>, Status> {
        let req = request.into_inner();
        let params = SearchParams {
            ef_search: 128,
            nprobe: req.nprobe as usize,
            include_payloads: false,
            local_only: true,
        };
        let (results, _stats) = self
            .coordinator
            .search(&req.collection_name, req.query_vector, req.top_k as usize, params)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(SearchDimRangeResponse {
            results: results
                .into_iter()
                .map(|r| ScoredPoint {
                    id: r.id,
                    score: r.score,
                    payload: r.payload.map(|p| proto::Payload {
                        content_type: p.content_type.to_string(),
                        data: p.data,
                    }),
                })
                .collect(),
        }))
    }

    async fn create_collection(
        &self, request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let config = req.config.ok_or_else(|| Status::invalid_argument("config required"))?;
        let success = if req.is_replication {
            self.coordinator
                .replicate_collection(&req.name, &config).await
                .map_err(Self::map_error)?
        } else {
            self.coordinator
                .create_collection(&req.name, config.dim, config.nlist, config.nprobe, config.replication_factor).await
                .map_err(Self::map_error)?
        };
        Ok(Response::new(CreateCollectionResponse {
            success,
            error: if success { String::new() } else { format!("collection '{}' exists", req.name) },
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        let key = format!("collection:{}", request.into_inner().name);
        self.coordinator.store().delete_metadata(&key).map_err(Self::map_error)?;
        Ok(Response::new(DropCollectionResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let entries = self.coordinator.store().iter_metadata_prefix("collection:")
            .map_err(Self::map_error)?;
        let collections: Vec<crate::proto::CollectionInfo> = entries
            .into_iter()
            .filter_map(|(_key, value)| {
                let config: CollectionConfig = serde_json::from_slice(&value).ok()?;
                let name = _key.strip_prefix("collection:")?.to_string();
                Some(crate::proto::CollectionInfo {
                    name,
                    config: Some(crate::proto::CollectionConfig {
                        dim: config.dim,
                        num_vector_shards: config.num_vector_shards,
                        replication_factor: config.replication_factor,
                        num_dim_groups: config.num_dim_groups,
                        dim_group_size: config.dim_group_size,
                        nlist: config.nlist,
                        nprobe: config.nprobe,
                        pq_num_sub_vectors: config.pq_num_sub_vectors,
                        pq_num_centroids: config.pq_num_centroids,
                        re_rank_k: config.re_rank_k,
                    }),
                    vector_count: 0,
                    index_ready: false,
                })
            })
            .collect();
        Ok(Response::new(ListCollectionsResponse { collections }))
    }

    async fn collection_exists(
        &self,
        request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        let key = format!("collection:{}", request.into_inner().name);
        let exists = self.coordinator.store().get_metadata(&key)
            .map_err(Self::map_error)?
            .is_some();
        Ok(Response::new(CollectionExistsResponse { exists }))
    }

    type TransferShardStream = tokio_stream::wrappers::ReceiverStream<Result<TransferShardChunk, Status>>;

    async fn transfer_shard(
        &self,
        request: Request<TransferShardRequest>,
    ) -> Result<Response<Self::TransferShardStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let coordinator = self.coordinator.clone();
        tokio::spawn(async move {
            let cfg_key = format!("collection:{}", req.collection_name);
            let cfg_bytes = match coordinator.store().get_metadata(&cfg_key) {
                Ok(Some(d)) => d,
                _ => { let _ = tx.send(Err(Status::not_found("collection not found"))).await; return; }
            };
            let cfg: CollectionConfig = match serde_json::from_slice(&cfg_bytes) {
                Ok(c) => c,
                Err(_) => { let _ = tx.send(Err(Status::internal("bad config"))).await; return; }
            };

            let centroids: Vec<crate::proto::Vector> = Vec::new(); // centroids rebuilt on receiver
            let nlist = cfg.nlist;
            let nprobe = cfg.nprobe;

            let col_store = coordinator.store().as_ref().clone().with_namespace(req.collection_name.clone());
            let ids = match col_store.iter_ids() {
                Ok(d) => d,
                Err(e) => { let _ = tx.send(Err(Status::internal(e.to_string()))).await; return; }
            };

            let shard_ids: Vec<u64> = ids.into_iter()
                .filter(|id| id % cfg.num_vector_shards == req.shard_id)
                .collect();

            let mut batch = Vec::new();
            let batch_size = 500u64;

            for (i, vid) in shard_ids.iter().enumerate() {
                let vec_data = match col_store.get_vector(*vid) {
                    Ok(Some(v)) => v,
                    _ => continue,
                };
                let payload = match col_store.get_payload(*vid) {
                    Ok(Some(p)) => Some(p),
                    _ => None,
                };
                batch.push(crate::proto::VectorWithCluster {
                    id: *vid,
                    data: vec_data,
                    cluster_id: 0,
                    payload,
                });

                let last_idx = shard_ids.len().saturating_sub(1);
                if batch.len() as u64 >= batch_size || i == last_idx {
                    let chunk = TransferShardChunk {
                        centroids: centroids.clone(),
                        nlist,
                        nprobe,
                        total_dim: cfg.dim,
                        vector_batches: vec![crate::proto::VectorBatch {
                            vectors: std::mem::take(&mut batch),
                        }],
                        final_chunk: i == last_idx,
                    };
                    if tx.send(Ok(chunk)).await.is_err() { break; }
                }
            }
        });

        Ok(Response::new(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_error_not_found() {
        let err = RekhaError::NotFound("test".into());
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn test_map_error_invalid_argument() {
        let err = RekhaError::InvalidArgument("bad".into());
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_map_error_index_full() {
        let err = RekhaError::IndexFull {
            capacity: 100,
            attempted: 101,
        };
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_map_error_invalid_dimension() {
        let err = RekhaError::InvalidDimension {
            expected: 768,
            actual: 64,
        };
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_map_error_timeout() {
        let err = RekhaError::Timeout {
            operation: "search",
            elapsed_ms: 5000,
        };
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
    }

    #[test]
    fn test_map_error_unavailable() {
        let err = RekhaError::Unavailable {
            detail: "down".into(),
        };
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[test]
    fn test_map_error_internal() {
        let err = RekhaError::Internal {
            detail: "oops".into(),
        };
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_map_error_storage() {
        let err = RekhaError::Storage(rekha_core::StorageError::Corruption {
            detail: "bad".into(),
        });
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_map_error_index() {
        let err = RekhaError::Index(rekha_core::IndexError::EmptyIndex);
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_new_service() {
        let config = crate::config::ServerConfig::dev_default("test-node", "/tmp/rekha_svc_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/rekha_svc_test_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);
        // service is initialized; verify by checking it doesn't panic
        let _ = service;
    }

    #[tokio::test]
    async fn test_insert_payload_content_types() {
        let config = crate::config::ServerConfig::dev_default("test-node", "/tmp/svc_ct_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/svc_ct_test_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        // Test "json" content type
        let req = tonic::Request::new(InsertRequest {
            is_replication: false,
            id: 0,
            vector: vec![0.1],
            payload: Some(crate::proto::Payload {
                content_type: "json".into(),
                data: br#"{"key":"value"}"#.to_vec(),
            }),
            collection_name: "default".into(),
        });
        let resp = service.insert(req).await.unwrap();
        assert!(resp.into_inner().success);

        // Test "text" content type
        let req = tonic::Request::new(InsertRequest {
            is_replication: false,
            id: 0,
            vector: vec![0.2],
            payload: Some(crate::proto::Payload {
                content_type: "text".into(),
                data: b"hello".to_vec(),
            }),
            collection_name: "default".into(),
        });
        let resp = service.insert(req).await.unwrap();
        assert!(resp.into_inner().success);

        // Test unknown content type (maps to Raw)
        let req = tonic::Request::new(InsertRequest {
            is_replication: false,
            id: 0,
            vector: vec![0.3],
            payload: Some(crate::proto::Payload {
                content_type: "protobuf".into(),
                data: vec![0, 1, 2],
            }),
            collection_name: "default".into(),
        });
        let resp = service.insert(req).await.unwrap();
        assert!(resp.into_inner().success);

        // Test no payload (None path)
        let req = tonic::Request::new(InsertRequest {
            is_replication: false,
            id: 0,
            vector: vec![0.4],
            payload: None,
            collection_name: "default".into(),
        });
        let resp = service.insert(req).await.unwrap();
        assert!(resp.into_inner().success);
    }

    #[tokio::test]
    async fn test_search_before_init_returns_error() {
        let config = crate::config::ServerConfig::dev_default("test-node", "/tmp/svc_search_err");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/svc_search_err_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(SearchRequest {
            query_vector: vec![0.0; 8],
            top_k: 5,
            collection_name: "default".into(),
            local_only: false,
            params: None,
        });
        let result = service.search(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn test_delete_with_empty_ids() {
        let config = crate::config::ServerConfig::dev_default("test-node", "/tmp/svc_del_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/svc_del_test_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(DeleteRequest {
            ids: vec![],
            collection_name: "default".into(),
        });
        let resp = service.delete(req).await.unwrap();
        assert_eq!(resp.into_inner().deleted_count, 0);
    }

    static NEXT_SVC_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn svc_dir(id: u64) -> String {
        format!("/tmp/rekha_svc_{pid}_{id}", pid = std::process::id())
    }

    fn make_service() -> RekhaService {
        let id = NEXT_SVC_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = svc_dir(id);
        let mut config =
            crate::config::ServerConfig::dev_default("test-node", &format!("{dir}/config"));
        let _ = std::fs::remove_dir_all(&dir);
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(format!("{dir}/db")).unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        RekhaService::new(coord)
    }

    #[tokio::test]
    async fn test_handshake_handler() {
        let service = make_service();
        let req = tonic::Request::new(HandshakeRequest {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
        });
        let resp = service.handshake(req).await.unwrap();
        let inner = resp.into_inner();
        assert_eq!(inner.cluster_id, "rekha-dev");
        // The requester should be excluded from peers list
        assert!(inner.peers.is_empty());
    }

    #[tokio::test]
    async fn test_handshake_with_known_peers() {
        let service = make_service();
        // Register a peer first
        let peer = tonic::Request::new(HandshakeRequest {
            node_id: "peer-1".into(),
            address: "10.0.0.2:50051".into(),
        });
        service.handshake(peer).await.unwrap();

        // Now a second node handshakes — should see peer-1 in the response
        let req = tonic::Request::new(HandshakeRequest {
            node_id: "peer-2".into(),
            address: "10.0.0.3:50051".into(),
        });
        let resp = service.handshake(req).await.unwrap();
        let peers = resp.into_inner().peers;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].node_id, "peer-1");
    }

    #[tokio::test]
    async fn test_heartbeat_handler() {
        let service = make_service();
        let req = tonic::Request::new(HeartbeatRequest {
            node_id: "heartbeat-node".into(),
            address: "10.0.0.5:50051".into(),
            storage_bytes: 1024,
        });
        let resp = service.heartbeat(req).await.unwrap();
        let inner = resp.into_inner();
        assert!(inner.success);
    }

    #[tokio::test]
    async fn test_fetch_handler() {
        let service = make_service();
        // Insert a vector into the service's store
        service
            .coordinator
            .store()
            .put_vector(1, &[1.0, 2.0, 3.0])
            .unwrap();
        service
            .coordinator
            .store()
            .put_payload(1, b"fetch-payload")
            .unwrap();

        let req = tonic::Request::new(FetchRequest {
            ids: vec![1],
            collection_name: "default".into(),
            include_payloads: true,
        });
        let resp = service.fetch(req).await.unwrap();
        let inner = resp.into_inner();
        assert_eq!(inner.vectors.len(), 1);
        assert_eq!(inner.vectors[0].id, 1);
        assert_eq!(inner.points.len(), 1);
    }

    #[tokio::test]
    async fn test_fetch_handler_missing_id() {
        let service = make_service();
        // Fetch a non-existent ID should succeed with empty vectors
        let req = tonic::Request::new(FetchRequest {
            ids: vec![999],
            collection_name: "default".into(),
            include_payloads: false,
        });
        let resp = service.fetch(req).await.unwrap();
        let inner = resp.into_inner();
        assert!(inner.vectors.is_empty());
    }

    #[tokio::test]
    async fn test_create_collection() {
        let service = make_service();
        let req = tonic::Request::new(CreateCollectionRequest {
            name: "test-collection".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 8,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 8,
                nlist: 16,
                nprobe: 4,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 256,
                re_rank_k: 200,
            }),
        });
        let resp = service.create_collection(req).await.unwrap();
        assert!(resp.into_inner().success);
    }

    #[tokio::test]
    async fn test_create_duplicate_collection() {
        let service = make_service();
        let req = tonic::Request::new(CreateCollectionRequest {
            name: "dup".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
        });
        let resp = service.create_collection(req).await.unwrap();
        assert!(resp.into_inner().success);
        let req2 = tonic::Request::new(CreateCollectionRequest {
            name: "dup".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
        });
        let resp = service.create_collection(req2).await.unwrap();
        assert!(!resp.into_inner().success);
    }

    #[tokio::test]
    async fn test_drop_collection() {
        let service = make_service();
        // Create first
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "to-drop".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
        });
        service.create_collection(create_req).await.unwrap();
        // Drop it
        let drop_req = tonic::Request::new(DropCollectionRequest { name: "to-drop".into() });
        let resp = service.drop_collection(drop_req).await.unwrap();
        assert!(resp.into_inner().success);
    }

    #[tokio::test]
    async fn test_list_collections() {
        let service = make_service();

        let req1 = tonic::Request::new(ListCollectionsRequest {});
        let resp = service.list_collections(req1).await.unwrap();
        assert!(resp.into_inner().collections.is_empty());

        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "list-me".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
        });
        service.create_collection(create_req).await.unwrap();

        let req2 = tonic::Request::new(ListCollectionsRequest {});
        let resp = service.list_collections(req2).await.unwrap();
        assert_eq!(resp.into_inner().collections.len(), 1);
    }

    #[tokio::test]
    async fn test_collection_exists() {
        let service = make_service();
        let req1 = tonic::Request::new(CollectionExistsRequest { name: "existent".into() });
        let resp = service.collection_exists(req1).await.unwrap();
        assert!(!resp.into_inner().exists);

        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "existent".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
        });
        service.create_collection(create_req).await.unwrap();

        let req2 = tonic::Request::new(CollectionExistsRequest { name: "existent".into() });
        let resp = service.collection_exists(req2).await.unwrap();
        assert!(resp.into_inner().exists);
    }

}
