use rekha_core::{now_micros, CollectionConfig, CollectionMeta, ConsistencyLevel, NodeInfo, NodeStatus, RekhaError, SearchParams, VectorStoreBackend};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use rekha_coordinator::Coordinator;
use crate::proto::{
    self, rekha_server::Rekha, CollectionExistsRequest, CollectionExistsResponse,
    CreateCollectionRequest, CreateCollectionResponse, DeleteRequest, DeleteResponse,
    DropCollectionRequest, DropCollectionResponse, FetchRequest, FetchResponse,
    HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse, InsertBatchResponse,
    InsertRequest, InsertResponse, ListCollectionsRequest, ListCollectionsResponse,
    RepairCollectionRequest, RepairProgress,
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
        let payload = req.payload.map(rekha_core::Payload::from);

        let actual_id = if req.is_replication {
            self.coordinator
                .replica_insert(&req.collection_name, req.id, &req.vector, &payload, req.timestamp)
                .await
                .map_err(Self::map_error)?
        } else {
            self.coordinator
                .insert(
                    &req.collection_name,
                    req.id,
                    req.vector,
                    payload,
                    req.timestamp,
                    self.coordinator.resolve_consistency(req.consistency),
                )
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
            let payload = item.payload.map(rekha_core::Payload::from);

            match self.coordinator.insert(
                &item.collection_name,
                item.id,
                item.vector,
                payload,
                item.timestamp,
                self.coordinator.resolve_consistency(item.consistency),
            ).await {
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
        let consistency = self.coordinator.resolve_consistency(req.consistency);
        let deleted = self
            .coordinator
            .delete(&req.collection_name, &req.ids, req.timestamp, consistency)
            .await
            .map_err(Self::map_error)?;

        Ok(Response::new(DeleteResponse {
            deleted_count: deleted,
            error: String::new(),
            timestamp: req.timestamp,
        }))
    }

    // ── Fetch ─────────────────────────────────────────────────
    async fn fetch(
        &self,
        request: Request<FetchRequest>,
    ) -> Result<Response<FetchResponse>, Status> {
        let req = request.into_inner();
        let consistency = self.coordinator.resolve_consistency(req.consistency);
        let records = self
            .coordinator
            .fetch(&req.collection_name, &req.ids, consistency)
            .await
            .map_err(Self::map_error)?;

        let mut vectors = Vec::new();
        let mut points = Vec::new();

        for record in records {
            if let Some(data) = record.data {
                vectors.push(proto::Vector {
                    id: record.id,
                    data,
                    timestamp: record.timestamp,
                });
                if req.include_payloads {
                    let payload = self.coordinator.store().get_payload(record.id).ok().flatten();
                    points.push(ScoredPoint {
                        id: record.id,
                        score: 0.0,
                        payload: payload.map(|data| proto::Payload {
                            content_type: "raw".into(),
                            data,
                        }),
                        timestamp: record.timestamp,
                    });
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

        let consistency = self.coordinator.resolve_consistency(req.consistency);

        match self
            .coordinator
            .search(&req.collection_name, req.query_vector, req.top_k as usize, search_params, consistency)
            .await
        {
            Ok((results, stats)) => {
                let points: Vec<ScoredPoint> = results
                    .into_iter()
                    .map(proto::ScoredPoint::from)
                    .collect();

                Ok(Response::new(SearchResponse {
                    results: points,
                    stats: Some(proto::SearchStats::from(stats)),
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

        let mut search_params = rekha_core::SearchParams::from(params);
        search_params.local_only = req.local_only;

        let consistency = self.coordinator.resolve_consistency(req.consistency);

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let coordinator = self.coordinator.clone();

        tokio::spawn(async move {
            let coll = req.collection_name.clone();
            match coordinator
                .search(&coll, req.query_vector, req.top_k as usize, search_params, consistency)
                .await
            {
                Ok((results, _stats)) => {
                    for r in results {
                        let point = proto::ScoredPoint::from(r);
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
        let proto_peers: Vec<crate::proto::NodeInfo> = peers
            .into_iter()
            .map(crate::proto::NodeInfo::from)
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
        let params = rekha_core::SearchParams {
            ef_search: 128,
            nprobe: req.nprobe as usize,
            include_payloads: false,
            local_only: true,
        };
        let (results, _stats) = self
            .coordinator
            .search(&req.collection_name, req.query_vector, req.top_k as usize, params, ConsistencyLevel::One)
            .await
            .map_err(Self::map_error)?;
        Ok(Response::new(SearchDimRangeResponse {
            results: results
                .into_iter()
                .map(proto::ScoredPoint::from)
                .collect(),
        }))
    }

    async fn create_collection(
        &self, request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let config = req.config.ok_or_else(|| Status::invalid_argument("config required"))?;
        let timestamp = if req.timestamp == 0 { now_micros() } else { req.timestamp };
        let success = if req.is_replication {
            self.coordinator
                .replicate_collection(&req.name, &config, timestamp).await
                .map_err(Self::map_error)?
        } else {
            let consistency = self.coordinator.resolve_consistency(req.consistency);
            self.coordinator
                .create_collection(&req.name, config.dim, config.nlist, config.nprobe, config.replication_factor, timestamp, consistency).await
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
        let req = request.into_inner();
        let timestamp = if req.timestamp == 0 { now_micros() } else { req.timestamp };
        let success = if req.is_replication {
            self.coordinator.replicate_drop_collection(&req.name, timestamp).await
                .map_err(Self::map_error)?
        } else {
            let consistency = self.coordinator.resolve_consistency(req.consistency);
            self.coordinator.drop_collection(&req.name, timestamp, consistency).await
                .map_err(Self::map_error)?
        };
        Ok(Response::new(DropCollectionResponse { success, error: String::new() }))
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
                let name = _key.strip_prefix("collection:")?.to_string();

                // Try new CollectionMeta format
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&value) {
                    if meta.is_deleted { return None; }
                    return Some(crate::proto::CollectionInfo {
                        name,
                        config: Some(crate::proto::CollectionConfig::from(meta.config)),
                        vector_count: meta.vector_count,
                        index_ready: false,
                        config_timestamp: meta.timestamp,
                    });
                }

                // Fall back to old CollectionConfig format
                let config: CollectionConfig = serde_json::from_slice(&value).ok()?;
                Some(crate::proto::CollectionInfo {
                    name,
                    config: Some(crate::proto::CollectionConfig::from(config)),
                    vector_count: 0,
                    index_ready: false,
                    config_timestamp: 0,
                })
            })
            .collect();
        Ok(Response::new(ListCollectionsResponse { collections }))
    }

    async fn collection_exists(
        &self,
        request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        let req = request.into_inner();
        let key = format!("collection:{}", req.name);
        let exists = match self.coordinator.store().get_metadata(&key).map_err(Self::map_error)? {
            Some(data) => {
                if let Ok(meta) = serde_json::from_slice::<CollectionMeta>(&data) {
                    !meta.is_deleted
                } else {
                    // Old format without CollectionMeta — treat as live
                    true
                }
            }
            None => false,
        };
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

    // ── Repair Collection ─────────────────────────────────────
    type RepairCollectionStream = ReceiverStream<Result<RepairProgress, Status>>;

    async fn repair_collection(
        &self,
        request: Request<RepairCollectionRequest>,
    ) -> Result<Response<Self::RepairCollectionStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        let coordinator = self.coordinator.clone();
        tokio::spawn(async move {
            let collection = req.collection_name;
            let ns = coordinator.store().as_ref().clone().with_namespace(collection.clone());

            let ids = match ns.iter_ids() {
                Ok(ids) => ids,
                Err(e) => {
                    let _ = tx.send(Err(Status::internal(e.to_string()))).await;
                    return;
                }
            };

            let total = ids.len() as u64;
            let mut repaired = 0u64;

            for id in &ids {
                if let Ok(Some(record)) = ns.get_vector_record(*id) {
                    let _ = record;
                    repaired += 1;
                }
                if total > 0 && repaired % 100 == 0 {
                    let _ = tx.send(Ok(RepairProgress {
                        repaired,
                        total,
                        current_node: coordinator.node_id().into(),
                    })).await;
                }
            }

            let _ = tx.send(Ok(RepairProgress {
                repaired,
                total,
                current_node: coordinator.node_id().into(),
            })).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
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
        let dir = tempfile::TempDir::new().unwrap();
        let config = crate::config::ServerConfig::dev_default("test-node", dir.path().to_string_lossy().as_ref());
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(dir.path().join("db")).unwrap(),
        );
        let coord = std::sync::Arc::new(rekha_coordinator::Coordinator::new(rekha_coordinator::CoordinatorConfig { node_id: config.cluster.node_id.clone(), bind_addr: config.cluster.bind_addr.clone(), seed_nodes: config.cluster.seed_nodes.clone(), default_write_consistency: config.cluster.default_write_consistency.clone(), hinted_handoff_enabled: config.cluster.hinted_handoff_enabled, max_hint_window_secs: config.cluster.max_hint_window_secs, gc_grace_seconds: config.storage.gc_grace_seconds, peer_timeout_ms: 10000 }, store));
        let service = RekhaService::new(coord);
        // service is initialized; verify by checking it doesn't panic
        let _ = service;
    }

    #[tokio::test]
    async fn test_insert_payload_content_types() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = crate::config::ServerConfig::dev_default("test-node", dir.path().to_string_lossy().as_ref());
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(dir.path().join("db")).unwrap(),
        );
        let coord = std::sync::Arc::new(rekha_coordinator::Coordinator::new(rekha_coordinator::CoordinatorConfig { node_id: config.cluster.node_id.clone(), bind_addr: config.cluster.bind_addr.clone(), seed_nodes: config.cluster.seed_nodes.clone(), default_write_consistency: config.cluster.default_write_consistency.clone(), hinted_handoff_enabled: config.cluster.hinted_handoff_enabled, max_hint_window_secs: config.cluster.max_hint_window_secs, gc_grace_seconds: config.storage.gc_grace_seconds, peer_timeout_ms: 10000 }, store));
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
            timestamp: 0,
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
        });
        let resp = service.insert(req).await.unwrap();
        assert!(resp.into_inner().success);
    }

    #[tokio::test]
    async fn test_search_before_init_returns_error() {
        let dir = tempfile::TempDir::new().unwrap();
        let config = crate::config::ServerConfig::dev_default("test-node", dir.path().to_string_lossy().as_ref());
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(dir.path().join("db")).unwrap(),
        );
        let coord = std::sync::Arc::new(rekha_coordinator::Coordinator::new(rekha_coordinator::CoordinatorConfig { node_id: config.cluster.node_id.clone(), bind_addr: config.cluster.bind_addr.clone(), seed_nodes: config.cluster.seed_nodes.clone(), default_write_consistency: config.cluster.default_write_consistency.clone(), hinted_handoff_enabled: config.cluster.hinted_handoff_enabled, max_hint_window_secs: config.cluster.max_hint_window_secs, gc_grace_seconds: config.storage.gc_grace_seconds, peer_timeout_ms: 10000 }, store));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(SearchRequest {
            query_vector: vec![0.0; 8],
            top_k: 5,
            collection_name: "default".into(),
            local_only: false,
            params: None,
            consistency: 0,
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
        let coord = std::sync::Arc::new(rekha_coordinator::Coordinator::new(rekha_coordinator::CoordinatorConfig { node_id: config.cluster.node_id.clone(), bind_addr: config.cluster.bind_addr.clone(), seed_nodes: config.cluster.seed_nodes.clone(), default_write_consistency: config.cluster.default_write_consistency.clone(), hinted_handoff_enabled: config.cluster.hinted_handoff_enabled, max_hint_window_secs: config.cluster.max_hint_window_secs, gc_grace_seconds: config.storage.gc_grace_seconds, peer_timeout_ms: 10000 }, store));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(DeleteRequest {
            ids: vec![],
            collection_name: "default".into(),
            timestamp: 0,
            consistency: 0,
            is_replication: false,
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
        let config =
            crate::config::ServerConfig::dev_default("test-node", &format!("{dir}/config"));
        let _ = std::fs::remove_dir_all(&dir);
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(format!("{dir}/db")).unwrap(),
        );
        let coord = std::sync::Arc::new(rekha_coordinator::Coordinator::new(rekha_coordinator::CoordinatorConfig { node_id: config.cluster.node_id.clone(), bind_addr: config.cluster.bind_addr.clone(), seed_nodes: config.cluster.seed_nodes.clone(), default_write_consistency: config.cluster.default_write_consistency.clone(), hinted_handoff_enabled: config.cluster.hinted_handoff_enabled, max_hint_window_secs: config.cluster.max_hint_window_secs, gc_grace_seconds: config.storage.gc_grace_seconds, peer_timeout_ms: 10000 }, store));
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
        // Insert a vector into the service's store within the collection namespace
        let ns = service.coordinator.store().as_ref().clone().with_namespace("default".into());
        ns.put_vector(1, &[1.0, 2.0, 3.0], 100).unwrap();
        ns.put_payload(1, b"fetch-payload").unwrap();

        let req = tonic::Request::new(FetchRequest {
            ids: vec![1],
            collection_name: "default".into(),
            include_payloads: true,
            consistency: 0,
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
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
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
            timestamp: 100,
            consistency: 0,
        });
        let resp = service.create_collection(req).await.unwrap();
        assert!(resp.into_inner().success);
        // Duplicate with same timestamp should be rejected by LWW
        let req2 = tonic::Request::new(CreateCollectionRequest {
            name: "dup".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 999,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
            timestamp: 100,
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
        });
        service.create_collection(create_req).await.unwrap();
        // Drop it
        let drop_req = tonic::Request::new(DropCollectionRequest { name: "to-drop".into(), is_replication: false, timestamp: 0, consistency: 0 });
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
            timestamp: 0,
            consistency: 0,
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
            timestamp: 0,
            consistency: 0,
        });
        service.create_collection(create_req).await.unwrap();

        let req2 = tonic::Request::new(CollectionExistsRequest { name: "existent".into() });
        let resp = service.collection_exists(req2).await.unwrap();
        assert!(resp.into_inner().exists);
    }

    #[tokio::test]
    async fn test_list_collections_excludes_tombstoned() {
        let service = make_service();

        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "to-tombstone".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
            timestamp: 0,
            consistency: 0,
        });
        service.create_collection(create_req).await.unwrap();

        // Should be in list
        let resp = service.list_collections(tonic::Request::new(ListCollectionsRequest {})).await.unwrap();
        assert_eq!(resp.into_inner().collections.len(), 1);

        // Drop it
        let drop_req = tonic::Request::new(DropCollectionRequest {
            name: "to-tombstone".into(), is_replication: false, timestamp: 0, consistency: 0,
        });
        service.drop_collection(drop_req).await.unwrap();

        // Should NOT be in list anymore (filtered by tombstone)
        let resp2 = service.list_collections(tonic::Request::new(ListCollectionsRequest {})).await.unwrap();
        assert_eq!(resp2.into_inner().collections.len(), 0);
    }

    #[tokio::test]
    async fn test_collection_exists_false_after_drop() {
        let service = make_service();
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "temp-drop".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
            timestamp: 0,
            consistency: 0,
        });
        service.create_collection(create_req).await.unwrap();
        assert!(service.collection_exists(tonic::Request::new(CollectionExistsRequest { name: "temp-drop".into() })).await.unwrap().into_inner().exists);

        service.drop_collection(tonic::Request::new(DropCollectionRequest {
            name: "temp-drop".into(), is_replication: false, timestamp: 0, consistency: 0,
        })).await.unwrap();
        assert!(!service.collection_exists(tonic::Request::new(CollectionExistsRequest { name: "temp-drop".into() })).await.unwrap().into_inner().exists);
    }

    #[tokio::test]
    async fn test_list_collections_includes_timestamp() {
        let service = make_service();
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "timestamp-test".into(),
            is_replication: false,
            config: Some(crate::proto::CollectionConfig {
                dim: 4, num_vector_shards: 1, replication_factor: 1,
                num_dim_groups: 1, dim_group_size: 4, nlist: 4, nprobe: 2,
                pq_num_sub_vectors: 2, pq_num_centroids: 8, re_rank_k: 4,
            }),
            timestamp: 98765,
            consistency: 0,
        });
        service.create_collection(create_req).await.unwrap();

        let resp = service.list_collections(tonic::Request::new(ListCollectionsRequest {})).await.unwrap();
        let infos = resp.into_inner().collections;
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].config_timestamp, 98765);
    }

}
