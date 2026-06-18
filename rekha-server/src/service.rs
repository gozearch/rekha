use rekha_core::{NodeInfo, NodeStatus, Payload, RekhaError, SearchParams, VectorStoreBackend};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::coordinator::Coordinator;
use crate::proto::{
    self, rekha_server::Rekha, AppendEntriesRequest, CollectionExistsRequest,
    CollectionExistsResponse, CreateCollectionRequest, CreateCollectionResponse, DeleteRequest,
    DeleteResponse, DropCollectionRequest, DropCollectionResponse, FetchRequest, FetchResponse,
    HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse, InsertBatchResponse,
    InsertRequest, InsertResponse, ListCollectionsRequest, ListCollectionsResponse, RaftAck,
    RaftSnapshotChunk, RaftVoteRequest, RaftVoteResponse, ScoredPoint, SearchRequest,
    SearchResponse, TransferRequest, TransferResponse,
};
use rekha_core::RaftError;
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
            RekhaError::Consensus(rekha_core::RaftError::NotLeader { .. }) => {
                Status::failed_precondition(e.to_string())
            }
            RekhaError::Consensus(_) => Status::internal(e.to_string()),
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

        let result = self.coordinator.insert(req.id, req.vector, payload).await;
        if let Err(ref e) = result {
            if let RekhaError::Consensus(RaftError::NotLeader {
                leader_hint: Some(leader_id),
            }) = e
            {
                let addr = self.coordinator.peer_address(leader_id).await;
                let detail = match addr {
                    Some(a) => format!("not leader, try {leader_id}@{a}"),
                    None => format!("not leader, try {leader_id}"),
                };
                return Err(Status::failed_precondition(detail));
            }
            return Err(Self::map_error(e.clone()));
        }
        let actual_id = result.unwrap();

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

            match self.coordinator.insert(item.id, item.vector, payload).await {
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
            beam_width: params.beam_width as usize,
            include_payloads: params.include_payloads,
            partition_hint: params.partition_hint,
            local_only: req.local_only,
        };

        match self
            .coordinator
            .search(req.query_vector, req.top_k as usize, search_params)
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
            beam_width: params.beam_width as usize,
            include_payloads: params.include_payloads,
            partition_hint: params.partition_hint,
            local_only: req.local_only,
        };

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let coordinator = self.coordinator.clone();

        tokio::spawn(async move {
            match coordinator
                .search(req.query_vector, req.top_k as usize, search_params.clone())
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
            dim_groups: (0..self.coordinator.config_ref().partition.num_dim_groups).collect(),
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
                is_leader: n.is_leader,
                raft_term: n.raft_term,
                commit_index: n.commit_index,
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
        let local = self.coordinator.local_node_info();

        // Register/update the sender.
        let peer_info = NodeInfo {
            node_id: req.node_id.clone(),
            address: req.address.clone(),
            partition_id: 0,
            dim_groups: Vec::new(),
            is_leader: false,
            raft_term: req.raft_term,
            commit_index: req.commit_index,
            storage_bytes: req.storage_bytes,
            status: NodeStatus::Healthy,
            last_heartbeat: 0,
        };
        self.coordinator.register_peer(peer_info).await;

        Ok(Response::new(HeartbeatResponse {
            success: true,
            leader_hint: if local.is_leader {
                local.node_id
            } else {
                String::new()
            },
            leader_term: local.raft_term,
        }))
    }

    async fn transfer_shard(
        &self,
        _request: Request<TransferRequest>,
    ) -> Result<Response<TransferResponse>, Status> {
        Ok(Response::new(TransferResponse {
            success: false,
            transferred_count: 0,
            error: "not implemented".into(),
        }))
    }

    // ── Raft ──────────────────────────────────────────────────
    async fn raft_append_entries(
        &self,
        request: Request<AppendEntriesRequest>,
    ) -> Result<Response<RaftAck>, Status> {
        let req = request.into_inner();

        // Find the Raft node for this partition.
        let raft_node = self
            .coordinator
            .raft_node(req.partition_id)
            .ok_or_else(|| Status::not_found("no raft node for partition"))?;

        // Convert proto entries to RaftLogEntry.
        let entries: Vec<_> = req
            .entries
            .into_iter()
            .map(|e| rekha_raft::node::RaftLogEntry {
                term: e.term,
                index: e.index,
                command: proto_raft_command_to_internal(e.command.unwrap_or_default()),
            })
            .collect();

        match raft_node
            .handle_append_entries(
                req.leader_term,
                &req.leader_id,
                req.prev_log_index,
                req.prev_log_term,
                entries,
                req.leader_commit,
            )
            .await
        {
            Ok((success, current_term)) => Ok(Response::new(RaftAck {
                success,
                current_term,
                commit_index: raft_node.commit_index().await,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(RaftAck {
                success: false,
                current_term: raft_node.current_term().await,
                commit_index: 0,
                error: e.to_string(),
            })),
        }
    }

    async fn raft_request_vote(
        &self,
        request: Request<RaftVoteRequest>,
    ) -> Result<Response<RaftVoteResponse>, Status> {
        let req = request.into_inner();

        let raft_node = self
            .coordinator
            .raft_node(req.partition_id)
            .ok_or_else(|| {
                Status::not_found(format!("no raft node for partition {}", req.partition_id))
            })?;

        match raft_node
            .handle_request_vote(
                req.term,
                &req.candidate_id,
                req.last_log_index,
                req.last_log_term,
            )
            .await
        {
            Ok((vote_granted, term)) => Ok(Response::new(RaftVoteResponse { term, vote_granted })),
            Err(e) => Err(Self::map_error(e)),
        }
    }

    async fn raft_install_snapshot(
        &self,
        _request: Request<tonic::Streaming<RaftSnapshotChunk>>,
    ) -> Result<Response<RaftAck>, Status> {
        Ok(Response::new(RaftAck {
            success: true,
            current_term: 0,
            commit_index: 0,
            error: String::new(),
        }))
    }

    // ── Collection Management ──────────────────────────────────

    async fn create_collection(
        &self,
        _request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        Err(Status::unimplemented(
            "create_collection not yet implemented",
        ))
    }

    async fn drop_collection(
        &self,
        _request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        Err(Status::unimplemented("drop_collection not yet implemented"))
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        Err(Status::unimplemented(
            "list_collections not yet implemented",
        ))
    }

    async fn collection_exists(
        &self,
        _request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        Err(Status::unimplemented(
            "collection_exists not yet implemented",
        ))
    }
}

/// Convert a proto RaftCommand to the internal RaftCommand type.
fn proto_raft_command_to_internal(
    cmd: crate::proto::RaftCommand,
) -> rekha_raft::state::RaftCommand {
    use crate::proto::raft_command::Cmd;
    match cmd.cmd {
        Some(Cmd::Insert(insert)) => rekha_raft::state::RaftCommand::Insert {
            id: insert.id,
            vector: insert.vector,
            payload: insert.payload.and_then(|p| {
                if p.data.is_empty() {
                    None
                } else {
                    Some(p.data)
                }
            }),
        },
        Some(Cmd::Delete(delete)) => rekha_raft::state::RaftCommand::Delete { ids: delete.ids },
        Some(Cmd::Custom(_)) | None => rekha_raft::state::RaftCommand::NoOp,
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
    fn test_map_error_not_leader() {
        let err = RekhaError::Consensus(rekha_core::RaftError::NotLeader {
            leader_hint: Some("n2".into()),
        });
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
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
    fn test_proto_raft_command_insert() {
        let insert = crate::proto::InsertRequest {
            id: 42,
            vector: vec![1.0, 2.0, 3.0],
            payload: Some(crate::proto::Payload {
                content_type: "raw".into(),
                data: vec![0, 1, 2],
            }),
            collection_name: String::new(),
        };
        let proto_cmd = crate::proto::RaftCommand {
            cmd: Some(crate::proto::raft_command::Cmd::Insert(insert)),
        };
        let cmd = super::proto_raft_command_to_internal(proto_cmd);
        match cmd {
            rekha_raft::state::RaftCommand::Insert {
                id,
                vector,
                payload,
            } => {
                assert_eq!(id, 42);
                assert_eq!(vector, vec![1.0, 2.0, 3.0]);
                assert_eq!(payload, Some(vec![0, 1, 2]));
            }
            _ => panic!("expected Insert variant"),
        }
    }

    #[test]
    fn test_proto_raft_command_delete() {
        let delete = crate::proto::DeleteRequest {
            ids: vec![1, 2, 3],
            collection_name: String::new(),
        };
        let proto_cmd = crate::proto::RaftCommand {
            cmd: Some(crate::proto::raft_command::Cmd::Delete(delete)),
        };
        let cmd = super::proto_raft_command_to_internal(proto_cmd);
        match cmd {
            rekha_raft::state::RaftCommand::Delete { ids } => {
                assert_eq!(ids, vec![1, 2, 3]);
            }
            _ => panic!("expected Delete variant"),
        }
    }

    #[test]
    fn test_proto_raft_command_custom() {
        let proto_cmd = crate::proto::RaftCommand {
            cmd: Some(crate::proto::raft_command::Cmd::Custom(vec![1, 2, 3])),
        };
        let cmd = super::proto_raft_command_to_internal(proto_cmd);
        assert!(matches!(cmd, rekha_raft::state::RaftCommand::NoOp));
    }

    #[test]
    fn test_proto_raft_command_default() {
        let proto_cmd = crate::proto::RaftCommand { cmd: None };
        let cmd = super::proto_raft_command_to_internal(proto_cmd);
        assert!(matches!(cmd, rekha_raft::state::RaftCommand::NoOp));
    }

    #[test]
    fn test_proto_raft_command_insert_empty_payload() {
        let insert = crate::proto::InsertRequest {
            id: 1,
            vector: vec![0.5],
            payload: Some(crate::proto::Payload {
                content_type: "raw".into(),
                data: vec![], // empty payload should become None
            }),
            collection_name: String::new(),
        };
        let proto_cmd = crate::proto::RaftCommand {
            cmd: Some(crate::proto::raft_command::Cmd::Insert(insert)),
        };
        let cmd = super::proto_raft_command_to_internal(proto_cmd);
        match cmd {
            rekha_raft::state::RaftCommand::Insert { payload, .. } => {
                assert!(payload.is_none());
            }
            _ => panic!("expected Insert variant"),
        }
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

    #[test]
    fn test_map_error_consensus_non_leader() {
        let err = RekhaError::Consensus(rekha_core::RaftError::LogCompaction {
            detail: "test".into(),
        });
        let status = RekhaService::map_error(err);
        assert_eq!(status.code(), tonic::Code::Internal);
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

    fn make_service() -> RekhaService {
        let id = NEXT_SVC_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let config = crate::config::ServerConfig::dev_default(
            "test-node",
            &format!("/tmp/svc_handler_{id}"),
        );
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(format!("/tmp/svc_handler_db_{id}")).unwrap(),
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
            raft_term: 3,
            commit_index: 10,
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
    async fn test_create_collection_stub() {
        let service = make_service();
        let req = tonic::Request::new(CreateCollectionRequest {
            name: "test".into(),
            config: Some(crate::proto::CollectionConfig {
                dim: 8,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 8,
                graph_degree: 32,
                search_list_size: 100,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 256,
                re_rank_k: 200,
            }),
        });
        let result = service.create_collection(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn test_drop_collection_stub() {
        let service = make_service();
        let req = tonic::Request::new(DropCollectionRequest {
            name: "test".into(),
        });
        let result = service.drop_collection(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn test_list_collections_stub() {
        let service = make_service();
        let req = tonic::Request::new(ListCollectionsRequest {});
        let result = service.list_collections(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn test_collection_exists_stub() {
        let service = make_service();
        let req = tonic::Request::new(CollectionExistsRequest {
            name: "test".into(),
        });
        let result = service.collection_exists(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unimplemented);
    }

    #[tokio::test]
    async fn test_transfer_shard_stub() {
        let service = make_service();
        let req = tonic::Request::new(TransferRequest {
            partition_id: 0,
            target_node: "".into(),
            vector_ids: vec![],
        });
        let resp = service.transfer_shard(req).await.unwrap();
        let inner = resp.into_inner();
        assert!(!inner.success);
    }

    #[tokio::test]
    async fn test_raft_append_entries_no_node() {
        let service = make_service();
        let req = tonic::Request::new(AppendEntriesRequest {
            collection_name: "default".into(),
            partition_id: 0,
            leader_term: 1,
            leader_id: "leader".into(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
        });
        let result = service.raft_append_entries(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_raft_request_vote_no_node() {
        let service = make_service();
        let req = tonic::Request::new(RaftVoteRequest {
            collection_name: "default".into(),
            term: 1,
            candidate_id: "candidate".into(),
            last_log_index: 0,
            last_log_term: 0,
            partition_id: 0,
        });
        let result = service.raft_request_vote(req).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::NotFound);
    }

    #[tokio::test]
    async fn test_raft_append_entries_with_node() {
        let service = make_service();
        // Register a Raft node on the coordinator
        use rekha_raft::{RaftNode, ReplicatedState};
        let state = ReplicatedState::new(0);
        let raft_log_store = service.coordinator.raft_log_store();
        let node = std::sync::Arc::new(RaftNode::with_store(
            "test-node".into(),
            0,
            vec![],
            state,
            Some(raft_log_store),
            None,
        ));
        node.start_election().await.unwrap();
        service.coordinator.register_raft_node(0, node);

        let req = tonic::Request::new(AppendEntriesRequest {
            collection_name: "default".into(),
            partition_id: 0,
            leader_term: 2,
            leader_id: "new-leader".into(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![crate::proto::RaftEntry {
                term: 2,
                index: 1,
                command: Some(crate::proto::RaftCommand {
                    cmd: Some(crate::proto::raft_command::Cmd::Insert(
                        crate::proto::InsertRequest {
                            id: 0,
                            vector: vec![1.0],
                            payload: None,
                            collection_name: "default".into(),
                        },
                    )),
                }),
            }],
            leader_commit: 1,
        });
        let resp = service.raft_append_entries(req).await.unwrap();
        let ack = resp.into_inner();
        assert!(ack.success);
    }

    #[tokio::test]
    async fn test_raft_request_vote_with_node() {
        let service = make_service();
        use rekha_raft::{RaftNode, ReplicatedState};
        let state = ReplicatedState::new(0);
        let raft_log_store = service.coordinator.raft_log_store();
        let node = std::sync::Arc::new(RaftNode::with_store(
            "test-node".into(),
            0,
            vec![],
            state,
            Some(raft_log_store),
            None,
        ));
        service.coordinator.register_raft_node(0, node);

        let req = tonic::Request::new(RaftVoteRequest {
            collection_name: "default".into(),
            term: 1,
            candidate_id: "candidate".into(),
            last_log_index: 0,
            last_log_term: 0,
            partition_id: 0,
        });
        let resp = service.raft_request_vote(req).await.unwrap();
        let vote_resp = resp.into_inner();
        // Follower should grant vote to candidate with higher term
        assert!(vote_resp.vote_granted);
    }
}
