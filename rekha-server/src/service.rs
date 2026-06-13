use rekha_core::{
    Coordinator as CoordinatorTrait, NodeInfo, NodeStatus, Payload, RekhaError, SearchParams,
    VectorStoreBackend,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::coordinator::Coordinator;
use crate::proto::{
    self, rekha_server::Rekha, AppendEntriesRequest, DeleteRequest, DeleteResponse, FetchRequest,
    FetchResponse, HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse,
    InsertBatchResponse, InsertRequest, InsertResponse, RaftAck, RaftSnapshotChunk,
    RaftVoteRequest, RaftVoteResponse, ScoredPoint, SearchRequest, SearchResponse, TransferRequest,
    TransferResponse,
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

        self.coordinator
            .insert(req.id, req.vector, payload)
            .await
            .map_err(Self::map_error)?;

        Ok(Response::new(InsertResponse {
            id: req.id,
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
                Ok(()) => count += 1,
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
        };

        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let coordinator = self.coordinator.clone();

        tokio::spawn(async move {
            match coordinator
                .search(req.query_vector, req.top_k as usize, search_params)
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
            address: String::new(), // heartbeat doesn't carry address; use stored or skip
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

        // The vote request doesn't carry partition_id in the proto message.
        // For now, use partition 0 (single-partition assumption).
        // In a multi-partition setup, the partition_id would need to be in the message.
        let raft_node = self
            .coordinator
            .raft_node(0)
            .ok_or_else(|| Status::not_found("no raft node for partition 0"))?;

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
        let delete = crate::proto::DeleteRequest { ids: vec![1, 2, 3] };
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
}
