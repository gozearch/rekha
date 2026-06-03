use rekha_core::{
    Coordinator as CoordinatorTrait, Payload, RekhaError, SearchParams, VectorStoreBackend,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::coordinator::Coordinator;
use crate::proto::{
    self, rekha_server::Rekha, DeleteRequest, DeleteResponse, FetchRequest, FetchResponse,
    HandshakeRequest, HandshakeResponse, HeartbeatRequest, HeartbeatResponse, InsertBatchResponse,
    InsertRequest, InsertResponse, RaftAck, RaftEntry, RaftSnapshotChunk, RaftVoteRequest,
    RaftVoteResponse, ScoredPoint, SearchRequest, SearchResponse, TransferRequest,
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

        Ok(Response::new(HandshakeResponse {
            cluster_id: "rekha-dev".into(),
            peers: vec![],
            error: String::new(),
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let _req = request.into_inner();
        Ok(Response::new(HeartbeatResponse {
            success: true,
            leader_hint: String::new(),
            leader_term: 0,
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
        _request: Request<tonic::Streaming<RaftEntry>>,
    ) -> Result<Response<RaftAck>, Status> {
        Ok(Response::new(RaftAck {
            success: true,
            current_term: 0,
            commit_index: 0,
            error: String::new(),
        }))
    }

    async fn raft_request_vote(
        &self,
        _request: Request<RaftVoteRequest>,
    ) -> Result<Response<RaftVoteResponse>, Status> {
        Ok(Response::new(RaftVoteResponse {
            term: 0,
            vote_granted: false,
        }))
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
}
