use rekha_core::{
    CollectionConfig, DistanceMetric, NodeInfo, NodeStatus, Payload, RekhaError, SearchParams,
    VectorStoreBackend,
};
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{info, warn};

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

pub struct RekhaService {
    coordinator: Arc<Coordinator>,
}

impl RekhaService {
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        Self { coordinator }
    }

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

    fn collection_name(req: &str) -> &str {
        if req.is_empty() {
            "default"
        } else {
            req
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
        let cname = Self::collection_name(&req.collection_name);
        let payload = req.payload.map(|p| Payload {
            content_type: match p.content_type.as_str() {
                "json" => rekha_core::PayloadType::Json,
                "text" => rekha_core::PayloadType::Text,
                _ => rekha_core::PayloadType::Raw,
            },
            data: p.data,
        });

        let result = self
            .coordinator
            .insert_into_collection(cname, req.id, req.vector, payload)
            .await;
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
            let cname = Self::collection_name(&item.collection_name);
            let payload = item.payload.map(|p| Payload {
                content_type: match p.content_type.as_str() {
                    "json" => rekha_core::PayloadType::Json,
                    "text" => rekha_core::PayloadType::Text,
                    _ => rekha_core::PayloadType::Raw,
                },
                data: p.data,
            });

            match self
                .coordinator
                .insert_into_collection(cname, item.id, item.vector, payload)
                .await
            {
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
        let cname = Self::collection_name(&req.collection_name);
        let deleted = self
            .coordinator
            .delete_from_collection(cname, &req.ids)
            .await
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
        let cname = Self::collection_name(&req.collection_name);

        let results = self
            .coordinator
            .fetch_from_collection(cname, &req.ids)
            .await
            .map_err(Self::map_error)?;

        let mut vectors = Vec::new();
        let mut points = Vec::new();
        for (id, vec_opt, payload_opt) in results {
            if let Some(data) = vec_opt {
                vectors.push(proto::Vector { id, data });
                if req.include_payloads {
                    points.push(ScoredPoint {
                        id,
                        score: 0.0,
                        payload: payload_opt.map(|data| proto::Payload {
                            content_type: "raw".into(),
                            data,
                        }),
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
        let cname = Self::collection_name(&req.collection_name);
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
            .search_for_collection(cname, req.query_vector, req.top_k as usize, search_params)
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
        let cname = Self::collection_name(&req.collection_name).to_string();
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
        let query = req.query_vector;
        let top_k = req.top_k as usize;

        tokio::spawn(async move {
            match coordinator
                .search_for_collection(&cname, query, top_k, search_params.clone())
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
        let cname = Self::collection_name(&req.collection_name);

        let raft_node = self
            .coordinator
            .raft_node(cname, req.partition_id)
            .ok_or_else(|| {
                warn!(
                    "raft_append_entries: no node for cname='{}' pid={}",
                    cname, req.partition_id
                );
                Status::not_found(format!(
                    "no raft node for partition {} collection '{}'",
                    req.partition_id, cname
                ))
            })?;

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
        let cname = Self::collection_name(&req.collection_name);

        let raft_node = self
            .coordinator
            .raft_node(cname, req.partition_id)
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
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let config = req
            .config
            .ok_or_else(|| Status::invalid_argument("collection config required"))?;

        let collection_config = CollectionConfig {
            dim: config.dim,
            num_vector_shards: config.num_vector_shards,
            replication_factor: config.replication_factor,
            num_dim_groups: config.num_dim_groups,
            dim_group_size: config.dim_group_size,
            graph_degree: config.graph_degree,
            search_list_size: config.search_list_size,
            pq_num_sub_vectors: config.pq_num_sub_vectors,
            pq_num_centroids: config.pq_num_centroids,
            re_rank_k: config.re_rank_k,
            distance_metric: DistanceMetric::L2,
        };

        // Create locally first (so single-node works immediately).
        if let Err(e) = self
            .coordinator
            .create_collection(&req.name, collection_config.clone())
            .await
        {
            return Ok(Response::new(CreateCollectionResponse {
                success: false,
                error: e.to_string(),
            }));
        }

        // Replicate via system Raft group.
        if let Some(sys_node) = self.coordinator.system_raft_node() {
            let cmd = rekha_raft::state::RaftCommand::CreateCollection {
                name: req.name.clone(),
                config: collection_config,
            };
            match sys_node.propose(cmd).await {
                Ok(()) => {}
                Err(RekhaError::Consensus(RaftError::NotLeader { leader_hint })) => {
                    // Not the system leader — forward to leader via create_collection RPC.
                    let leader_id = leader_hint.unwrap_or_default();
                    if let Some(addr) = self.coordinator.peer_address(&leader_id).await {
                        let uri = format!("http://{addr}");
                        let endpoint = tonic::transport::Endpoint::from_shared(uri);
                        if let Ok(endpoint) = endpoint {
                            let endpoint = endpoint.connect_timeout(std::time::Duration::from_secs(5));
                            if let Ok(ch) = endpoint.connect().await {
                                let mut leader_client =
                                    crate::proto::rekha_client::RekhaClient::new(ch);
                                let _ = leader_client.create_collection(tonic::Request::new(
                                    crate::proto::CreateCollectionRequest {
                                        name: req.name.clone(),
                                        config: Some(crate::proto::CollectionConfig {
                                            dim: config.dim,
                                            num_vector_shards: config.num_vector_shards,
                                            replication_factor: config.replication_factor,
                                            num_dim_groups: config.num_dim_groups,
                                            dim_group_size: config.dim_group_size,
                                            graph_degree: config.graph_degree,
                                            search_list_size: config.search_list_size,
                                            pq_num_sub_vectors: config.pq_num_sub_vectors,
                                            pq_num_centroids: config.pq_num_centroids,
                                            re_rank_k: config.re_rank_k,
                                        }),
                                    },
                                )).await;
                            }
                        }
                    }
                }
                Err(e) => info!("system raft propose: {e}"),
            }
        }

        Ok(Response::new(CreateCollectionResponse {
            success: true,
            error: String::new(),
        }))
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        let req = request.into_inner();

        // Propose through system Raft group first.
        if let Some(sys_node) = self.coordinator.system_raft_node() {
            let cmd = rekha_raft::state::RaftCommand::DropCollection {
                name: req.name.clone(),
            };
            let _ = sys_node.propose(cmd).await;
        }

        // Drop locally.
        match self.coordinator.drop_collection(&req.name).await {
            Ok(()) => Ok(Response::new(DropCollectionResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(DropCollectionResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn list_collections(
        &self,
        _request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        let collections = self.coordinator.list_collections().await;
        let proto_collections = collections
            .into_iter()
            .map(|m| crate::proto::CollectionInfo {
                name: m.name.clone(),
                config: Some(crate::proto::CollectionConfig {
                    dim: m.config.dim,
                    num_vector_shards: m.config.num_vector_shards,
                    replication_factor: m.config.replication_factor,
                    num_dim_groups: m.config.num_dim_groups,
                    dim_group_size: m.config.dim_group_size,
                    graph_degree: m.config.graph_degree,
                    search_list_size: m.config.search_list_size,
                    pq_num_sub_vectors: m.config.pq_num_sub_vectors,
                    pq_num_centroids: m.config.pq_num_centroids,
                    re_rank_k: m.config.re_rank_k,
                }),
                vector_count: m.vector_count,
                index_ready: m.index_ready,
            })
            .collect();

        Ok(Response::new(ListCollectionsResponse {
            collections: proto_collections,
        }))
    }

    async fn collection_exists(
        &self,
        request: Request<CollectionExistsRequest>,
    ) -> Result<Response<CollectionExistsResponse>, Status> {
        let req = request.into_inner();
        let exists = self.coordinator.collection_exists(&req.name).await;
        Ok(Response::new(CollectionExistsResponse { exists }))
    }
}

/// Convert internal RaftCommand → proto RaftCommand for AppendEntries.
pub fn raft_command_to_proto(cmd: &rekha_raft::state::RaftCommand) -> crate::proto::raft_command::Cmd {
    use crate::proto::raft_command::Cmd;
    match cmd {
        rekha_raft::state::RaftCommand::Insert { id, vector, payload } => {
            Cmd::Insert(crate::proto::InsertRequest {
                id: *id,
                vector: vector.clone(),
                payload: payload.clone().map(|data| crate::proto::Payload {
                    content_type: "raw".into(), data,
                }),
                collection_name: String::new(),
            })
        }
        rekha_raft::state::RaftCommand::Delete { ids } => {
            Cmd::Delete(crate::proto::DeleteRequest {
                ids: ids.clone(), collection_name: String::new(),
            })
        }
        rekha_raft::state::RaftCommand::CreateCollection { name, config } => {
            let pb_config = crate::proto::CollectionConfig {
                dim: config.dim,
                num_vector_shards: config.num_vector_shards,
                replication_factor: config.replication_factor,
                num_dim_groups: config.num_dim_groups,
                dim_group_size: config.dim_group_size,
                graph_degree: config.graph_degree,
                search_list_size: config.search_list_size,
                pq_num_sub_vectors: config.pq_num_sub_vectors,
                pq_num_centroids: config.pq_num_centroids,
                re_rank_k: config.re_rank_k,
            };
            Cmd::CreateCollection(crate::proto::CreateCollectionCommand {
                name: name.clone(),
                config: Some(pb_config),
            })
        }
        rekha_raft::state::RaftCommand::DropCollection { name } => {
            Cmd::DropCollection(crate::proto::DropCollectionCommand {
                name: name.clone(),
            })
        }
        rekha_raft::state::RaftCommand::NoOp => Cmd::Custom(vec![]),
    }
}

/// Convert proto RaftCommand → internal RaftCommand (from AppendEntries).
fn proto_raft_command_to_internal(
    cmd: crate::proto::RaftCommand,
) -> rekha_raft::state::RaftCommand {
    use crate::proto::raft_command::Cmd;
    match cmd.cmd {
        Some(Cmd::Insert(insert)) => rekha_raft::state::RaftCommand::Insert {
            id: insert.id,
            vector: insert.vector,
            payload: insert.payload.and_then(|p| {
                if p.data.is_empty() { None } else { Some(p.data) }
            }),
        },
        Some(Cmd::Delete(delete)) => rekha_raft::state::RaftCommand::Delete { ids: delete.ids },
        Some(Cmd::CreateCollection(cc)) => {
            let cfg = cc.config.unwrap_or_default();
            rekha_raft::state::RaftCommand::CreateCollection {
                name: cc.name,
                config: rekha_core::CollectionConfig {
                    dim: cfg.dim,
                    num_vector_shards: cfg.num_vector_shards,
                    replication_factor: cfg.replication_factor,
                    num_dim_groups: cfg.num_dim_groups,
                    dim_group_size: cfg.dim_group_size,
                    graph_degree: cfg.graph_degree,
                    search_list_size: cfg.search_list_size,
                    pq_num_sub_vectors: cfg.pq_num_sub_vectors,
                    pq_num_centroids: cfg.pq_num_centroids,
                    re_rank_k: cfg.re_rank_k,
                    distance_metric: rekha_core::DistanceMetric::L2,
                },
            }
        }
        Some(Cmd::DropCollection(dc)) => rekha_raft::state::RaftCommand::DropCollection {
            name: dc.name,
        },
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
                ..
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

    #[tokio::test]
    async fn test_create_collection_handler() {
        let config =
            crate::config::ServerConfig::dev_default("test-node", "/tmp/rekha_svc_create_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/rekha_svc_create_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(CreateCollectionRequest {
            name: "test_col".into(),
            config: Some(crate::proto::CollectionConfig {
                dim: 64,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 64,
                graph_degree: 32,
                search_list_size: 100,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 16,
                re_rank_k: 200,
            }),
        });
        let resp = service.create_collection(req).await.unwrap();
        assert!(resp.into_inner().success);
    }

    static LIST_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[tokio::test]
    async fn test_list_collections_handler() {
        let n = LIST_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = format!("/tmp/rekha_svc_list_{}", n);
        let _ = std::fs::remove_dir_all(&dir);
        let config = crate::config::ServerConfig::dev_default("test-node", &format!("{dir}/cfg"));
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open(format!("{dir}/db")).unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(ListCollectionsRequest {});
        let resp = service.list_collections(req).await.unwrap();
        assert!(resp.into_inner().collections.is_empty());

        // Create a collection and verify it's listed
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "list_me".into(),
            config: Some(crate::proto::CollectionConfig {
                dim: 64,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 64,
                graph_degree: 32,
                search_list_size: 100,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 16,
                re_rank_k: 200,
            }),
        });
        service.create_collection(create_req).await.unwrap();

        let req2 = tonic::Request::new(ListCollectionsRequest {});
        let resp2 = service.list_collections(req2).await.unwrap();
        assert_eq!(resp2.into_inner().collections.len(), 1);
    }

    #[tokio::test]
    async fn test_collection_exists_handler() {
        let config =
            crate::config::ServerConfig::dev_default("test-node", "/tmp/rekha_svc_exists_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/rekha_svc_exists_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        // Non-existent collection
        let req = tonic::Request::new(CollectionExistsRequest {
            name: "nope".into(),
        });
        let resp = service.collection_exists(req).await.unwrap();
        assert!(!resp.into_inner().exists);

        // Create one
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "exists_col".into(),
            config: Some(crate::proto::CollectionConfig {
                dim: 64,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 64,
                graph_degree: 32,
                search_list_size: 100,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 16,
                re_rank_k: 200,
            }),
        });
        service.create_collection(create_req).await.unwrap();

        let req2 = tonic::Request::new(CollectionExistsRequest {
            name: "exists_col".into(),
        });
        let resp2 = service.collection_exists(req2).await.unwrap();
        assert!(resp2.into_inner().exists);
    }

    #[tokio::test]
    async fn test_collection_name_helper() {
        assert_eq!(RekhaService::collection_name(""), "default");
        assert_eq!(RekhaService::collection_name("my_col"), "my_col");
    }

    #[tokio::test]
    async fn test_insert_with_collection() {
        let config =
            crate::config::ServerConfig::dev_default("test-node", "/tmp/rekha_svc_ins_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/rekha_svc_ins_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        coord.create_default_collection().await.unwrap();
        let service = RekhaService::new(coord);

        let req = tonic::Request::new(InsertRequest {
            id: 42,
            vector: vec![0.1, 0.2, 0.3],
            payload: None,
            collection_name: "default".into(),
        });
        let resp = service.insert(req).await.unwrap();
        assert_eq!(resp.into_inner().id, 42);
    }

    #[tokio::test]
    async fn test_drop_collection_handler() {
        let config =
            crate::config::ServerConfig::dev_default("test-node", "/tmp/rekha_svc_drop_test");
        let store = std::sync::Arc::new(
            rekha_storage::RocksVectorStore::open("/tmp/rekha_svc_drop_db").unwrap(),
        );
        let pm = std::sync::Arc::new(tokio::sync::RwLock::new(
            rekha_partition::PartitionManager::new(std::collections::HashMap::new(), 4, 768),
        ));
        let coord = std::sync::Arc::new(crate::coordinator::Coordinator::new(config, store, pm));
        let service = RekhaService::new(coord);

        // Create then drop
        let create_req = tonic::Request::new(CreateCollectionRequest {
            name: "to_drop".into(),
            config: Some(crate::proto::CollectionConfig {
                dim: 64,
                num_vector_shards: 1,
                replication_factor: 1,
                num_dim_groups: 1,
                dim_group_size: 64,
                graph_degree: 32,
                search_list_size: 100,
                pq_num_sub_vectors: 4,
                pq_num_centroids: 16,
                re_rank_k: 200,
            }),
        });
        service.create_collection(create_req).await.unwrap();

        let drop_req = tonic::Request::new(DropCollectionRequest {
            name: "to_drop".into(),
        });
        let resp = service.drop_collection(drop_req).await.unwrap();
        assert!(resp.into_inner().success);

        // Verify gone
        let exists_req = tonic::Request::new(CollectionExistsRequest {
            name: "to_drop".into(),
        });
        let exists_resp = service.collection_exists(exists_req).await.unwrap();
        assert!(!exists_resp.into_inner().exists);
    }
}
