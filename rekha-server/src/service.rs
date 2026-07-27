use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use rekha_coordinator::Coordinator;
use rekha_core::{ConsistencyLevel, IvfConfig, RekhaError, SearchParams};
use rekha_proto::proto::{self, rekha_server::Rekha};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::metrics;

pub struct RekhaService {
    pub coordinator: Arc<Coordinator>,
}

impl RekhaService {
    pub fn new(coordinator: Arc<Coordinator>) -> Self {
        RekhaService { coordinator }
    }
}

fn map_err(e: RekhaError) -> Status {
    match e {
        RekhaError::NotFound(_) => Status::not_found(e.to_string()),
        RekhaError::InvalidArgument(_) => Status::invalid_argument(e.to_string()),
        RekhaError::Timeout(_) => Status::deadline_exceeded(e.to_string()),
        RekhaError::Unavailable(_) => Status::unavailable(e.to_string()),
        _ => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl Rekha for RekhaService {
    async fn create_collection(
        &self,
        request: Request<proto::CreateCollectionRequest>,
    ) -> Result<Response<proto::CreateCollectionResponse>, Status> {
        let req = request.into_inner();
        let config: IvfConfig = req.config.map(IvfConfig::from).unwrap_or_default();
        let consistency = ConsistencyLevel::from(req.consistency());
        match self.coordinator.create_collection(&req.name, config, &req.origin_node_id, req.timestamp as i64, consistency, req.is_replication).await {
            Ok(_) => Ok(Response::new(proto::CreateCollectionResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(proto::CreateCollectionResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn drop_collection(
        &self,
        request: Request<proto::DropCollectionRequest>,
    ) -> Result<Response<proto::DropCollectionResponse>, Status> {
        let req = request.into_inner();
        let consistency = ConsistencyLevel::from(req.consistency());
        match self.coordinator.drop_collection(&req.name, &req.origin_node_id, req.timestamp as i64, consistency, req.is_replication).await {
            Ok(_) => Ok(Response::new(proto::DropCollectionResponse {
                success: true,
                error: String::new(),
            })),
            Err(e) => Ok(Response::new(proto::DropCollectionResponse {
                success: false,
                error: e.to_string(),
            })),
        }
    }

    async fn list_collections(
        &self,
        _request: Request<proto::ListCollectionsRequest>,
    ) -> Result<Response<proto::ListCollectionsResponse>, Status> {
        let names = self.coordinator.list_collections().await.map_err(map_err)?;
        let mut collections = Vec::new();
        for name in &names {
            let config = self.coordinator.store.load_collection_config(name).ok();
            let info = proto::CollectionInfo {
                name: name.clone(),
                config: config.map(proto::CollectionConfig::from),
                vector_count: 0,
                index_ready: true,
                config_timestamp: 0,
            };
            collections.push(info);
        }
        Ok(Response::new(proto::ListCollectionsResponse {
            collections,
        }))
    }

    async fn collection_exists(
        &self,
        request: Request<proto::CollectionExistsRequest>,
    ) -> Result<Response<proto::CollectionExistsResponse>, Status> {
        let req = request.into_inner();
        let exists = self
            .coordinator
            .collection_exists(&req.name)
            .await
            .map_err(map_err)?;
        Ok(Response::new(proto::CollectionExistsResponse { exists }))
    }

    async fn import(
        &self,
        request: Request<tonic::Streaming<proto::ImportChunk>>,
    ) -> Result<Response<proto::ImportResponse>, Status> {
        let mut stream = request.into_inner();
        let mut inserted = 0u64;
        let mut errors = Vec::new();

        while let Some(chunk) = stream.message().await.unwrap_or(None) {
            for req in chunk.requests {
                let consistency: ConsistencyLevel = req.consistency().into();
                let payload = req.payload.map(|p| p.data);
                match self.coordinator.insert(
                    &req.collection_name,
                    req.id,
                    req.vector,
                    payload,
                    req.timestamp as i64,
                    &req.origin_node_id,
                    consistency,
                    req.is_replication,
                ).await {
                    Ok(_) => inserted += 1,
                    Err(e) => errors.push(e.to_string()),
                }
            }
        }

        Ok(Response::new(proto::ImportResponse {
            inserted_count: inserted,
            errors,
        }))
    }

    type ExportStream = Pin<Box<dyn Stream<Item = Result<proto::ExportChunk, Status>> + Send>>;

    async fn export(
        &self,
        request: Request<proto::ExportRequest>,
    ) -> Result<Response<Self::ExportStream>, Status> {
        let req = request.into_inner();

        let mut rx = self.coordinator.export_stream(
            &req.collection_name, req.offset, req.limit,
            req.include_vectors, req.include_payloads, 500,
        );

        let (tx, out_rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(batch_result) = rx.recv().await {
                match batch_result {
                    Ok(vectors) => {
                        let chunk = proto::ExportChunk {
                            vectors: vectors.into_iter().map(|v| proto::ExportedVector {
                                id: v.id,
                                vector: v.vector,
                                payload: v.payload.map(|data| proto::Payload {
                                    content_type: "application/octet-stream".into(),
                                    data,
                                }),
                                timestamp: v.timestamp,
                            }).collect(),
                        };
                        if tx.send(Ok(chunk)).await.is_err() { break; }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(map_err(e))).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(out_rx))
                as Self::ExportStream
        ))
    }

    async fn insert(
        &self,
        request: Request<proto::InsertRequest>,
    ) -> Result<Response<proto::InsertResponse>, Status> {
        let _timer = metrics::TimerGuard::new("insert");
        metrics::record_insert();
        let req = request.into_inner();
        let consistency = ConsistencyLevel::from(req.consistency());
        let payload = req.payload.map(|p| p.data);
        match self.coordinator.insert(
            &req.collection_name, req.id, req.vector, payload,
            req.timestamp as i64, &req.origin_node_id, consistency, req.is_replication,
        ).await {
            Ok(_) => Ok(Response::new(proto::InsertResponse { id: req.id, success: true, error: String::new() })),
            Err(e) => Ok(Response::new(proto::InsertResponse { id: req.id, success: false, error: e.to_string() })),
        }
    }

    async fn insert_batch(
        &self,
        request: Request<tonic::Streaming<proto::InsertRequest>>,
    ) -> Result<Response<proto::InsertBatchResponse>, Status> {
        let mut stream = request.into_inner();
        let mut count = 0u64;
        let mut errors = Vec::new();
        while let Some(req) = stream.message().await.unwrap_or(None) {
            let consistency = ConsistencyLevel::from(req.consistency());
            let payload = req.payload.map(|p| p.data);
            match self.coordinator.insert(
                &req.collection_name, req.id, req.vector, payload,
                req.timestamp as i64, &req.origin_node_id, consistency, req.is_replication,
            ).await {
                Ok(_) => count += 1,
                Err(e) => errors.push(e.to_string()),
            }
        }
        Ok(Response::new(proto::InsertBatchResponse {
            inserted_count: count,
            errors,
        }))
    }

    async fn delete(
        &self,
        request: Request<proto::DeleteRequest>,
    ) -> Result<Response<proto::DeleteResponse>, Status> {
        let _timer = metrics::TimerGuard::new("delete");
        metrics::record_delete();
        let req = request.into_inner();
        let consistency = ConsistencyLevel::from(req.consistency());
        match self.coordinator.delete(
            &req.collection_name, &req.ids, req.timestamp as i64, &req.origin_node_id, consistency, req.is_replication,
        ).await {
            Ok(deleted) => Ok(Response::new(proto::DeleteResponse {
                deleted_count: deleted,
                error: String::new(),
                timestamp: req.timestamp,
            })),
            Err(e) => Ok(Response::new(proto::DeleteResponse {
                deleted_count: 0,
                error: e.to_string(),
                timestamp: req.timestamp,
            })),
        }
    }

    async fn fetch(
        &self,
        request: Request<proto::FetchRequest>,
    ) -> Result<Response<proto::FetchResponse>, Status> {
        let req = request.into_inner();
        let points = self
            .coordinator
            .fetch(&req.collection_name, &req.ids, req.include_payloads)
            .await
            .map_err(map_err)?;
        let proto_points: Vec<proto::ScoredPoint> = points
            .into_iter()
            .map(|sp| proto::ScoredPoint {
                id: sp.id,
                score: sp.score,
                payload: sp.payload.map(|data| proto::Payload {
                    content_type: "application/octet-stream".into(),
                    data,
                }),
                timestamp: sp.timestamp as u64,
            })
            .collect();
        Ok(Response::new(proto::FetchResponse {
            vectors: vec![],
            points: proto_points,
            error: String::new(),
        }))
    }

    async fn search(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<proto::SearchResponse>, Status> {
        let _timer = metrics::TimerGuard::new("search");
        metrics::record_search();
        let req = request.into_inner();
        let params = req.params.unwrap_or_default();
        let search_params = SearchParams {
            nprobe: params.nprobe,
            k: req.top_k,
            include_payloads: params.include_payloads,
            pre_filter: None,
            local_only: req.local_only,
        };
        match self
            .coordinator
            .search(
                &req.collection_name,
                req.query_vector,
                req.top_k,
                search_params,
            )
            .await
        {
            Ok(results) => {
                let proto_results: Vec<proto::ScoredPoint> = results
                    .into_iter()
                    .map(|sp| proto::ScoredPoint {
                        id: sp.id,
                        score: sp.score,
                        payload: sp.payload.map(|data| proto::Payload {
                            content_type: "application/octet-stream".into(),
                            data,
                        }),
                        timestamp: sp.timestamp as u64,
                    })
                    .collect();
                Ok(Response::new(proto::SearchResponse {
                    results: proto_results,
                    stats: Some(proto::SearchStats::default()),
                }))
            }
            Err(e) => Err(map_err(e)),
        }
    }

    type SearchStreamStream =
        Pin<Box<dyn Stream<Item = Result<proto::ScoredPoint, Status>> + Send>>;

    async fn search_stream(
        &self,
        request: Request<proto::SearchRequest>,
    ) -> Result<Response<Self::SearchStreamStream>, Status> {
        let req = request.into_inner();
        let params = req.params.unwrap_or_default();
        let search_params = SearchParams {
            nprobe: params.nprobe,
            k: req.top_k,
            include_payloads: params.include_payloads,
            pre_filter: None,
            local_only: req.local_only,
        };
        let results = self
            .coordinator
            .search(
                &req.collection_name,
                req.query_vector,
                req.top_k,
                search_params,
            )
            .await
            .map_err(map_err)?;

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            for sp in results {
                let proto_sp = proto::ScoredPoint {
                    id: sp.id,
                    score: sp.score,
                    payload: sp.payload.map(|data| proto::Payload {
                        content_type: "application/octet-stream".into(),
                        data,
                    }),
                    timestamp: sp.timestamp as u64,
                };
                if tx.send(Ok(proto_sp)).await.is_err() {
                    break;
                }
            }
        });
        Ok(Response::new(
            Box::pin(ReceiverStream::new(rx)) as Self::SearchStreamStream
        ))
    }

    async fn search_dim_range(
        &self,
        _request: Request<proto::SearchDimRangeRequest>,
    ) -> Result<Response<proto::SearchDimRangeResponse>, Status> {
        Err(Status::unimplemented("dim range search not implemented"))
    }

    async fn handshake(
        &self,
        request: Request<proto::HandshakeRequest>,
    ) -> Result<Response<proto::HandshakeResponse>, Status> {
        let req = request.into_inner();
        let membership = self.coordinator.membership.read().await;
        membership
            .handle_heartbeat(&req.node_id, &req.address)
            .await;
        let peers: Vec<proto::NodeInfo> = membership
            .all_peers()
            .iter()
            .map(|n| proto::NodeInfo {
                node_id: n.node_id.clone(),
                address: n.address.clone(),
                partition_id: 0,
                dim_groups: vec![],
                storage_bytes: 0,
                status: if n.is_alive {
                    "alive".to_string()
                } else {
                    "dead".to_string()
                },
            })
            .collect();
        drop(membership);

        Ok(Response::new(proto::HandshakeResponse {
            cluster_id: "rekha-cluster".to_string(),
            peers,
            error: String::new(),
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<proto::HeartbeatRequest>,
    ) -> Result<Response<proto::HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let membership = self.coordinator.membership.read().await;
        membership
            .handle_heartbeat(&req.node_id, &req.address)
            .await;
        Ok(Response::new(proto::HeartbeatResponse { success: true }))
    }

    type TransferShardStream =
        Pin<Box<dyn Stream<Item = Result<proto::TransferShardChunk, Status>> + Send>>;

    async fn transfer_shard(
        &self,
        request: Request<proto::TransferShardRequest>,
    ) -> Result<Response<Self::TransferShardStream>, Status> {
        let req = request.into_inner();
        let mut rx = self.coordinator.transfer_shard_stream(&req.collection_name, 500);

        let (tx, out_rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(chunk_result) = rx.recv().await {
                match chunk_result {
                    Ok(chunk) => {
                        if tx.send(Ok(chunk)).await.is_err() { break; }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(map_err(e))).await;
                        break;
                    }
                }
            }
        });

        Ok(Response::new(
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(out_rx))
                as Self::TransferShardStream
        ))
    }

    type RepairCollectionStream =
        Pin<Box<dyn Stream<Item = Result<proto::RepairProgress, Status>> + Send>>;

    async fn repair_collection(
        &self,
        request: Request<proto::RepairCollectionRequest>,
    ) -> Result<Response<Self::RepairCollectionStream>, Status> {
        let req = request.into_inner();
        let progress = self.coordinator.repair_collection(&req.collection_name)
            .await
            .map_err(map_err)?;

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            let _ = tx.send(Ok(progress)).await;
        });

        Ok(Response::new(
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
                as Self::RepairCollectionStream
        ))
    }

    async fn find_successor(
        &self,
        request: Request<proto::FindSuccessorRequest>,
    ) -> Result<Response<proto::FindSuccessorResponse>, Status> {
        let req = request.into_inner();
        let id_bytes: [u8; 16] = req.id.as_slice().try_into()
            .map_err(|_| Status::invalid_argument("id must be 16 bytes"))?;
        let id = u128::from_le_bytes(id_bytes);
        let result = self.coordinator.chord.handle_find_successor(id);
        match result {
            Some((node_id, address)) => Ok(Response::new(proto::FindSuccessorResponse {
                successor: Some(proto::NodeInfo {
                    node_id,
                    address,
                    partition_id: 0,
                    dim_groups: vec![],
                    storage_bytes: 0,
                    status: "alive".to_string(),
                }),
            })),
            None => Ok(Response::new(proto::FindSuccessorResponse {
                successor: Some(proto::NodeInfo {
                    node_id: self.coordinator.node_id_str.clone(),
                    address: String::new(),
                    partition_id: 0,
                    dim_groups: vec![],
                    storage_bytes: 0,
                    status: "alive".to_string(),
                }),
            })),
        }
    }

    async fn get_predecessor(
        &self,
        _request: Request<proto::GetPredecessorRequest>,
    ) -> Result<Response<proto::GetPredecessorResponse>, Status> {
        let pred = self.coordinator.chord.predecessor.read().await;
        let pred_addr = self.coordinator.chord.predecessor_address.read().await;
        let pred_info = match (pred.as_ref(), pred_addr.as_ref()) {
            (Some(id), Some(addr)) => Some(proto::NodeInfo {
                node_id: id.clone(),
                address: addr.clone(),
                partition_id: 0,
                dim_groups: vec![],
                storage_bytes: 0,
                status: "alive".to_string(),
            }),
            _ => None,
        };
        Ok(Response::new(proto::GetPredecessorResponse {
            predecessor: pred_info,
        }))
    }

    async fn notify_chord(
        &self,
        request: Request<proto::NotifyChordRequest>,
    ) -> Result<Response<proto::NotifyChordResponse>, Status> {
        let req = request.into_inner();
        let node = req.node.ok_or_else(|| Status::invalid_argument("node required"))?;
        let accepted = self.coordinator.chord.notify(&node.node_id, &node.address);
        Ok(Response::new(proto::NotifyChordResponse { success: accepted }))
    }

    async fn get_successor_list(
        &self,
        _request: Request<proto::GetSuccessorListRequest>,
    ) -> Result<Response<proto::GetSuccessorListResponse>, Status> {
        let successors = self.coordinator.chord.successor_list.read().await;
        let infos: Vec<proto::NodeInfo> = successors.iter().map(|id| proto::NodeInfo {
            node_id: id.clone(),
            address: String::new(),
            partition_id: 0,
            dim_groups: vec![],
            storage_bytes: 0,
            status: "alive".to_string(),
        }).collect();
        Ok(Response::new(proto::GetSuccessorListResponse { successors: infos }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_error_not_found() {
        let e = RekhaError::NotFound("test".into());
        let status = map_err(e);
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn test_map_error_invalid_argument() {
        let e = RekhaError::InvalidArgument("bad".into());
        let status = map_err(e);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_map_error_internal() {
        let e = RekhaError::Internal("oops".into());
        let status = map_err(e);
        assert_eq!(status.code(), tonic::Code::Internal);
    }
}
