use rekha_core::{ConsistencyLevel, IvfConfig, RekhaError, ScoredPoint, SearchParams};
use rekha_proto::proto::{self, rekha_client::RekhaClient as ProtoRekhaClient};
use tonic::transport::Channel;

pub struct Client {
    inner: ProtoRekhaClient<Channel>,
}

impl Client {
    pub async fn connect(address: &str) -> Result<Self, RekhaError> {
        let inner = ProtoRekhaClient::connect(address.to_string())
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?;
        Ok(Client { inner })
    }

    pub async fn create_collection(
        &mut self,
        name: &str,
        config: IvfConfig,
    ) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(proto::CreateCollectionRequest {
            name: name.to_string(),
            config: Some(config.into()),
            is_replication: false,
            timestamp: 0,
            consistency: proto::ConsistencyLevel::Quorum as i32,
            origin_node_id: "client".to_string(),
        });
        let resp = self
            .inner
            .create_collection(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        if !resp.success {
            return Err(RekhaError::InvalidArgument(resp.error));
        }
        Ok(true)
    }

    pub async fn drop_collection(&mut self, name: &str) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(proto::DropCollectionRequest {
            name: name.to_string(),
            is_replication: false,
            timestamp: 0,
            consistency: proto::ConsistencyLevel::Quorum as i32,
            origin_node_id: "client".to_string(),
        });
        let resp = self
            .inner
            .drop_collection(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        if !resp.success {
            return Err(RekhaError::InvalidArgument(resp.error));
        }
        Ok(true)
    }

    pub async fn list_collections(&mut self) -> Result<Vec<String>, RekhaError> {
        let req = tonic::Request::new(proto::ListCollectionsRequest {});
        let resp = self
            .inner
            .list_collections(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp.collections.into_iter().map(|c| c.name).collect())
    }

    pub async fn collection_exists(&mut self, name: &str) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(proto::CollectionExistsRequest {
            name: name.to_string(),
        });
        let resp = self
            .inner
            .collection_exists(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp.exists)
    }

    pub async fn insert(
        &mut self,
        collection: &str,
        id: u64,
        vector: Vec<f32>,
        payload: Option<Vec<u8>>,
        timestamp: u64,
        consistency: ConsistencyLevel,
    ) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(proto::InsertRequest {
            id,
            vector,
            payload: payload.map(|data| proto::Payload {
                content_type: "application/octet-stream".into(),
                data,
            }),
            collection_name: collection.to_string(),
            is_replication: false,
            timestamp,
            consistency: proto::ConsistencyLevel::from(consistency) as i32,
            origin_node_id: "client".to_string(),
        });
        let resp = self
            .inner
            .insert(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        if !resp.success {
            return Err(RekhaError::Internal(resp.error));
        }
        Ok(true)
    }

    pub async fn delete(
        &mut self,
        collection: &str,
        ids: &[u64],
        timestamp: u64,
        consistency: ConsistencyLevel,
    ) -> Result<u64, RekhaError> {
        let req = tonic::Request::new(proto::DeleteRequest {
            ids: ids.to_vec(),
            collection_name: collection.to_string(),
            timestamp,
            consistency: proto::ConsistencyLevel::from(consistency) as i32,
            is_replication: false,
            origin_node_id: "client".to_string(),
        });
        let resp = self
            .inner
            .delete(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp.deleted_count)
    }

    pub async fn fetch(
        &mut self,
        collection: &str,
        ids: &[u64],
        include_payloads: bool,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let req = tonic::Request::new(proto::FetchRequest {
            ids: ids.to_vec(),
            include_payloads,
            collection_name: collection.to_string(),
            consistency: proto::ConsistencyLevel::One as i32,
        });
        let resp = self
            .inner
            .fetch(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp
            .points
            .into_iter()
            .map(|sp| ScoredPoint {
                id: sp.id,
                score: sp.score,
                payload: sp.payload.map(|p| p.data),
                timestamp: sp.timestamp as i64,
            })
            .collect())
    }

    pub async fn search(
        &mut self,
        collection: &str,
        query: Vec<f32>,
        top_k: u32,
        params: SearchParams,
    ) -> Result<Vec<ScoredPoint>, RekhaError> {
        let req = tonic::Request::new(proto::SearchRequest {
            query_vector: query,
            top_k,
            params: Some(proto::SearchParams {
                ef_search: 0,
                nprobe: params.nprobe,
                include_payloads: params.include_payloads,
            }),
            local_only: false,
            collection_name: collection.to_string(),
            consistency: proto::ConsistencyLevel::Quorum as i32,
        });
        let resp = self
            .inner
            .search(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp
            .results
            .into_iter()
            .map(|sp| ScoredPoint {
                id: sp.id,
                score: sp.score,
                payload: sp.payload.map(|p| p.data),
                timestamp: sp.timestamp as i64,
            })
            .collect())
    }

    pub async fn health(&mut self) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(proto::HeartbeatRequest {
            node_id: "client".to_string(),
            address: String::new(),
            storage_bytes: 0,
        });
        let resp = self
            .inner
            .heartbeat(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp.success)
    }

    pub async fn import(
        &mut self,
        requests: Vec<proto::InsertRequest>,
    ) -> Result<proto::ImportResponse, RekhaError> {
        let chunk = proto::ImportChunk { requests };
        use tokio_stream::wrappers::ReceiverStream;
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.send(chunk)
            .await
            .map_err(|_| RekhaError::Internal("channel closed".into()))?;
        drop(tx);
        let resp = self
            .inner
            .import(tonic::Request::new(ReceiverStream::new(rx)))
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp)
    }

    pub async fn export(
        &mut self,
        collection: &str,
        offset: u64,
        limit: u64,
        include_vectors: bool,
        include_payloads: bool,
    ) -> Result<Vec<rekha_core::ExportedVector>, RekhaError> {
        let req = tonic::Request::new(proto::ExportRequest {
            collection_name: collection.to_string(),
            offset,
            limit,
            include_vectors,
            include_payloads,
        });
        let mut resp = self
            .inner
            .export(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        let mut all = Vec::new();
        while let Some(chunk) = resp.message().await.unwrap_or(None) {
            for v in chunk.vectors {
                all.push(rekha_core::ExportedVector {
                    id: v.id,
                    vector: v.vector,
                    payload: v.payload.map(|p| p.data),
                    timestamp: v.timestamp,
                });
            }
        }
        Ok(all)
    }

    pub async fn import_stream(
        &mut self,
        mut chunks: impl tokio_stream::Stream<Item = Vec<proto::InsertRequest>> + Send + Unpin + 'static,
    ) -> Result<proto::ImportResponse, RekhaError> {
        use tokio_stream::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        tokio::spawn(async move {
            while let Some(requests) = chunks.next().await {
                let chunk = proto::ImportChunk { requests };
                if tx.send(chunk).await.is_err() {
                    break;
                }
            }
        });

        let resp = self
            .inner
            .import(tonic::Request::new(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            ))
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp)
    }

    pub async fn export_stream(
        &mut self,
        collection: &str,
        offset: u64,
        limit: u64,
        include_vectors: bool,
        include_payloads: bool,
    ) -> Result<
        impl tokio_stream::Stream<Item = Result<rekha_core::ExportedVector, RekhaError>>,
        RekhaError,
    > {
        let req = tonic::Request::new(proto::ExportRequest {
            collection_name: collection.to_string(),
            offset,
            limit,
            include_vectors,
            include_payloads,
        });
        let resp = self
            .inner
            .export(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();

        let (tx, rx) =
            tokio::sync::mpsc::channel::<Result<rekha_core::ExportedVector, RekhaError>>(64);
        tokio::spawn(async move {
            use tokio_stream::StreamExt;
            let mut stream = resp;
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        for v in chunk.vectors {
                            let ev = rekha_core::ExportedVector {
                                id: v.id,
                                vector: v.vector,
                                payload: v.payload.map(|p| p.data),
                                timestamp: v.timestamp,
                            };
                            if tx.send(Ok(ev)).await.is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(RekhaError::Unavailable(e.to_string()))).await;
                        return;
                    }
                }
            }
        });

        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    pub async fn transfer_shard(
        &mut self,
        collection: &str,
        target_node_id: &str,
    ) -> Result<Vec<proto::TransferShardChunk>, RekhaError> {
        let req = tonic::Request::new(proto::TransferShardRequest {
            shard_id: 0,
            collection_name: collection.to_string(),
            target_node_id: target_node_id.to_string(),
        });
        let mut stream = self
            .inner
            .transfer_shard(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        let mut chunks = Vec::new();
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| RekhaError::Unavailable(e.to_string()))?;
            chunks.push(chunk);
        }
        Ok(chunks)
    }

    pub async fn repair_collection(
        &mut self,
        collection: &str,
    ) -> Result<Vec<proto::RepairProgress>, RekhaError> {
        let req = tonic::Request::new(proto::RepairCollectionRequest {
            collection_name: collection.to_string(),
        });
        let mut stream = self
            .inner
            .repair_collection(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        let mut results = Vec::new();
        use futures::StreamExt;
        while let Some(item) = stream.next().await {
            let item = item.map_err(|e| RekhaError::Unavailable(e.to_string()))?;
            results.push(item);
        }
        Ok(results)
    }

    pub async fn send_heartbeat(
        &mut self,
        node_id: &str,
        address: &str,
    ) -> Result<bool, RekhaError> {
        let req = tonic::Request::new(rekha_proto::proto::HeartbeatRequest {
            node_id: node_id.to_string(),
            address: address.to_string(),
            storage_bytes: 0,
        });
        let resp = self
            .inner
            .heartbeat(req)
            .await
            .map_err(|e| RekhaError::Unavailable(e.to_string()))?
            .into_inner();
        Ok(resp.success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_connect_fails() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(async {
            let client = Client::connect("http://127.0.0.1:1").await;
            // Should fail to connect
            assert!(client.is_err());
            Ok::<_, RekhaError>(())
        });
        assert!(result.is_ok());
    }
}
