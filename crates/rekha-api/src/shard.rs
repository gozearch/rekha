use crate::prelude::*;
use crate::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardWriteKind {
    Add {
        ids: Vec<Id>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<Metadata>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Upsert {
        ids: Vec<Id>,
        embeddings: Vec<Vec<f32>>,
        metadatas: Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Update {
        ids: Vec<Id>,
        embeddings: Option<Vec<Vec<f32>>>,
        metadatas: Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
        documents: Option<Vec<Option<String>>>,
    },
    Delete {
        ids: Vec<Id>,
    },
    DeleteWhere {
        ids: Option<Vec<Id>>,
        filter: Option<WhereFilter>,
        document: Option<rekha_core::filter::WhereDocument>,
        limit: Option<usize>,
    },
    Prepared {
        ops: Vec<rekha_core::op::Operation>,
        dimension: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardBatchRequest {
    pub collection_id: Uuid,
    pub kind: ShardWriteKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPrepareRequest {
    pub tx_id: Uuid,
    pub collection_id: Uuid,
    pub kind: ShardWriteKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCommitRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAbortRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxCommitRequest {
    pub tx_id: Uuid,
    pub shards: Vec<rekha_cluster::ShardId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxAbortRequest {
    pub tx_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardBatchResponse {
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardQueryRequest {
    pub collection_id: Uuid,
    pub query_embeddings: Vec<Vec<f32>>,
    pub n_results: usize,
    pub where_filter: Option<WhereFilter>,
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    pub with_embeddings: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardGetRequest {
    pub collection_id: Uuid,
    pub ids: Option<Vec<Id>>,
    pub where_filter: Option<WhereFilter>,
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    pub limit: Option<usize>,
    pub offset: usize,
}

pub(crate) type SubBatchAdd = (
    Vec<Id>,
    Vec<Vec<f32>>,
    Option<Vec<Option<Metadata>>>,
    Option<Vec<Option<String>>>,
);
pub(crate) type SubBatchUpsert = (
    Vec<Id>,
    Vec<Vec<f32>>,
    Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
    Option<Vec<Option<String>>>,
);
pub(crate) type SubBatchUpdate = (
    Vec<Id>,
    Option<Vec<Vec<f32>>>,
    Option<Vec<Option<rekha_core::types::MetadataPatch>>>,
    Option<Vec<Option<String>>>,
);

pub(crate) fn is_partitioned(state: &AppState, collection_id: &Uuid) -> bool {
    if state.shards.is_none() {
        return false;
    }
    match state.engine.get_collection_by_id(collection_id) {
        Ok(Some(record)) => record.shard.is_none(),
        _ => false,
    }
}

pub(crate) fn is_partitioned_requested(explicit: Option<bool>, meta: Option<&Metadata>) -> bool {
    if let Some(p) = explicit {
        return p;
    }
    if let Some(m) = meta
        && let Some(val) = m.get("partitioned")
    {
        let matches = match val {
            rekha_core::types::MetadataValue::Str(s) => s == "true",
            rekha_core::types::MetadataValue::Bool(b) => *b,
            _ => false,
        };
        if matches {
            return true;
        }
    }
    std::env::var("REKHA_INTRA_SHARDING")
        .map(|v| v == "always" || v == "1" || v == "true")
        .unwrap_or(false)
}

pub(crate) async fn check_shard_read_barrier(
    raft: &Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>,
) -> Result<(), ApiError> {
    let single_voter = {
        let metrics = raft.metrics();
        let m = metrics.borrow();
        m.membership_config.membership().voter_ids().count() <= 1
    };
    if single_voter {
        return Ok(());
    }

    match tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await
    {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => {
            tracing::warn!(%err, "rejecting shard read: this node can no longer prove it leads");
            let leader = raft.metrics().borrow().current_leader;
            Err(ApiError::NotLeader { leader })
        }
        Err(_) => {
            tracing::warn!("rejecting shard read: no quorum confirmation within deadline");
            let leader = raft.metrics().borrow().current_leader;
            Err(ApiError::NotLeader { leader })
        }
    }
}

pub(crate) async fn ensure_collection_on_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
) -> Result<(), ApiError> {
    let record = match state.engine.get_collection_by_id(&collection_id)? {
        Some(r) => r,
        None => {
            return Err(ApiError::Engine(EngineError::Unavailable(format!(
                "collection {collection_id} metadata not yet applied on shard replica; retry shortly"
            ))));
        }
    };
    if let Some(shards) = state.shards.as_ref()
        && let Some(raft) = shards.raft(shard)
        && !shards.is_ensured(collection_id, shard)
    {
        raft.client_write(ClusterOperation::AddCollection {
            config: record.config.clone(),
            shard: Some(shard),
        })
        .await
        .map_err(|e| {
            EngineError::Recovery(format!(
                "could not tell shard {shard} about collection {collection_id}: {e}"
            ))
        })?;
        shards.mark_ensured(collection_id, shard);
    }
    Ok(())
}

pub(crate) async fn resolve_shard_write_kind(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<(rekha_engine::PreparedBatch, usize), ApiError> {
    match kind {
        ShardWriteKind::Add {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Vec<Embedding> = embeddings.into_iter().map(Embedding::from).collect();
            let p = offload(move || {
                engine.prepare_add_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    &embs,
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Upsert {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Vec<Embedding> = embeddings.into_iter().map(Embedding::from).collect();
            let p = offload(move || {
                engine.prepare_upsert_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    &embs,
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Update {
            ids,
            embeddings,
            metadatas,
            documents,
        } => {
            let engine = state.engine.clone();
            let embs: Option<Vec<Embedding>> =
                embeddings.map(|e| e.into_iter().map(Embedding::from).collect());
            let p = offload(move || {
                engine.prepare_update_partition(
                    &collection_id,
                    Some(shard),
                    &ids,
                    embs.as_deref(),
                    metadatas.as_deref(),
                    documents.as_deref(),
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Delete { ids } => {
            let engine = state.engine.clone();
            let n = ids.len();
            let p =
                offload(move || engine.prepare_delete_partition(&collection_id, Some(shard), &ids))
                    .await?;
            Ok((p, n))
        }
        ShardWriteKind::DeleteWhere {
            ids,
            filter,
            document,
            limit,
        } => {
            let engine = state.engine.clone();
            let p = offload(move || {
                engine.prepare_delete_where_partition(
                    &collection_id,
                    Some(shard),
                    ids.as_deref(),
                    filter.as_ref(),
                    document.as_ref(),
                    limit,
                )
            })
            .await?;
            let n = p.ops.len();
            Ok((p, n))
        }
        ShardWriteKind::Prepared { ops, dimension } => {
            let n = ops.len();
            Ok((rekha_engine::PreparedBatch { ops, dimension }, n))
        }
    }
}

pub(crate) async fn commit_write_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<usize, ApiError> {
    let lock = state.write_lock(collection_id, Some(shard));
    let _guard = lock.lock_owned().await;

    ensure_collection_on_shard(state, shard, collection_id).await?;
    let (prepared, count) = resolve_shard_write_kind(state, shard, collection_id, kind).await?;
    commit_batch_for_shard(state, shard, collection_id, prepared).await?;
    Ok(count)
}

pub(crate) async fn prepare_write_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
    collection_id: Uuid,
    kind: ShardWriteKind,
) -> Result<usize, ApiError> {
    let lock = state.write_lock(collection_id, Some(shard));
    let _guard = lock.lock_owned().await;

    ensure_collection_on_shard(state, shard, collection_id).await?;
    let (prepared, count) = resolve_shard_write_kind(state, shard, collection_id, kind).await?;

    if prepared.ops.is_empty() {
        return Ok(0);
    }

    let ids: Vec<String> = prepared.ops.iter().map(|o| o.id().to_string()).collect();
    state
        .engine
        .reserve_tx_ids(&collection_id, Some(shard), tx_id, &ids)?;

    let Some(shards) = state.shards.as_ref() else {
        let _ = state
            .engine
            .release_tx_ids(&collection_id, Some(shard), &tx_id);
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        let _ = state
            .engine
            .release_tx_ids(&collection_id, Some(shard), &tx_id);
        return Err(ApiError::Engine(EngineError::NotReplica {
            collection: collection_id.to_string(),
            shard: Some(shard),
        }));
    };

    let operation = ClusterOperation::PrepareBatch {
        tx_id,
        collection_id,
        ops: prepared.ops,
        dimension: prepared.dimension,
    };

    match raft.client_write(operation).await {
        Ok(response) => {
            match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                Ok(applied) => match applied.error {
                    None => {
                        shards.mark_tx_staged(shard, tx_id);
                        Ok(count)
                    }
                    Some(error) => {
                        let _ = state
                            .engine
                            .release_tx_ids(&collection_id, Some(shard), &tx_id);
                        Err(ApiError::Engine(EngineError::Recovery(format!(
                            "2PC prepare committed at log index {} but could not stage: {error}",
                            applied.seq
                        ))))
                    }
                },
                Err(_) => {
                    shards.mark_tx_staged(shard, tx_id);
                    Ok(count)
                }
            }
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => {
            let _ = state
                .engine
                .release_tx_ids(&collection_id, Some(shard), &tx_id);
            Err(ApiError::NotLeader {
                leader: forward.leader_id.or(forward.leader_node),
            })
        }
        Err(e) => {
            let _ = state
                .engine
                .release_tx_ids(&collection_id, Some(shard), &tx_id);
            Err(ApiError::Engine(EngineError::Recovery(format!(
                "2PC prepare failed on shard {shard}: {e}"
            ))))
        }
    }
}

pub(crate) async fn commit_tx_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(ApiError::Engine(EngineError::Unavailable(format!(
            "shard {shard} not hosted here"
        ))));
    };

    let operation = ClusterOperation::CommitBatch { tx_id };
    match raft.client_write(operation).await {
        Ok(_) => {
            shards.mark_tx_resolved(&tx_id);
            Ok(())
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => Err(ApiError::NotLeader {
            leader: forward.leader_id.or(forward.leader_node),
        }),
        Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
            "commit 2PC tx {tx_id} failed on shard {shard}: {e}"
        )))),
    }
}

pub(crate) async fn abort_tx_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Ok(());
    };
    shards.mark_tx_resolved(&tx_id);
    let Some(raft) = shards.raft(shard) else {
        return Ok(());
    };

    let operation = ClusterOperation::AbortBatch { tx_id };
    let _ = raft.client_write(operation).await;
    Ok(())
}

/// One reply kind for a shard write endpoint.
///
/// `commit` answers with no body and `batch`/`prepare` answer with a count, so
/// the shared dispatcher decodes through this instead of duplicating the
/// request loop once per reply shape.
pub(crate) trait ShardWriteReply: Sized {
    fn decode(body: &[u8]) -> Result<Self, String>;
}

impl ShardWriteReply for usize {
    fn decode(body: &[u8]) -> Result<Self, String> {
        serde_json::from_slice::<ShardBatchResponse>(body)
            .map(|r| r.count)
            .map_err(|e| e.to_string())
    }
}

impl ShardWriteReply for () {
    fn decode(_body: &[u8]) -> Result<Self, String> {
        Ok(())
    }
}

/// Send one shard write to whichever node can commit it.
///
/// The three write endpoints (`batch`, `prepare`, `commit`) had identical retry
/// loops that differed only in the path segment and the reply shape. That is
/// exactly the kind of copy where a fix lands in one arm and not the others, so
/// the loop lives here once:
///
/// 1. if this node leads the shard, run the local path;
/// 2. otherwise try each owner, following the leader hint a `503` carries;
/// 3. retry until a deadline, and report `Unavailable` when none answers.
async fn dispatch_shard_write<B, O, F>(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    endpoint: &str,
    req: &B,
    local: impl Fn() -> F,
    unsharded: Option<O>,
    unavailable: String,
) -> Result<O, ApiError>
where
    B: serde::Serialize + ?Sized,
    O: ShardWriteReply,
    F: std::future::Future<Output = Result<O, ApiError>>,
{
    let Some(shards) = state.shards.as_ref() else {
        return match unsharded {
            Some(value) => Ok(value),
            None => Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            ))),
        };
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        if let Some(raft) = shards.raft(shard)
            && raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id)
        {
            match local().await {
                Ok(value) => return Ok(value),
                Err(ApiError::NotLeader { .. }) => {}
                Err(e) => return Err(e),
            }
        }

        let mut should_retry = false;
        for target in state.shard_targets(shard) {
            let Some(url) = state.peers.url(target) else {
                continue;
            };
            let url = format!("{url}/internal/shard/{}/{endpoint}", shard.0);
            match state.internal_post(&url).json(req).send().await {
                Ok(resp) if resp.status().is_success() => {
                    return decode_shard_reply(resp, shard, endpoint).await;
                }
                Ok(resp) if resp.status() == StatusCode::SERVICE_UNAVAILABLE => {
                    should_retry = true;
                    // A shard group's leader is independent of this node's, so
                    // the hint is the only way to avoid polling every owner.
                    if let Some(leader_id) = resp
                        .headers()
                        .get(crate::leadership::LEADER_HEADER)
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok())
                        && let Some(leader_url) = state.peers.url(leader_id)
                        && let Ok(resp2) = state
                            .internal_post(format!(
                                "{leader_url}/internal/shard/{}/{endpoint}",
                                shard.0
                            ))
                            .json(req)
                            .send()
                            .await
                        && resp2.status().is_success()
                    {
                        return decode_shard_reply(resp2, shard, endpoint).await;
                    }
                }
                Ok(resp) if resp.status().is_client_error() => {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(ApiError::Engine(EngineError::Validation(body)));
                }
                _ => should_retry = true,
            }
        }

        if should_retry {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        } else {
            break;
        }
    }

    Err(ApiError::Engine(EngineError::Unavailable(unavailable)))
}

async fn decode_shard_reply<O: ShardWriteReply>(
    resp: reqwest::Response,
    shard: rekha_cluster::ShardId,
    endpoint: &str,
) -> Result<O, ApiError> {
    let body = resp.bytes().await.map_err(|e| {
        ApiError::Engine(EngineError::Recovery(format!(
            "could not read {endpoint} response from shard {}: {e}",
            shard.0
        )))
    })?;
    O::decode(&body).map_err(|e| {
        ApiError::Engine(EngineError::Recovery(format!(
            "invalid {endpoint} response from shard {}: {e}",
            shard.0
        )))
    })
}

pub(crate) async fn dispatch_shard_prepare(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardPrepareRequest,
) -> Result<usize, ApiError> {
    dispatch_shard_write(
        state,
        shard,
        "prepare",
        &req,
        || prepare_write_for_shard(state, shard, req.tx_id, req.collection_id, req.kind.clone()),
        None,
        format!(
            "could not reach leader for shard {} to prepare 2PC",
            shard.0
        ),
    )
    .await
}

pub(crate) async fn dispatch_shard_commit(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) -> Result<(), ApiError> {
    dispatch_shard_write(
        state,
        shard,
        "commit",
        &ShardCommitRequest { tx_id },
        || commit_tx_for_shard(state, shard, tx_id),
        Some(()),
        format!(
            "could not reach leader for shard {} to commit 2PC transaction {tx_id}",
            shard.0
        ),
    )
    .await
}

pub(crate) async fn dispatch_shard_batch(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardBatchRequest,
) -> Result<usize, ApiError> {
    dispatch_shard_write(
        state,
        shard,
        "batch",
        &req,
        || commit_write_for_shard(state, shard, req.collection_id, req.kind.clone()),
        None,
        format!("could not reach leader for shard {}", shard.0),
    )
    .await
}

pub(crate) async fn dispatch_metadata_commit_tx(
    state: &AppState,
    tx_id: Uuid,
    shards: Vec<rekha_cluster::ShardId>,
) -> Result<(), ApiError> {
    let Some(raft) = state.raft.as_ref() else {
        return Ok(());
    };

    let op = ClusterOperation::CommitTx {
        tx_id,
        shards: shards.clone(),
    };

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        let (is_leader, leader_id) = {
            let m = raft.metrics();
            let b = m.borrow();
            (b.current_leader == Some(b.id), b.current_leader)
        };
        if is_leader {
            match raft.client_write(op.clone()).await {
                Ok(_) => return Ok(()),
                Err(openraft::error::RaftError::APIError(
                    openraft::error::ClientWriteError::ForwardToLeader(_),
                )) => {}
                Err(e) => {
                    return Err(ApiError::Engine(EngineError::Recovery(format!(
                        "failed to commit 2PC transaction {tx_id} on metadata group: {e}"
                    ))));
                }
            }
        }

        // Try forwarding to the metadata leader
        if let Some(leader) = leader_id
            && let Some(url) = state.peers.url(leader)
        {
            let req = TxCommitRequest {
                tx_id,
                shards: shards.clone(),
            };
            if let Ok(resp) = state
                .internal_post(format!("{url}/internal/cluster/tx/commit"))
                .json(&req)
                .send()
                .await
                && resp.status().is_success()
            {
                return Ok(());
            }
        }

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    }

    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not reach metadata leader to commit 2PC transaction {tx_id}"
    ))))
}

pub(crate) async fn dispatch_metadata_abort_tx(state: &AppState, tx_id: Uuid) {
    let Some(raft) = state.raft.as_ref() else {
        return;
    };
    let op = ClusterOperation::AbortTx { tx_id };
    let (is_leader, leader_id) = {
        let m = raft.metrics();
        let b = m.borrow();
        (b.current_leader == Some(b.id), b.current_leader)
    };
    if is_leader {
        let _ = raft.client_write(op).await;
        return;
    }
    if let Some(leader) = leader_id
        && let Some(url) = state.peers.url(leader)
    {
        let req = TxAbortRequest { tx_id };
        let _ = state
            .internal_post(format!("{url}/internal/cluster/tx/abort"))
            .json(&req)
            .send()
            .await;
    }
}

pub(crate) async fn dispatch_shard_abort(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    tx_id: Uuid,
) {
    let Some(shards) = state.shards.as_ref() else {
        return;
    };
    let is_leader = if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let b = m.borrow();
        b.current_leader == Some(b.id)
    } else {
        false
    };
    if is_leader {
        let _ = abort_tx_for_shard(state, shard, tx_id).await;
        return;
    }

    let req = ShardAbortRequest { tx_id };
    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/abort", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
        {
            return;
        }
    }
}

pub(crate) async fn write_partitioned_2pc(
    state: &AppState,
    collection_id: Uuid,
    sub_batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)>,
) -> Result<(), ApiError> {
    let tx_id = Uuid::new_v4();
    let prepare_futs = sub_batches.into_iter().map(|(shard, kind)| async move {
        let req = ShardPrepareRequest {
            tx_id,
            collection_id,
            kind,
        };
        dispatch_shard_prepare(state, shard, req)
            .await
            .map(|count| (shard, count))
    });

    let prepare_results = futures::future::join_all(prepare_futs).await;
    let mut prepared_shards = Vec::with_capacity(prepare_results.len());
    let mut first_err = None;

    for res in prepare_results {
        match res {
            Ok((shard, _)) => prepared_shards.push(shard),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
    }

    if let Some(err) = first_err {
        let abort_futs = prepared_shards.into_iter().map(|shard| async move {
            dispatch_shard_abort(state, shard, tx_id).await;
        });
        futures::future::join_all(abort_futs).await;
        dispatch_metadata_abort_tx(state, tx_id).await;
        return Err(err);
    }

    if let Err(e) = dispatch_metadata_commit_tx(state, tx_id, prepared_shards.clone()).await {
        let abort_futs = prepared_shards.into_iter().map(|shard| async move {
            dispatch_shard_abort(state, shard, tx_id).await;
        });
        futures::future::join_all(abort_futs).await;
        dispatch_metadata_abort_tx(state, tx_id).await;
        return Err(e);
    }

    let commit_futs = prepared_shards
        .into_iter()
        .map(|shard| async move { dispatch_shard_commit(state, shard, tx_id).await });
    let commit_results = futures::future::join_all(commit_futs).await;
    for res in commit_results {
        res?;
    }
    Ok(())
}

pub(crate) async fn dispatch_shard_query(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardQueryRequest,
) -> Result<Vec<Vec<rekha_engine::ScoredPoint>>, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            let engine = state.engine.clone();
            let collection_id = req.collection_id;
            let query_embeddings = req.query_embeddings;
            let n_results = req.n_results;
            let where_filter = req.where_filter;
            let where_document = req.where_document;
            let with_embeddings = req.with_embeddings;
            return offload(move || {
                let opts = QueryOptions {
                    ef: 0,
                    where_filter,
                    where_document,
                    oversampling: 4,
                };
                let embeddings_vec: Vec<Embedding> =
                    query_embeddings.into_iter().map(Embedding::from).collect();
                engine.query_many_partition(
                    &collection_id,
                    Some(shard),
                    &embeddings_vec,
                    n_results,
                    &opts,
                    with_embeddings,
                )
            })
            .await
            .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/query", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(hits) = resp.json::<Vec<Vec<rekha_engine::ScoredPoint>>>().await
        {
            return Ok(hits);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not query shard {}",
        shard.0
    ))))
}

pub(crate) async fn dispatch_shard_count(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
) -> Result<u64, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            return state
                .engine
                .count_partition(&collection_id, Some(shard))
                .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/count", shard.0))
            .json(&collection_id)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(val) = resp.json::<serde_json::Value>().await
            && let Some(c) = val.get("count").and_then(|v| v.as_u64())
        {
            return Ok(c);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not count shard {}",
        shard.0
    ))))
}

pub(crate) async fn dispatch_shard_get(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    req: ShardGetRequest,
) -> Result<Vec<rekha_engine::Record>, ApiError> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };

    if let Some(raft) = shards.raft(shard) {
        let m = raft.metrics();
        let leads = m.borrow().current_leader == Some(m.borrow().id);
        if leads && check_shard_read_barrier(&raft).await.is_ok() {
            return state
                .engine
                .get_records_partition(
                    &req.collection_id,
                    Some(shard),
                    req.ids.as_deref(),
                    req.where_filter.as_ref(),
                    req.where_document.as_ref(),
                    req.limit,
                    req.offset,
                )
                .map_err(ApiError::Engine);
        }
    }

    for target in state.shard_targets(shard) {
        let Some(url) = state.peers.url(target) else {
            continue;
        };
        if let Ok(resp) = state
            .internal_post(format!("{url}/internal/shard/{}/get", shard.0))
            .json(&req)
            .send()
            .await
            && resp.status().is_success()
            && let Ok(records) = resp.json::<Vec<rekha_engine::Record>>().await
        {
            return Ok(records);
        }
    }
    Err(ApiError::Engine(EngineError::Unavailable(format!(
        "could not get records from shard {}",
        shard.0
    ))))
}
