use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Internal cluster endpoints (WAL shipping + Raft RPCs)
// ---------------------------------------------------------------------------

pub(crate) async fn raft_append_entries(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<AppendEntriesRequest<RaftTypeConfig>>,
) -> Result<Json<AppendEntriesResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    raft.append_entries(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

pub(crate) async fn raft_vote(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<VoteRequest<u64>>,
) -> Result<Json<VoteResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    raft.vote(req)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

#[allow(clippy::type_complexity)]
pub(crate) async fn raft_install_snapshot(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(group): axum::extract::Path<u32>,
    Json(req): Json<(
        openraft::Vote<u64>,
        openraft::SnapshotMeta<u64, u64>,
        Vec<u8>,
    )>,
) -> Result<Json<openraft::raft::SnapshotResponse<u64>>, StatusCode> {
    let raft = state.raft_for(group).ok_or(StatusCode::NOT_FOUND)?;
    let (vote, meta, data) = req;
    let snapshot = openraft::Snapshot {
        meta,
        snapshot: Box::new(std::io::Cursor::new(data)),
    };
    raft.install_full_snapshot(vote, snapshot)
        .await
        .map(Json)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Receive one chunk of a snapshot and, on the last, install it.
///
/// Raw bytes rather than JSON: the previous endpoint took the whole archive in a
/// `Vec<u8>` field, base64-encoded, capped by the router's 32 MiB body limit,
/// with a 200 ms deadline. A replica that had fallen behind could not send one,
/// and this is the only path that repairs such a replica.
///
/// Staging is on disk under the data directory, at an exact offset, so a
/// half-transferred snapshot costs disk rather than memory and a retried chunk
/// is refused instead of being appended twice. The install itself still reads
/// the archive into memory (`SnapshotData` is a `Cursor<Vec<u8>>`); removing
/// that copy is the streaming follow-up, and is why this is chunked transport,
/// not chunked restore.
pub(crate) async fn raft_snapshot_chunk(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((group, snapshot_id)): axum::extract::Path<(u32, String)>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<openraft::raft::SnapshotResponse<u64>>, (StatusCode, Json<serde_json::Value>)> {
    fn bad(detail: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": detail.into() })),
        )
    }

    fn header_u64(
        headers: &axum::http::HeaderMap,
        name: &str,
    ) -> Result<u64, (StatusCode, Json<serde_json::Value>)> {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .ok_or_else(|| bad(format!("`{name}` is required and must be a whole number")))
    }

    // The id becomes part of a file name in the staging directory.
    if snapshot_id.is_empty()
        || snapshot_id.contains('/')
        || snapshot_id.contains('\\')
        || snapshot_id.contains("..")
    {
        return Err(bad("snapshot id is not a usable file name"));
    }

    let offset = header_u64(&headers, rekha_cluster::network::SNAPSHOT_OFFSET_HEADER)?;
    let total = header_u64(&headers, rekha_cluster::network::SNAPSHOT_TOTAL_HEADER)?;
    let is_final = headers
        .get(rekha_cluster::network::SNAPSHOT_FINAL_HEADER)
        .and_then(|v| v.to_str().ok())
        == Some("1");

    let vote: openraft::Vote<u64> = headers
        .get(rekha_cluster::network::SNAPSHOT_VOTE_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| serde_json::from_str(v).ok())
        .ok_or_else(|| bad("`x-rekha-snapshot-vote` is required and must be JSON"))?;
    let meta: openraft::SnapshotMeta<u64, u64> = headers
        .get(rekha_cluster::network::SNAPSHOT_META_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| serde_json::from_str(v).ok())
        .ok_or_else(|| bad("`x-rekha-snapshot-meta` is required and must be JSON"))?;

    let end = offset
        .checked_add(body.len() as u64)
        .ok_or_else(|| bad("chunk offset overflows"))?;
    if end > total {
        return Err(bad(format!(
            "chunk covers bytes {offset}..{end}, beyond the declared total of {total}"
        )));
    }
    if is_final && end != total {
        return Err(bad(format!(
            "the final chunk ends at byte {end}, but the snapshot is declared as {total} bytes"
        )));
    }

    let raft = state.raft_for(group).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("no Raft group {group} on this node") })),
        )
    })?;

    let staging_dir = state
        .engine
        .storage_dir()
        .join("snapshots")
        .join(".incoming");
    std::fs::create_dir_all(&staging_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({ "error": format!("cannot create the staging directory: {e}") }),
            ),
        )
    })?;
    let staged = staging_dir.join(format!("{group}-{snapshot_id}.part"));

    // One transfer at a time: appends are offset-checked against the file, and
    // two concurrent chunks would both observe the same length.
    let _guard = state.snapshot_staging.lock().await;

    let existing = std::fs::metadata(&staged).map(|m| m.len()).unwrap_or(0);
    // Offset zero is a sender starting over, which is exactly what openraft does
    // after a failed transfer: discard the partial file rather than answering
    // 409 and costing a whole extra attempt.
    let existing = if offset == 0 && existing > 0 {
        let _ = std::fs::remove_file(&staged);
        0
    } else {
        existing
    };
    if offset != existing {
        // Start over rather than splice a chunk into the wrong place. The next
        // attempt begins at zero, which is what openraft does after a failure.
        let _ = std::fs::remove_file(&staged);
        return Err((
            StatusCode::CONFLICT,
            Json(serde_json::json!({
                "error": format!(
                    "chunk starts at byte {offset}, but {existing} bytes are already staged; \
                     the transfer must restart from zero"
                )
            })),
        ));
    }

    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&staged)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": format!("cannot stage the chunk: {e}") })),
                )
            })?;
        file.write_all(&body).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("cannot write the chunk: {e}") })),
            )
        })?;
    }

    if !is_final {
        // No meta is returned for an intermediate chunk: openraft only reads the
        // response to the last one.
        return Ok(Json(openraft::raft::SnapshotResponse::new(vote)));
    }

    let read_path = staged.clone();
    let data = tokio::task::spawn_blocking(move || std::fs::read(&read_path))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("staging task failed: {e}") })),
            )
        })?
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({ "error": format!("cannot read the staged snapshot: {e}") }),
                ),
            )
        })?;

    // The staged file is only needed until it is installed. Leaving it would
    // double the disk a snapshot costs, permanently, because nothing else walks
    // this directory.
    let _ = std::fs::remove_file(&staged);

    let snapshot = openraft::Snapshot {
        meta,
        snapshot: Box::new(std::io::Cursor::new(data)),
    };
    raft.install_full_snapshot(vote, snapshot)
        .await
        .map(Json)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": format!("snapshot install failed: {e}") })),
            )
        })
}

pub(crate) async fn shard_ensure_started(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    if shards.hosts(shard) {
        return Ok(StatusCode::OK);
    }

    if let (Some(raft_config), Some(node_id)) = (&state.raft_config, state.node_id) {
        let wal_dir = state.engine.storage_dir();
        let data_dir = wal_dir.parent().unwrap_or(wal_dir);
        let log_dir = data_dir.join("raft_log");
        let _ = std::fs::create_dir_all(&log_dir);
        let mut log_store = match rekha_cluster::RedbLogStore::open(
            log_dir.join(format!("shard-{}.redb", shard.0)),
        ) {
            Ok(s) => s,
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };
        let purge_watermark = log_store.purge_watermark().await;
        let mut shard_sm = rekha_cluster::ClusterStateMachine::for_shard(node_id, shard);
        shard_sm.set_catalog(state.engine.catalog());
        shard_sm.set_engine(state.engine.clone() as Arc<dyn rekha_cluster::EngineSnapshotProvider>);
        shard_sm.set_apply_failure_flag(state.apply_failure.clone());
        let snapshot_root = wal_dir.join("snapshots");
        if let Ok(store) = rekha_cluster::SnapshotStore::open(
            snapshot_root.join(format!("shard-{}", shard.0)),
            shard_sm.group_label(),
        ) {
            shard_sm.set_snapshot_store(Arc::new(store));
        }

        if let Some(boundary) = purge_watermark {
            shard_sm.seed_nodes(
                state
                    .peers
                    .ids()
                    .into_iter()
                    .filter_map(|id| state.peers.addr(id).map(|addr| (id, addr))),
            );
            let _ = shard_sm.seed_placements(state.engine.catalog().as_ref());
            let voters = shards.owners(shard);
            shard_sm.resume_from(
                Some(boundary),
                openraft::StoredMembership::new(
                    Some(boundary),
                    openraft::Membership::new(vec![voters.into_iter().collect()], ()),
                ),
            );
        }

        let network = rekha_cluster::RaftNetworkFactoryImpl::new(state.peers.clone())
            .with_group(rekha_cluster::ShardCluster::group_id(shard))
            .with_cluster_secret(state.cluster_secret.clone());

        let shard_raft = match openraft::Raft::<rekha_cluster::RaftTypeConfig>::new(
            node_id,
            raft_config.clone(),
            network,
            log_store,
            shard_sm,
        )
        .await
        {
            Ok(r) => Arc::new(r),
            Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
        };

        shards.insert(shard, shard_raft);
        let _ = state.engine.materialize_shard(shard);
        return Ok(StatusCode::OK);
    }

    let _ = state.engine.materialize_shard(shard);
    Ok(StatusCode::OK)
}

pub(crate) async fn shard_drop(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    if let Some(shards) = state.shards.as_ref()
        && let Some(raft) = shards.remove(shard)
    {
        let _ = raft.shutdown().await;
    }
    let _ = state.engine.drop_shard(shard);
    Ok(StatusCode::OK)
}

pub(crate) async fn shard_batch_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardBatchRequest>,
) -> Result<Json<ShardBatchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let shard = rekha_cluster::ShardId(shard_num);
    let count = match commit_write_for_shard(&state, shard, req.collection_id, req.kind).await {
        Ok(c) => c,
        Err(ApiError::NotLeader { leader }) => {
            let mut headers = axum::http::HeaderMap::new();
            if let Some(id) = leader
                && let Ok(v) = axum::http::HeaderValue::from_str(&id.to_string())
            {
                headers.insert(
                    axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                    v,
                );
            }
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not shard leader", "leader": leader })),
            ));
        }
        Err(ApiError::Engine(e)) => {
            return Err((
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok(Json(ShardBatchResponse { count }))
}

pub(crate) async fn shard_prepare_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardPrepareRequest>,
) -> Result<Json<ShardBatchResponse>, (StatusCode, Json<serde_json::Value>)> {
    let shard = rekha_cluster::ShardId(shard_num);
    let count = match prepare_write_for_shard(&state, shard, req.tx_id, req.collection_id, req.kind)
        .await
    {
        Ok(c) => c,
        Err(ApiError::NotLeader { leader }) => {
            let mut headers = axum::http::HeaderMap::new();
            if let Some(id) = leader
                && let Ok(v) = axum::http::HeaderValue::from_str(&id.to_string())
            {
                headers.insert(
                    axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                    v,
                );
            }
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not shard leader", "leader": leader })),
            ));
        }
        Err(ApiError::Engine(e)) => {
            return Err((
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ));
        }
    };
    Ok(Json(ShardBatchResponse { count }))
}

pub(crate) async fn shard_commit_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardCommitRequest>,
) -> Result<StatusCode, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    match commit_tx_for_shard(&state, shard, req.tx_id).await {
        Ok(()) => Ok(StatusCode::OK),
        Err(ApiError::NotLeader { .. }) => Err(StatusCode::SERVICE_UNAVAILABLE),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub(crate) async fn shard_abort_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardAbortRequest>,
) -> StatusCode {
    let shard = rekha_cluster::ShardId(shard_num);
    let _ = abort_tx_for_shard(&state, shard, req.tx_id).await;
    StatusCode::OK
}

pub(crate) async fn tx_status_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(tx_id): axum::extract::Path<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let committed = shards.is_tx_committed(&tx_id);
    Ok(Json(serde_json::json!({ "committed": committed })))
}

pub(crate) async fn tx_commit_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TxCommitRequest>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    dispatch_metadata_commit_tx(&state, req.tx_id, req.shards)
        .await
        .map(|_| StatusCode::OK)
        .map_err(|e| match e {
            ApiError::NotLeader { leader } => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({ "error": "not metadata leader", "leader": leader })),
            ),
            ApiError::Engine(e) => (
                engine_error_status(&e),
                Json(serde_json::json!({ "error": e.to_string() })),
            ),
        })
}

pub(crate) async fn tx_abort_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TxAbortRequest>,
) -> StatusCode {
    dispatch_metadata_abort_tx(&state, req.tx_id).await;
    StatusCode::OK
}

pub(crate) async fn shard_membership_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let m = raft.metrics().borrow().clone();
    let (voters, learner_matched, last_log_index) = group_learners(&m);
    let group = format!("shard:{}", shard.0);
    Ok(Json(membership_body(
        voters,
        learner_matched,
        last_log_index,
        last_promotion_for(state.shards.as_ref(), &group),
    )))
}

pub(crate) async fn shard_query_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardQueryRequest>,
) -> Result<Json<Vec<Vec<rekha_engine::ScoredPoint>>>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    if !leads {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await;
    let engine = state.engine.clone();
    let collection_id = req.collection_id;
    let query_embeddings = req.query_embeddings;
    let n_results = req.n_results;
    let where_filter = req.where_filter;
    let where_document = req.where_document;
    let with_embeddings = req.with_embeddings;

    let hits = offload(move || {
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
    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(hits))
}

pub(crate) async fn shard_count_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
    Json(collection_id): Json<Uuid>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    let local = params
        .get("consistency")
        .map(|v| v == "local")
        .unwrap_or(false);
    if !leads && !local {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    if leads && !local {
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            raft.ensure_linearizable(),
        )
        .await;
    }
    let count = state
        .engine
        .count_partition(&collection_id, Some(shard))
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Json(serde_json::json!({ "count": count })))
}

pub(crate) async fn shard_get_handler(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(shard_num): axum::extract::Path<u32>,
    Json(req): Json<ShardGetRequest>,
) -> Result<Json<Vec<rekha_engine::Record>>, StatusCode> {
    let shard = rekha_cluster::ShardId(shard_num);
    let Some(shards) = state.shards.as_ref() else {
        return Err(StatusCode::NOT_FOUND);
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(StatusCode::NOT_FOUND);
    };
    let leads = raft.metrics().borrow().current_leader == Some(raft.metrics().borrow().id);
    if !leads {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        raft.ensure_linearizable(),
    )
    .await;
    let records = state
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
        .map_err(|_| StatusCode::NOT_FOUND)?;
    Ok(Json(records))
}
