use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

pub(crate) async fn create_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<Json<CollectionResponse>, ApiError> {
    require_scope_permission(
        &auth.permissions,
        "collection.create",
        &state.tenant,
        &state.database,
    )?;
    validate_name(&req.name)?;
    if req.dimension > MAX_DIMENSION {
        return Err(ApiError::Engine(validation_err(format!(
            "dimension {} exceeds max {MAX_DIMENSION}",
            req.dimension
        ))));
    }
    let mut config = CollectionConfig::new(req.name.clone(), req.dimension, req.distance);
    config.tenant = state.tenant.clone();
    config.database = state.database.clone();
    let partitioned = is_partitioned_requested(req.partitioned, None);
    let record = create_collection_clustered(&state, &config, partitioned).await?;
    // Known to be empty by construction, and a fresh collection answers `Some(0)`
    // rather than `None`: this is a fact the server just created, not a count it
    // failed to fetch.
    let count = Some(0);

    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count,
        shard: record.shard.map(|shard| shard.0),
    }))
}

pub(crate) async fn list_collections(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
) -> Result<Json<Vec<CollectionResponse>>, ApiError> {
    require_scope_permission(
        &auth.permissions,
        "collection.list",
        &state.tenant,
        &state.database,
    )?;
    let records = state
        .engine
        .list_collections(&state.tenant, &state.database)?;
    let ids: Vec<uuid::Uuid> = records.iter().map(|r| r.config.id).collect();
    // One unreachable shard owner must not fail the whole list, so a missing
    // count is reported as `null` for that row instead of aborting the response
    // or inventing a zero.
    let counts = counts_for(&state, &ids).await.unwrap_or_default();
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let count = counts.get(&record.config.id).copied();
        out.push(CollectionResponse {
            id: record.config.id,
            name: record.config.name,
            dimension: record.config.dimension,
            distance: record.config.space,
            count,
            shard: record.shard.map(|shard| shard.0),
        });
    }
    Ok(Json(out))
}

pub(crate) async fn get_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<CollectionResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.get",
        &id,
    )?;
    let record = state
        .engine
        .get_collection_by_id(&id)?
        .ok_or_else(|| EngineError::CollectionNotFound(id.to_string()))?;
    let count = counts_for(&state, &[id])
        .await?
        .get(&id)
        .copied()
        .ok_or_else(|| {
            ApiError::Engine(EngineError::Unavailable(format!(
                "no replica answered the count for collection {id}"
            )))
        })?;
    Ok(Json(CollectionResponse {
        id: record.config.id,
        name: record.config.name,
        dimension: record.config.dimension,
        distance: record.config.space,
        count: Some(count),
        shard: record.shard.map(|shard| shard.0),
    }))
}

pub(crate) async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.delete",
        &id,
    )?;
    delete_collection_clustered(&state, id).await?;
    Ok(StatusCode::OK)
}

/// POST /api/v2/collections/{id}/optimize — force-flush, rebuild the ANN graph,
/// publish a checkpoint, and prune the WAL.
///
/// Operationally this is how an operator (or a test) forces a checkpoint
/// without waiting for `sync_threshold` writes, which is also the only way to
/// exercise the publication path over HTTP.
pub(crate) async fn optimize_collection(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "collection.optimize",
        &id,
    )?;
    // `optimize` compacts segments and rebuilds the index — measured at 6.5 s
    // for 40k records. Running that on an async worker thread would starve
    // Raft heartbeats for the whole duration and force an election.
    let engine = state.engine.clone();
    let stats = offload(move || engine.optimize(&id)).await?;
    Ok(Json(serde_json::json!({
        "records": stats.records,
        "indexed": stats.indexed,
        "wal_bytes_before": stats.wal_bytes_before,
        "wal_bytes_after": stats.wal_bytes_after,
        "flush_us": stats.flush_us,
        "rebuild_us": stats.rebuild_us,
        "checkpoint": {
            "records_us": stats.checkpoint.records_us,
            "index_us": stats.checkpoint.index_us,
            "vectors_us": stats.checkpoint.vectors_us,
            "index_map_us": stats.checkpoint.index_map_us,
            "manifest_us": stats.checkpoint.manifest_us,
            "validate_us": stats.checkpoint.validate_us,
            "publish_us": stats.checkpoint.publish_us,
            "cleanup_us": stats.checkpoint.cleanup_us,
            "total_us": stats.checkpoint.total_us,
        },
    })))
}
