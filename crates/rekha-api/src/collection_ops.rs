use crate::prelude::*;
use crate::*;

/// Create a collection and make it a cluster fact.
///
/// The log decides. In cluster mode the collection is proposed through Raft and
/// then materialized by the state machine on every node, including the leader;
/// How many collections are already placed on each node, read from the catalog.
///
/// This is the rebuild path, not the per-creation path: [`AppState::placement_cache`]
/// holds the incrementally-maintained answer and calls this only when the
/// leadership term changes, because a previous leader's placements are the only
/// ones in the catalog that this cache did not see happen.
pub(crate) fn load_from_catalog(
    state: &AppState,
    placement: &rekha_cluster::Placement,
) -> Result<std::collections::HashMap<u64, u32>, EngineError> {
    let mut load: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
    for record in state.engine.catalog().list_collections_all()? {
        let Some(shard) = record.shard else {
            // An unsharded row in a sharded cluster: nothing known holds it, so
            // it contributes no load rather than a guessed one.
            continue;
        };
        for &node in placement.owners(shard) {
            *load.entry(node).or_default() += 1;
        }
    }
    Ok(load)
}

/// Decide where `config` goes and replicate the decision.
///
/// The leader decides because only it can see how loaded the nodes already are.
/// Recording the voter set instead made every node a replica, which put the data
/// back on every disk — the exact thing sharding removes — while the write path
/// had already started routing by shard, so the two disagreed about where the
/// data was.
pub(crate) async fn propose_with_placement(
    state: &AppState,
    raft: &Arc<openraft::Raft<rekha_cluster::RaftTypeConfig>>,
    shards: &Arc<rekha_cluster::ShardCluster>,
    config: &CollectionConfig,
) -> Result<(), EngineError> {
    let placement = shards.placement();

    // Held across the propose below on purpose: two creations choosing from the
    // same load picture would pick the same shard and land twice on one node.
    // Collection creation already serialises at the metadata leader, so this
    // costs nothing that was not already serialised.
    let mut cache = state.placement_cache.lock().await;
    let term = raft.metrics().borrow().current_term;
    let mut load = match cache.take() {
        Some((cached_term, load)) if cached_term == term => load,
        // A new term means a previous leader placed collections this process did
        // not witness. One rebuild per election, not one per creation.
        _ => load_from_catalog(state, &placement)?,
    };
    let shard = placement.place(config.id, &load);

    raft.client_write(ClusterOperation::AddCollection {
        config: config.clone(),
        shard: Some(shard),
    })
    .await
    .map_err(|e| EngineError::Recovery(format!("failed to replicate collection creation: {e}")))?;

    // Only after the entry commits: a rolled-back proposal must not leave load
    // counted for a collection that does not exist.
    for &owner in placement.owners(shard) {
        *load.entry(owner).or_default() += 1;
    }
    *cache = Some((term, load));
    Ok(())
}

/// Create a collection and make it a cluster fact.
///
/// The log decides. In cluster mode the collection is proposed through Raft and
/// then materialized by the state machine on every node, including the leader;
/// the leader does not create it locally first, because a create that is
/// interrupted before the log commits would leave one node holding a collection
/// the cluster has never heard of.
pub(crate) async fn create_collection_clustered(
    state: &AppState,
    config: &CollectionConfig,
    partitioned: bool,
) -> Result<rekha_storage::CollectionRecord, EngineError> {
    // Friendly early duplicate detection so the common case is a 409 without a
    // log round trip. The authoritative check is the catalog's own uniqueness
    // constraint inside the state machine.
    if state
        .engine
        .get_collection(&config.tenant, &config.database, &config.name)?
        .is_some()
    {
        return Err(EngineError::Catalog(
            rekha_storage::CatalogError::AlreadyExists,
        ));
    }

    let Some(raft) = state.raft.clone() else {
        let engine = state.engine.clone();
        let config = config.clone();
        return offload(move || engine.create_collection(&config)).await;
    };

    match state.shards.clone() {
        Some(shards) => {
            if partitioned {
                raft.client_write(ClusterOperation::AddCollection {
                    config: config.clone(),
                    shard: None,
                })
                .await
                .map_err(|e| {
                    EngineError::Recovery(format!("failed to replicate collection creation: {e}"))
                })?;
            } else {
                propose_with_placement(state, &raft, &shards, config).await?;
            }
        }
        // One group holds everything, and a single node has no shards at all.
        None => {
            raft.client_write(ClusterOperation::AddCollection {
                config: config.clone(),
                shard: None,
            })
            .await
            .map_err(|e| {
                EngineError::Recovery(format!("failed to replicate collection creation: {e}"))
            })?;
        }
    }

    let engine = state.engine.clone();
    let id = config.id;
    offload(move || engine.get_collection_by_id(&id))
        .await?
        .ok_or_else(|| {
            EngineError::Recovery(format!(
                "collection {id} committed to the log but was not materialized locally"
            ))
        })
}

/// Delete a collection through the log.
///
/// The removal is proposed before anything is deleted locally, so a failed
/// proposal leaves the collection intact everywhere rather than removing it
/// from one node.
pub(crate) async fn delete_collection_clustered(
    state: &AppState,
    collection_id: Uuid,
) -> Result<(), EngineError> {
    let Some(raft) = state.raft.clone() else {
        let engine = state.engine.clone();
        return offload(move || engine.delete_collection(&collection_id)).await;
    };
    raft.client_write(ClusterOperation::RemoveCollection { collection_id })
        .await
        .map_err(|e| {
            EngineError::Recovery(format!("failed to replicate collection deletion: {e}"))
        })?;
    Ok(())
}

/// Prepare a write on the leader and get it committed.
///
/// Two paths, one meaning:
///
/// * Single-node — append through the WAL and apply, exactly as before.
/// * Cluster — validate locally, then propose one entry. The state machine
///   applies it on every node, so the log index is the dataset sequence and
///   there is one replication mechanism instead of two.
///
/// The per-collection lock is held across prepare and propose so a metadata
/// patch cannot be resolved from a base that another write is about to replace.
pub(crate) async fn write_batch<F>(
    state: &AppState,
    collection_id: Uuid,
    prepare: F,
) -> Result<usize, ApiError>
where
    F: FnOnce() -> Result<rekha_engine::PreparedBatch, EngineError> + Send + 'static,
{
    // The guard is owned so it can be held across awaits without borrowing the
    // state borrow that produced it.
    let shard = state.shard_of(&collection_id);
    let lock = state.write_lock(collection_id, shard);
    let _guard = lock.lock_owned().await;

    let prepared = offload(prepare).await?;
    let ops = prepared.ops.len();
    commit_batch(state, collection_id, prepared).await?;
    Ok(ops)
}

pub(crate) async fn commit_batch_for_shard(
    state: &AppState,
    shard: rekha_cluster::ShardId,
    collection_id: Uuid,
    prepared: rekha_engine::PreparedBatch,
) -> Result<(), ApiError> {
    if prepared.ops.is_empty() {
        return Ok(());
    }

    if !state.data_via_log {
        let engine = state.engine.clone();
        offload(move || engine.append_ops_partition(&collection_id, Some(shard), &prepared.ops))
            .await?;
        return Ok(());
    }

    let Some(shards) = state.shards.as_ref() else {
        return Err(ApiError::Engine(EngineError::Unavailable(
            "no shards configured".into(),
        )));
    };
    let Some(raft) = shards.raft(shard) else {
        return Err(ApiError::Engine(EngineError::NotReplica {
            collection: collection_id.to_string(),
            shard: Some(shard),
        }));
    };
    let record = state
        .engine
        .get_collection_by_id(&collection_id)?
        .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;

    if !shards.is_ensured(collection_id, shard) {
        raft.client_write(ClusterOperation::AddCollection {
            config: record.config,
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

    let operation = ClusterOperation::Batch {
        collection_id,
        ops: prepared.ops,
        dimension: prepared.dimension,
    };

    match raft.client_write(operation).await {
        Ok(response) => {
            match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                Ok(applied) => match applied.error {
                    None => Ok(()),
                    Some(error) => Err(ApiError::Engine(EngineError::Recovery(format!(
                        "the write committed at log index {} but this node could not apply it: \
                         {error}",
                        applied.seq
                    )))),
                },
                Err(e) => {
                    tracing::warn!(
                        %e,
                        "unreadable apply response; the entry committed but its outcome is unknown"
                    );
                    Ok(())
                }
            }
        }
        Err(openraft::error::RaftError::APIError(
            openraft::error::ClientWriteError::ForwardToLeader(forward),
        )) => Err(ApiError::NotLeader {
            leader: forward.leader_id.or(forward.leader_node),
        }),
        Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
            "the write could not be committed to the log: {e}"
        )))),
    }
}

pub(crate) async fn commit_batch(
    state: &AppState,
    collection_id: Uuid,
    prepared: rekha_engine::PreparedBatch,
) -> Result<(), ApiError> {
    if !state.data_via_log {
        let engine = state.engine.clone();
        offload(move || engine.append_ops(&collection_id, &prepared.ops)).await?;
        return Ok(());
    }

    let shard = match state.shards.as_ref() {
        Some(_) => {
            let record = state
                .engine
                .get_collection_by_id(&collection_id)?
                .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;
            record.shard
        }
        None => None,
    };

    match (state.shards.as_ref(), shard) {
        (Some(_), Some(shard)) => {
            commit_batch_for_shard(state, shard, collection_id, prepared).await
        }
        (Some(_), None) => {
            // Partitioned collection committed through single batch fallback
            commit_batch_for_shard(state, rekha_cluster::ShardId(0), collection_id, prepared).await
        }
        (None, _) => {
            let raft = state.raft.clone().ok_or_else(|| {
                EngineError::Recovery(
                    "this node is configured to write through the log but has no Raft group".into(),
                )
            })?;
            let operation = ClusterOperation::Batch {
                collection_id,
                ops: prepared.ops,
                dimension: prepared.dimension,
            };
            match raft.client_write(operation).await {
                Ok(response) => {
                    match serde_json::from_str::<rekha_cluster::BatchResponse>(&response.data) {
                        Ok(applied) => match applied.error {
                            None => Ok(()),
                            Some(error) => Err(ApiError::Engine(EngineError::Recovery(format!(
                                "the write committed at log index {} but this node could not apply it: \
                                 {error}",
                                applied.seq
                            )))),
                        },
                        Err(e) => {
                            tracing::warn!(
                                %e,
                                "unreadable apply response; the entry committed but its outcome is unknown"
                            );
                            Ok(())
                        }
                    }
                }
                Err(openraft::error::RaftError::APIError(
                    openraft::error::ClientWriteError::ForwardToLeader(forward),
                )) => Err(ApiError::NotLeader {
                    leader: forward.leader_id.or(forward.leader_node),
                }),
                Err(e) => Err(ApiError::Engine(EngineError::Recovery(format!(
                    "the write could not be committed to the log: {e}"
                )))),
            }
        }
    }
}
