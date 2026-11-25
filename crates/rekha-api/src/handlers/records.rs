use crate::prelude::*;
use crate::*;

pub(crate) async fn add_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<AddRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.add",
        &id,
    )?;
    validate_add_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchAdd> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    Vec::new(),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            entry.1.push(req.embeddings[i].clone());
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten(),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Add {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let engine = state.engine.clone();
    let ids = req.ids;
    let metadatas = req.metadatas;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_add_partition(
            &id,
            shard,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

pub(crate) async fn upsert_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpsertRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.upsert",
        &id,
    )?;
    validate_upsert_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchUpsert> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    Vec::new(),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            entry.1.push(req.embeddings[i].clone());
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten()
                        .map(rekha_core::types::metadata_patch_from),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Upsert {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let metadatas = req.metadatas.map(|list| {
        list.into_iter()
            .map(|o| o.map(rekha_core::types::metadata_patch_from))
            .collect::<Vec<_>>()
    });
    let engine = state.engine.clone();
    let ids = req.ids;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_upsert_partition(
            &id,
            shard,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

pub(crate) async fn update_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.update",
        &id,
    )?;
    validate_update_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, SubBatchUpdate> = HashMap::new();
        for (i, record_id) in req.ids.iter().enumerate() {
            let shard = placement.shard_for_record(record_id);
            let entry = sub_batches.entry(shard).or_insert_with(|| {
                (
                    Vec::new(),
                    req.embeddings.as_ref().map(|_| Vec::new()),
                    req.metadatas.as_ref().map(|_| Vec::new()),
                    req.documents.as_ref().map(|_| Vec::new()),
                )
            });
            entry.0.push(record_id.clone());
            if let (Some(embs), Some(src_embs)) = (&mut entry.1, &req.embeddings) {
                embs.push(src_embs[i].to_vec());
            }
            if let Some(ref mut metas) = entry.2 {
                metas.push(
                    req.metadatas
                        .as_ref()
                        .and_then(|m| m.get(i).cloned())
                        .flatten()
                        .map(rekha_core::types::metadata_patch_from),
                );
            }
            if let Some(ref mut docs) = entry.3 {
                docs.push(
                    req.documents
                        .as_ref()
                        .and_then(|d| d.get(i).cloned())
                        .flatten(),
                );
            }
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, (ids, embs, metas, docs))| {
                (
                    shard,
                    ShardWriteKind::Update {
                        ids,
                        embeddings: embs,
                        metadatas: metas,
                        documents: docs,
                    },
                )
            })
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let metadatas = req.metadatas.map(|list| {
        list.into_iter()
            .map(|o| o.map(rekha_core::types::metadata_patch_from))
            .collect::<Vec<_>>()
    });
    let engine = state.engine.clone();
    let ids = req.ids;
    let embeddings = req.embeddings;
    let documents = req.documents;
    write_batch(&state, id, move || {
        engine.prepare_update_partition(
            &id,
            shard,
            &ids,
            embeddings.as_deref(),
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok(StatusCode::OK)
}

pub(crate) async fn delete_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<DeleteRequest>,
) -> Result<StatusCode, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.delete",
        &id,
    )?;
    validate_delete_request(&req)?;

    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let placement = shards.placement();
        let mut sub_batches: HashMap<rekha_cluster::ShardId, Vec<Id>> = HashMap::new();
        for record_id in req.ids {
            let shard = placement.shard_for_record(&record_id);
            sub_batches.entry(shard).or_default().push(record_id);
        }
        let batches: Vec<(rekha_cluster::ShardId, ShardWriteKind)> = sub_batches
            .into_iter()
            .map(|(shard, ids)| (shard, ShardWriteKind::Delete { ids }))
            .collect();
        write_partitioned_2pc(&state, id, batches).await?;
        return Ok(StatusCode::OK);
    }

    let shard = state.shard_of(&id);
    let engine = state.engine.clone();
    let ids = req.ids;
    write_batch(&state, id, move || {
        engine.prepare_delete_partition(&id, shard, &ids)
    })
    .await?;
    Ok(StatusCode::OK)
}

pub(crate) async fn query_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<QueryRequest>,
) -> Result<Json<QueryResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.query",
        &id,
    )?;
    validate_query_request(&req)?;
    let include_metadatas = req.include.iter().any(|s| s == "metadatas");
    let include_documents = req.include.iter().any(|s| s == "documents");
    let include_distances = req.include.iter().any(|s| s == "distances");

    let all_results: Vec<Vec<rekha_engine::ScoredPoint>> = if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let shard_count = shards.placement().shard_count();
        let mut futures = Vec::new();
        for s in 0..shard_count {
            let shard = rekha_cluster::ShardId(s);
            let state = state.clone();
            let query_req = ShardQueryRequest {
                collection_id: id,
                query_embeddings: req.query_embeddings.clone(),
                n_results: req.n_results,
                where_filter: req.where_filter.clone(),
                where_document: req.where_document.clone(),
                with_embeddings: false,
            };
            futures.push(async move { dispatch_shard_query(&state, shard, query_req).await });
        }
        let shard_results = futures::future::try_join_all(futures).await?;
        let n_queries = req.query_embeddings.len();
        let mut merged_queries = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let mut hits = Vec::new();
            for shard_res in &shard_results {
                if let Some(shard_hits) = shard_res.get(q) {
                    hits.extend(shard_hits.iter().cloned());
                }
            }
            hits.sort_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.id.cmp(&b.id))
            });
            hits.truncate(req.n_results);
            merged_queries.push(hits);
        }
        merged_queries
    } else {
        let engine = state.engine.clone();
        let query_embeddings = req.query_embeddings.clone();
        let where_filter = req.where_filter.clone();
        let where_document = req.where_document.clone();
        let n_results = req.n_results;
        offload(move || {
            let mut results = Vec::with_capacity(query_embeddings.len());
            for query_emb in &query_embeddings {
                let embedding: Embedding = query_emb.clone().into();
                let opts = QueryOptions {
                    ef: 0,
                    where_filter: where_filter.clone(),
                    where_document: where_document.clone(),
                    oversampling: 4,
                };
                results.push(engine.query(&id, &embedding, n_results, &opts)?);
            }
            Ok(results)
        })
        .await?
    };

    let n_queries = all_results.len();
    let k = all_results.first().map_or(0, |r| r.len());

    let mut ids: Vec<Vec<String>> = Vec::with_capacity(n_queries);
    let mut distances: Option<Vec<Vec<f32>>> = if include_distances {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };
    let mut metadatas: Option<Vec<Vec<Option<Metadata>>>> = if include_metadatas {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };
    let mut documents: Option<Vec<Vec<Option<String>>>> = if include_documents {
        Some(Vec::with_capacity(n_queries))
    } else {
        None
    };

    for results in &all_results {
        ids.push(results.iter().map(|r| r.id.clone()).collect());
        if let Some(ref mut d) = distances {
            d.push(results.iter().map(|r| r.distance).collect());
        }
        if let Some(ref mut m) = metadatas {
            m.push(results.iter().map(|r| r.metadata.clone()).collect());
        }
        if let Some(ref mut doc) = documents {
            doc.push(results.iter().map(|r| r.document.clone()).collect());
        }
    }

    // Pad shorter result sets to k if needed (Chroma returns Vec<Vec<...>>
    // where inner vecs are all length k, with empty strings / None for missing).
    for vec in ids.iter_mut() {
        let len = vec.len();
        if len < k {
            vec.resize_with(k, String::new);
        }
    }
    if let Some(ref mut d) = distances {
        for vec in d.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || f32::INFINITY);
            }
        }
    }
    if let Some(ref mut m) = metadatas {
        for vec in m.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || None);
            }
        }
    }
    if let Some(ref mut doc) = documents {
        for vec in doc.iter_mut() {
            let len = vec.len();
            if len < k {
                vec.resize_with(k, || None);
            }
        }
    }

    Ok(Json(QueryResponse {
        ids,
        distances,
        metadatas,
        documents,
    }))
}

pub(crate) async fn get_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
    Json(req): Json<GetRequest>,
) -> Result<Json<GetResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.get",
        &id,
    )?;
    let ids = req.ids.unwrap_or_default();
    let include_embeddings = req.include.iter().any(|s| s == "embeddings");

    let records: Vec<Option<rekha_engine::Record>> = if is_partitioned(&state, &id) {
        if ids.is_empty() && req.where_filter.is_none() && req.where_document.is_none() {
            Vec::new()
        } else if !ids.is_empty() {
            let Some(shards) = state.shards.as_ref() else {
                return Err(ApiError::Engine(EngineError::Unavailable(
                    "no shards configured".into(),
                )));
            };
            let placement = shards.placement();
            let mut by_shard: HashMap<rekha_cluster::ShardId, Vec<Id>> = HashMap::new();
            for item_id in &ids {
                let shard = placement.shard_for_record(item_id);
                by_shard.entry(shard).or_default().push(item_id.clone());
            }
            let mut futures = Vec::new();
            for (shard, shard_ids) in by_shard {
                let state = state.clone();
                let get_req = ShardGetRequest {
                    collection_id: id,
                    ids: Some(shard_ids),
                    where_filter: req.where_filter.clone(),
                    where_document: req.where_document.clone(),
                    limit: None,
                    offset: 0,
                };
                futures.push(async move { dispatch_shard_get(&state, shard, get_req).await });
            }
            let shard_records = futures::future::try_join_all(futures).await?;
            let mut record_map: HashMap<Id, rekha_engine::Record> = HashMap::new();
            for list in shard_records {
                for r in list {
                    record_map.insert(r.id.clone(), r);
                }
            }
            ids.iter()
                .map(|item_id| record_map.remove(item_id))
                .collect()
        } else {
            let Some(shards) = state.shards.as_ref() else {
                return Err(ApiError::Engine(EngineError::Unavailable(
                    "no shards configured".into(),
                )));
            };
            let shard_count = shards.placement().shard_count();
            let mut futures = Vec::new();
            for s in 0..shard_count {
                let shard = rekha_cluster::ShardId(s);
                let state = state.clone();
                let get_req = ShardGetRequest {
                    collection_id: id,
                    ids: None,
                    where_filter: req.where_filter.clone(),
                    where_document: req.where_document.clone(),
                    limit: None,
                    offset: 0,
                };
                futures.push(async move { dispatch_shard_get(&state, shard, get_req).await });
            }
            let shard_records = futures::future::try_join_all(futures).await?;
            let mut all_recs = Vec::new();
            for list in shard_records {
                all_recs.extend(list);
            }
            all_recs.sort_by_key(|r| r.seq);
            all_recs.into_iter().map(Some).collect()
        }
    } else if ids.is_empty() && req.where_filter.is_none() && req.where_document.is_none() {
        Vec::new()
    } else if req.where_filter.is_none() && req.where_document.is_none() {
        state.engine.get(&id, &ids)?
    } else {
        state
            .engine
            .get_records(
                &id,
                (!ids.is_empty()).then_some(ids.as_slice()),
                req.where_filter.as_ref(),
                req.where_document.as_ref(),
                None,
                0,
            )?
            .into_iter()
            .map(Some)
            .collect()
    };

    let out_ids: Vec<String> = records
        .iter()
        .filter_map(|r| r.as_ref().map(|r| r.id.clone()))
        .collect();
    let out_embeddings = if include_embeddings {
        Some(
            records
                .iter()
                .filter_map(|r| {
                    r.as_ref()
                        .and_then(|r| r.embedding.as_ref().map(|e| e.iter().copied().collect()))
                })
                .collect(),
        )
    } else {
        None
    };
    let out_metadatas: Vec<Option<Metadata>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| r.metadata.clone()))
        .collect();
    let out_documents: Vec<Option<String>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| r.document.clone()))
        .collect();

    Ok(Json(GetResponse {
        ids: out_ids,
        embeddings: out_embeddings,
        metadatas: Some(out_metadatas),
        documents: Some(out_documents),
    }))
}

pub(crate) async fn count_records(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<middleware::AuthContext>,
    Path(id): Path<Uuid>,
) -> Result<Json<CountResponse>, ApiError> {
    check_collection_access(
        state.engine.catalog().as_ref(),
        &auth.permissions,
        "record.count",
        &id,
    )?;
    if is_partitioned(&state, &id) {
        let Some(shards) = state.shards.as_ref() else {
            return Err(ApiError::Engine(EngineError::Unavailable(
                "no shards configured".into(),
            )));
        };
        let shard_count = shards.placement().shard_count();
        let mut futures = Vec::new();
        for s in 0..shard_count {
            let shard = rekha_cluster::ShardId(s);
            let state = state.clone();
            futures.push(async move { dispatch_shard_count(&state, shard, id).await });
        }
        let counts: Vec<u64> = futures::future::try_join_all(futures).await?;
        let total: u64 = counts.into_iter().sum();
        return Ok(Json(CountResponse { count: total }));
    }
    let count = state.engine.count(&id)?;
    Ok(Json(CountResponse { count }))
}
