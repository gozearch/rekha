//! ChromaDB Python client compatibility layer.
//!
//! Provides `/heartbeat`, `/version`, `/pre-flight-checks` endpoints and
//! rewrites ChromaDB v2 paths to RekhaDB engine calls.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use rekha_core::config::CollectionConfig;
use rekha_core::filter::WhereFilter;
use rekha_core::types::{Distance, Embedding, Metadata};
use rekha_engine::EngineError;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{AppState, engine_error_status, validation_err};

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HeartbeatResponse {
    #[serde(rename = "nanosecond heartbeat")]
    nanosecond_heartbeat: u64,
}

#[derive(Serialize)]
pub struct PreFlightResponse {
    max_batch_size: usize,
    supports_base64_encoding: bool,
}

#[derive(Serialize)]
pub struct ChromaCollectionResponse {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub collection_type: String,
    pub dimension: Option<usize>,
    pub metadata: Option<serde_json::Value>,
    pub tenant: String,
    pub database: String,
    #[serde(rename = "configuration_json")]
    pub configuration_json: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct ChromaQueryResponse {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Vec<Option<serde_json::Value>>>>,
    pub documents: Option<Vec<Vec<Option<String>>>>,
}

#[derive(Serialize)]
pub struct ChromaGetResponse {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Option<serde_json::Value>>>,
    pub documents: Option<Vec<Option<String>>>,
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// ChromaDB-compatible metadata that accepts flat JSON values (`"str"`, `123`,
/// `true`) and converts them to our tagged `MetadataValue` format.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum ChromaMetadataValue {
    Str(String),
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl From<ChromaMetadataValue> for rekha_core::types::MetadataValue {
    fn from(v: ChromaMetadataValue) -> Self {
        match v {
            ChromaMetadataValue::Str(s) => rekha_core::types::MetadataValue::Str(s),
            ChromaMetadataValue::Bool(b) => rekha_core::types::MetadataValue::Bool(b),
            ChromaMetadataValue::Int(i) => rekha_core::types::MetadataValue::Int(i),
            ChromaMetadataValue::Float(f) => rekha_core::types::MetadataValue::Float(f),
        }
    }
}

pub type ChromaMetadata = std::collections::HashMap<String, ChromaMetadataValue>;

fn convert_metadata(m: Option<ChromaMetadata>) -> Option<Metadata> {
    m.map(|m| m.into_iter().map(|(k, v)| (k, v.into())).collect())
}

fn flatten_metadata(meta: &Option<Metadata>) -> Option<serde_json::Value> {
    meta.as_ref().map(|m| {
        let flat: serde_json::Map<String, serde_json::Value> = m
            .iter()
            .map(|(k, v)| {
                let val = match v {
                    rekha_core::types::MetadataValue::Str(s) => {
                        serde_json::Value::String(s.clone())
                    }
                    rekha_core::types::MetadataValue::Bool(b) => serde_json::Value::Bool(*b),
                    rekha_core::types::MetadataValue::Int(i) => serde_json::json!(i),
                    rekha_core::types::MetadataValue::Float(f) => serde_json::json!(f),
                };
                (k.clone(), val)
            })
            .collect();
        serde_json::Value::Object(flat)
    })
}

#[derive(Deserialize)]
pub struct ChromaCreateCollectionRequest {
    pub name: String,
    pub metadata: Option<ChromaMetadata>,
    pub dimension: Option<usize>,
    #[serde(default)]
    pub get_or_create: bool,
}

#[derive(Deserialize)]
pub struct ChromaAddRequest {
    pub ids: Vec<String>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<ChromaMetadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct ChromaQueryRequest {
    pub query_embeddings: Vec<Vec<f32>>,
    #[serde(default = "default_n_results")]
    pub n_results: usize,
    #[serde(rename = "where")]
    pub where_filter: Option<WhereFilter>,
    #[serde(default)]
    pub include: Vec<String>,
}

fn default_n_results() -> usize {
    10
}

#[derive(Deserialize)]
pub struct ChromaGetRequest {
    pub ids: Option<Vec<String>>,
    #[serde(rename = "where")]
    pub where_filter: Option<WhereFilter>,
    #[serde(default)]
    pub include: Vec<String>,
}

#[derive(Deserialize)]
pub struct ChromaDeleteRequest {
    pub ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_collection(
    engine: &rekha_engine::Engine,
    tenant: &str,
    database: &str,
    name_or_id: &str,
) -> Result<Uuid, EngineError> {
    if let Ok(uuid) = Uuid::parse_str(name_or_id) {
        return engine
            .get_collection_by_id(&uuid)?
            .map(|r| r.config.id)
            .ok_or_else(|| EngineError::CollectionNotFound(name_or_id.to_string()));
    }
    engine
        .get_collection(tenant, database, name_or_id)?
        .map(|r| r.config.id)
        .ok_or_else(|| EngineError::CollectionNotFound(name_or_id.to_string()))
}

fn chroma_response(record: &rekha_storage::CollectionRecord) -> ChromaCollectionResponse {
    ChromaCollectionResponse {
        id: record.config.id.to_string(),
        name: record.config.name.clone(),
        collection_type: "collection".into(),
        dimension: if record.config.dimension == 0 {
            None
        } else {
            Some(record.config.dimension)
        },
        metadata: flatten_metadata(&record.config.metadata),
        tenant: record.config.tenant.clone(),
        database: record.config.database.clone(),
        configuration_json: Some(serde_json::json!({})),
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

pub(crate) struct ChromaError(pub(crate) EngineError);

impl IntoResponse for ChromaError {
    fn into_response(self) -> axum::response::Response {
        let status = engine_error_status(&self.0);
        let message = self.0.to_string();
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

impl From<EngineError> for ChromaError {
    fn from(e: EngineError) -> Self {
        ChromaError(e)
    }
}

// ---------------------------------------------------------------------------
// Handlers — Meta endpoints
// ---------------------------------------------------------------------------

/// GET /heartbeat — ChromaDB health check
pub(crate) async fn heartbeat() -> Json<HeartbeatResponse> {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    Json(HeartbeatResponse {
        nanosecond_heartbeat: ns,
    })
}

/// GET /version — Server version
pub(crate) async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!("0.1.0"))
}

/// GET /pre-flight-checks — Pre-flight validation
pub(crate) async fn pre_flight_checks() -> Json<PreFlightResponse> {
    Json(PreFlightResponse {
        max_batch_size: 1000,
        supports_base64_encoding: false,
    })
}

// ---------------------------------------------------------------------------
// Handlers — Collection CRUD
// ---------------------------------------------------------------------------

/// POST /api/v1/tenants/{t}/databases/{d}/collections — Create or get collection
pub(crate) async fn create_collection_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database)): Path<(String, String)>,
    Json(req): Json<ChromaCreateCollectionRequest>,
) -> Result<(StatusCode, Json<ChromaCollectionResponse>), ChromaError> {
    let dimension = req.dimension.unwrap_or(0);
    crate::validate_name(&req.name).map_err(ChromaError)?;
    if dimension > crate::MAX_DIMENSION {
        return Err(ChromaError(validation_err(format!(
            "dimension {} exceeds max {}",
            dimension,
            crate::MAX_DIMENSION
        ))));
    }

    if req.get_or_create
        && let Some(record) = state.engine.get_collection(&tenant, &database, &req.name)?
    {
        return Ok((StatusCode::OK, Json(chroma_response(&record))));
    }

    let mut config = CollectionConfig::new(req.name, dimension, Distance::L2);
    config.tenant = tenant;
    config.database = database;
    config.metadata = convert_metadata(req.metadata);

    let record = state.engine.create_collection(&config)?;
    Ok((StatusCode::CREATED, Json(chroma_response(&record))))
}

/// GET /api/v1/tenants/{t}/databases/{d}/collections — List collections
pub(crate) async fn list_collections_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database)): Path<(String, String)>,
) -> Result<Json<Vec<ChromaCollectionResponse>>, ChromaError> {
    let records = state.engine.list_collections(&tenant, &database)?;
    let collections = records.iter().map(chroma_response).collect();
    Ok(Json(collections))
}

/// GET /api/v1/tenants/{t}/databases/{d}/collections/{name} — Get collection
pub(crate) async fn get_collection_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
) -> Result<Json<ChromaCollectionResponse>, ChromaError> {
    let record = state
        .engine
        .get_collection(&tenant, &database, &name)?
        .ok_or_else(|| EngineError::CollectionNotFound(name))?;
    Ok(Json(chroma_response(&record)))
}

/// DELETE /api/v1/tenants/{t}/databases/{d}/collections/{name} — Delete collection
pub(crate) async fn delete_collection_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
) -> Result<Json<ChromaCollectionResponse>, ChromaError> {
    let record = state
        .engine
        .get_collection(&tenant, &database, &name)?
        .ok_or_else(|| EngineError::CollectionNotFound(name.clone()))?;
    state.engine.delete_collection(&record.config.id)?;

    if let Some(ref raft) = state.raft {
        let op = rekha_cluster::ClusterOperation::RemoveCollection {
            collection_id: record.config.id,
        };
        if let Err(e) = raft.client_write(op).await {
            tracing::error!("Raft client_write failed for RemoveCollection: {e}");
            return Err(ChromaError(validation_err(format!(
                "failed to replicate collection deletion: {e}"
            ))));
        }
    }

    Ok(Json(chroma_response(&record)))
}

// ---------------------------------------------------------------------------
// Handlers — Record operations
// ---------------------------------------------------------------------------

fn validate_chroma_add(req: &ChromaAddRequest) -> Result<(), EngineError> {
    if req.ids.is_empty() {
        return Err(validation_err("ids must not be empty"));
    }
    if req.ids.len() > crate::MAX_BATCH_SIZE {
        return Err(validation_err(format!(
            "batch size {} exceeds max {}",
            req.ids.len(),
            crate::MAX_BATCH_SIZE
        )));
    }
    if req.ids.len() != req.embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            req.embeddings.len()
        )));
    }
    for id in &req.ids {
        if id.is_empty() || id.len() > crate::MAX_ID_LENGTH {
            return Err(validation_err(format!("invalid id length: {id}")));
        }
    }
    for emb in &req.embeddings {
        if emb.is_empty() || emb.len() > crate::MAX_DIMENSION {
            return Err(validation_err(format!(
                "embedding dimension {} invalid (max {})",
                emb.len(),
                crate::MAX_DIMENSION
            )));
        }
    }
    if let Some(ref v) = req.metadatas
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "metadatas len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    if let Some(ref v) = req.documents
        && v.len() != req.ids.len()
    {
        return Err(validation_err(format!(
            "documents len {} != ids len {}",
            v.len(),
            req.ids.len()
        )));
    }
    Ok(())
}

/// POST .../collections/{name}/add
pub(crate) async fn add_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaAddRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    validate_chroma_add(&req).map_err(ChromaError)?;
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let metadatas: Option<Vec<Option<Metadata>>> = req.metadatas.map(|m| {
        m.into_iter()
            .map(|o| o.map(|m| m.into_iter().map(|(k, v)| (k, v.into())).collect()))
            .collect()
    });
    state.engine.add(
        &id,
        &req.ids,
        &embeddings,
        metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/upsert
pub(crate) async fn upsert_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaAddRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    validate_chroma_add(&req).map_err(ChromaError)?;
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let embeddings: Vec<Embedding> = req.embeddings.into_iter().map(|e| e.into()).collect();
    let metadatas: Option<Vec<Option<Metadata>>> = req.metadatas.map(|m| {
        m.into_iter()
            .map(|o| o.map(|m| m.into_iter().map(|(k, v)| (k, v.into())).collect()))
            .collect()
    });
    state.engine.upsert(
        &id,
        &req.ids,
        &embeddings,
        metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/update
pub(crate) async fn update_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaAddRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    if req.ids.is_empty() {
        return Err(ChromaError(validation_err("ids must not be empty")));
    }
    if req.ids.len() > crate::MAX_BATCH_SIZE {
        return Err(ChromaError(validation_err(format!(
            "batch size {} exceeds max {}",
            req.ids.len(),
            crate::MAX_BATCH_SIZE
        ))));
    }
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let metadatas: Option<Vec<Option<Metadata>>> = req.metadatas.map(|m| {
        m.into_iter()
            .map(|o| o.map(|m| m.into_iter().map(|(k, v)| (k, v.into())).collect()))
            .collect()
    });
    state.engine.update(
        &id,
        &req.ids,
        metadatas.as_deref(),
        req.documents.as_deref(),
    )?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/delete
pub(crate) async fn delete_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaDeleteRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    if req.ids.is_empty() {
        return Err(ChromaError(validation_err("ids must not be empty")));
    }
    if req.ids.len() > crate::MAX_BATCH_SIZE {
        return Err(ChromaError(validation_err(format!(
            "batch size {} exceeds max {}",
            req.ids.len(),
            crate::MAX_BATCH_SIZE
        ))));
    }
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    state.engine.delete(&id, &req.ids)?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/query
pub(crate) async fn query_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaQueryRequest>,
) -> Result<Json<ChromaQueryResponse>, ChromaError> {
    if req.query_embeddings.is_empty() {
        return Err(ChromaError(validation_err(
            "query_embeddings must not be empty",
        )));
    }
    if req.query_embeddings.len() > crate::MAX_QUERY_EMBEDDINGS {
        return Err(ChromaError(validation_err(format!(
            "query_embeddings len {} exceeds max {}",
            req.query_embeddings.len(),
            crate::MAX_QUERY_EMBEDDINGS
        ))));
    }
    if req.n_results == 0 || req.n_results > crate::MAX_N_RESULTS {
        return Err(ChromaError(validation_err(format!(
            "n_results {} out of range 1..={}",
            req.n_results,
            crate::MAX_N_RESULTS
        ))));
    }
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;

    let mut all_results = Vec::new();
    for query_emb in &req.query_embeddings {
        let embedding: Embedding = query_emb.clone().into();
        let opts = rekha_engine::QueryOptions {
            ef: 0,
            where_filter: req.where_filter.clone(),
            oversampling: 4,
        };
        let results = state.engine.query(&id, &embedding, req.n_results, &opts)?;
        all_results.push(results);
    }

    let include_metadatas = req.include.iter().any(|s| s == "metadatas");
    let include_documents = req.include.iter().any(|s| s == "documents");
    let include_distances = req.include.iter().any(|s| s == "distances");

    let mut ids = Vec::new();
    let mut distances = if include_distances {
        Some(Vec::new())
    } else {
        None
    };
    let mut metadatas = if include_metadatas {
        Some(Vec::new())
    } else {
        None
    };
    let mut documents = if include_documents {
        Some(Vec::new())
    } else {
        None
    };

    for results in &all_results {
        ids.push(results.iter().map(|r| r.id.clone()).collect());
        if let Some(ref mut d) = distances {
            d.push(results.iter().map(|r| r.distance).collect());
        }
        if let Some(ref mut m) = metadatas {
            m.push(
                results
                    .iter()
                    .map(|r| flatten_metadata(&r.metadata))
                    .collect(),
            );
        }
        if let Some(ref mut doc) = documents {
            doc.push(results.iter().map(|r| r.document.clone()).collect());
        }
    }

    Ok(Json(ChromaQueryResponse {
        ids,
        distances,
        metadatas,
        documents,
    }))
}

/// POST .../collections/{name}/get
pub(crate) async fn get_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaGetRequest>,
) -> Result<Json<ChromaGetResponse>, ChromaError> {
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let ids = req.ids.unwrap_or_default();
    let records = state.engine.get(&id, &ids)?;

    let include_embeddings = req.include.iter().any(|s| s == "embeddings");
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
                        .and_then(|r| r.embedding.as_ref())
                        .map(|e| e.iter().copied().collect())
                })
                .collect(),
        )
    } else {
        None
    };
    let out_metadatas: Vec<Option<serde_json::Value>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| flatten_metadata(&r.metadata)))
        .collect();
    let out_documents: Vec<Option<String>> = records
        .iter()
        .map(|r| r.as_ref().and_then(|r| r.document.clone()))
        .collect();

    Ok(Json(ChromaGetResponse {
        ids: out_ids,
        embeddings: out_embeddings,
        metadatas: Some(out_metadatas),
        documents: Some(out_documents),
    }))
}

/// GET .../collections/{name}/count
///
/// Returns a plain integer (the count), matching the ChromaDB API spec.
pub(crate) async fn count_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
) -> Result<Json<u64>, ChromaError> {
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let count = state.engine.count(&id)?;
    Ok(Json(count))
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the ChromaDB-compatible router.
///
/// All routes are fully prefixed with `/api/v2` so they can be merged alongside
/// the public router without a nested prefix conflict.
pub(crate) fn chroma_router() -> Router<Arc<AppState>> {
    let meta = Router::new()
        .route("/api/v2/heartbeat", get(heartbeat))
        .route("/api/v2/version", get(version))
        .route("/api/v2/pre-flight-checks", get(pre_flight_checks));

    let collections = Router::new()
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections",
            post(create_collection_chroma).get(list_collections_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}",
            get(get_collection_chroma).delete(delete_collection_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/add",
            post(add_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/upsert",
            post(upsert_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/update",
            post(update_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/delete",
            post(delete_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/query",
            post(query_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/get",
            post(get_records_chroma),
        )
        .route(
            "/api/v2/tenants/{tenant}/databases/{database}/collections/{name}/count",
            get(count_records_chroma),
        );

    meta.merge(collections)
}
