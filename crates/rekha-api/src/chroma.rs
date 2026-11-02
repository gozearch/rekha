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
use rekha_core::filter::{WhereDocument, WhereFilter};
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
    pub embeddings: Option<Vec<Vec<Vec<f32>>>>,
    pub uris: Option<Vec<Vec<Option<String>>>>,
    /// The include list actually applied, echoed like Chroma does.
    pub include: Vec<String>,
}

#[derive(Serialize)]
pub struct ChromaGetResponse {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Option<serde_json::Value>>>,
    pub documents: Option<Vec<Option<String>>>,
    pub uris: Option<Vec<Option<String>>>,
    /// The include list actually applied, echoed like Chroma does.
    pub include: Vec<String>,
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
    /// A JSON `null`. Chroma reads this as "remove this key" on update/upsert;
    /// it is never stored as a value.
    Null,
}

impl ChromaMetadataValue {
    /// The storable value, or `None` for a JSON null (which deletes a key).
    fn into_patch_value(self) -> Option<rekha_core::types::MetadataValue> {
        match self {
            ChromaMetadataValue::Str(s) => Some(rekha_core::types::MetadataValue::Str(s)),
            ChromaMetadataValue::Bool(b) => Some(rekha_core::types::MetadataValue::Bool(b)),
            ChromaMetadataValue::Int(i) => Some(rekha_core::types::MetadataValue::Int(i)),
            ChromaMetadataValue::Float(f) => Some(rekha_core::types::MetadataValue::Float(f)),
            ChromaMetadataValue::Null => None,
        }
    }
}

pub type ChromaMetadata = std::collections::HashMap<String, ChromaMetadataValue>;

/// Convert metadata for a create/add: nulls are dropped (never stored).
fn convert_metadata(m: Option<ChromaMetadata>) -> Option<Metadata> {
    let map: Metadata = m?
        .into_iter()
        .filter_map(|(k, v)| v.into_patch_value().map(|value| (k, value)))
        .collect();
    (!map.is_empty()).then_some(map)
}

/// Convert metadata for an update/upsert, where a null means "remove key".
fn convert_metadata_patch(m: Option<ChromaMetadata>) -> Option<rekha_engine::MetadataPatch> {
    Some(
        m?.into_iter()
            .map(|(k, v)| (k, v.into_patch_value()))
            .collect(),
    )
}

/// Convert a per-record metadata list into a patch list.
fn convert_metadata_patches(
    m: Option<Vec<Option<ChromaMetadata>>>,
) -> Option<Vec<Option<rekha_engine::MetadataPatch>>> {
    m.map(|list| list.into_iter().map(convert_metadata_patch).collect())
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
    /// Chroma v2 `configuration` object, e.g. `{"hnsw": {"space": "cosine"}}`.
    /// Takes precedence over the legacy `metadata["hnsw:space"]` key.
    #[serde(default)]
    pub configuration: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct ChromaAddRequest {
    pub ids: Vec<String>,
    /// Optional because `update` legitimately sends `embeddings: null`
    /// (metadata-only edits). `add`/`upsert` require it and reject `null`.
    #[serde(default)]
    pub embeddings: Option<Vec<Vec<f32>>>,
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
    /// Chroma's `where_document`: eligibility by document text.
    ///
    /// Deserialising refuses anything but `$contains` / `$not_contains`, so an
    /// unsupported operator is a `400` rather than a filter that silently does
    /// nothing and returns the unfiltered result set.
    #[serde(default)]
    pub where_document: Option<WhereDocument>,
    /// `None` means the field was **omitted** (use the endpoint defaults);
    /// `Some([])` means the caller explicitly asked for ids only. Collapsing
    /// the two returned the defaults for `include: []`.
    #[serde(default, deserialize_with = "deserialize_include")]
    pub include: Option<Vec<String>>,
}

/// Reject `include: null`, which Chroma answers with 422. An omitted field is
/// left to `#[serde(default)]`, so only an explicit null reaches this.
fn deserialize_include<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Null => Err(serde::de::Error::custom(
            "`include` must be an array, not null",
        )),
        other => serde_json::from_value(other)
            .map(Some)
            .map_err(serde::de::Error::custom),
    }
}

/// Resolve a request's `include` against an endpoint's documented defaults.
fn resolve_include(include: Option<&[String]>, defaults: &[&str]) -> Vec<String> {
    match include {
        None => defaults.iter().map(|s| (*s).to_string()).collect(),
        Some(fields) => fields.to_vec(),
    }
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
    pub where_document: Option<WhereDocument>,
    /// See [`ChromaQueryRequest::include`].
    #[serde(default, deserialize_with = "deserialize_include")]
    pub include: Option<Vec<String>>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Deserialize)]
pub struct ChromaDeleteRequest {
    #[serde(default)]
    pub ids: Option<Vec<String>>,
    #[serde(rename = "where")]
    pub where_filter: Option<WhereFilter>,
    #[serde(default)]
    pub where_document: Option<WhereDocument>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a Chroma collection reference to its id, enforcing scope.
///
/// Chroma accepts either a name or a UUID in the `{name}` position. A UUID is
/// globally unique keyed data, so resolving it without checking its owning
/// `(tenant, database)` let any caller reach another tenant's collection just
/// by knowing its id. Both forms therefore must match the requested scope.
///
/// A UUID in the wrong scope is reported as *not found* rather than forbidden:
/// the collection does not exist within this tenant/database, and a distinct
/// error would confirm that a guessed id belongs to someone else.
fn resolve_collection(
    engine: &rekha_engine::Engine,
    tenant: &str,
    database: &str,
    name_or_id: &str,
) -> Result<Uuid, EngineError> {
    let not_found = || EngineError::CollectionNotFound(name_or_id.to_string());
    if let Ok(uuid) = Uuid::parse_str(name_or_id) {
        let record = engine.get_collection_by_id(&uuid)?.ok_or_else(not_found)?;
        let config = &record.config;
        if config.tenant != tenant || config.database != database {
            return Err(not_found());
        }
        return Ok(config.id);
    }
    engine
        .get_collection(tenant, database, name_or_id)?
        .map(|r| r.config.id)
        .ok_or_else(not_found)
}

/// Resolve the collection dimension from a Chroma create request.
///
/// Clients set it either as a top-level `dimension` (legacy/JS) or inside
/// `configuration.hnsw.dimension` (Python >= 1.5 no longer accepts a
/// `dimension=` keyword). Both are honoured; absence means "infer from the
/// first write".
fn resolve_dimension(req: &ChromaCreateCollectionRequest) -> Option<usize> {
    req.dimension.or_else(|| {
        req.configuration
            .as_ref()
            .and_then(|c| c.get("hnsw"))
            .and_then(|h| h.get("dimension"))
            .and_then(|d| d.as_u64())
            .map(|d| d as usize)
    })
}

/// Resolve the distance function from a Chroma create request.
///
/// `configuration.hnsw.space` takes precedence over the legacy
/// `metadata["hnsw:space"]` key (that is the migration path the Python client
/// uses). An unknown value is rejected rather than silently defaulting, because
/// silently switching a `cosine` collection to L2 changes reported distances.
fn resolve_space(req: &ChromaCreateCollectionRequest) -> Result<Distance, EngineError> {
    let from_configuration = req
        .configuration
        .as_ref()
        .and_then(|c| c.get("hnsw"))
        .and_then(|h| h.get("space"))
        .and_then(|s| s.as_str());

    let from_metadata = req.metadata.as_ref().and_then(|m| {
        m.get("hnsw:space").and_then(|v| match v {
            ChromaMetadataValue::Str(s) => Some(s.as_str()),
            _ => None,
        })
    });

    let raw = from_configuration.or(from_metadata);
    match raw {
        None => Ok(Distance::L2),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Distance::L2),
            "cosine" => Ok(Distance::Cosine),
            "ip" | "inner_product" | "innerproduct" => Ok(Distance::Ip),
            other => Err(validation_err(format!(
                "unsupported hnsw:space `{other}` (expected l2, cosine, or ip)"
            ))),
        },
    }
}

fn chroma_response(record: &rekha_storage::CollectionRecord) -> ChromaCollectionResponse {
    let space = match record.config.space {
        Distance::L2 => "l2",
        Distance::Cosine => "cosine",
        Distance::Ip => "ip",
    };
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
        configuration_json: Some(serde_json::json!({ "hnsw": { "space": space } })),
    }
}

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

pub(crate) enum ChromaError {
    Engine(EngineError),
    /// This node may not perform the write; the response names the leader.
    NotLeader {
        leader: Option<u64>,
    },
}

impl IntoResponse for ChromaError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChromaError::Engine(e) => {
                let status = engine_error_status(&e);
                (status, Json(serde_json::json!({ "error": e.to_string() }))).into_response()
            }
            ChromaError::NotLeader { leader } => {
                crate::ApiError::NotLeader { leader }.into_response()
            }
        }
    }
}

impl From<EngineError> for ChromaError {
    fn from(e: EngineError) -> Self {
        ChromaError::Engine(e)
    }
}

impl From<crate::ApiError> for ChromaError {
    fn from(e: crate::ApiError) -> Self {
        match e {
            crate::ApiError::Engine(e) => ChromaError::Engine(e),
            crate::ApiError::NotLeader { leader } => ChromaError::NotLeader { leader },
        }
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
        // Must match the limit the write handlers actually enforce.
        max_batch_size: crate::MAX_BATCH_SIZE,
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
    let dimension = resolve_dimension(&req).unwrap_or(0);
    crate::validate_name(&req.name).map_err(ChromaError::Engine)?;
    if dimension > crate::MAX_DIMENSION {
        return Err(ChromaError::Engine(validation_err(format!(
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

    let space = resolve_space(&req)?;
    let mut config = CollectionConfig::new(req.name, dimension, space);
    config.tenant = tenant;
    config.database = database;
    // Keep the legacy metadata form (`hnsw:space`) out of the stored metadata so
    // it is not reported back as user metadata; the resolved space is returned
    // in `configuration`.
    config.metadata = convert_metadata(req.metadata).map(|mut m| {
        m.remove("hnsw:space");
        m
    });

    let record = crate::create_collection_clustered(&state, &config).await?;
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
    // Accept a name *or* a UUID, both scoped to this tenant/database.
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let record = state
        .engine
        .get_collection_by_id(&id)?
        .ok_or_else(|| EngineError::CollectionNotFound(name))?;
    Ok(Json(chroma_response(&record)))
}

/// DELETE /api/v1/tenants/{t}/databases/{d}/collections/{name} — Delete collection
pub(crate) async fn delete_collection_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
) -> Result<Json<ChromaCollectionResponse>, ChromaError> {
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let record = state
        .engine
        .get_collection_by_id(&id)?
        .ok_or_else(|| EngineError::CollectionNotFound(name.clone()))?;
    crate::delete_collection_clustered(&state, record.config.id).await?;
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
    let embeddings = req
        .embeddings
        .as_ref()
        .ok_or_else(|| validation_err("embeddings are required for add/upsert"))?;
    if req.ids.len() != embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            embeddings.len()
        )));
    }
    for id in &req.ids {
        if id.is_empty() || id.len() > crate::MAX_ID_LENGTH {
            return Err(validation_err(format!("invalid id length: {id}")));
        }
    }
    for emb in embeddings {
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
    validate_chroma_add(&req).map_err(ChromaError::Engine)?;
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let embeddings: Vec<Embedding> = req
        .embeddings
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    // `add` stores metadata as given: it is a create, not a merge.
    let metadatas: Option<Vec<Option<Metadata>>> = req
        .metadatas
        .map(|list| list.into_iter().map(convert_metadata).collect());
    let engine = state.engine.clone();
    let collection = id;
    let ids = req.ids;
    let documents = req.documents;
    crate::write_batch(&state, collection, move || {
        engine.prepare_add(
            &collection,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    // Chroma answers `add` with 201 Created (upsert/update use 200).
    Ok((StatusCode::CREATED, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/upsert
///
/// Chroma **merges** metadata into an existing record and preserves every field
/// the request does not mention, so this passes patches rather than full values.
pub(crate) async fn upsert_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaAddRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    validate_chroma_add(&req).map_err(ChromaError::Engine)?;
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    let embeddings: Vec<Embedding> = req
        .embeddings
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    let metadatas = convert_metadata_patches(req.metadatas);
    let engine = state.engine.clone();
    let collection = id;
    let ids = req.ids;
    let documents = req.documents;
    crate::write_batch(&state, collection, move || {
        engine.prepare_upsert(
            &collection,
            &ids,
            &embeddings,
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/update
pub(crate) async fn update_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaAddRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    if req.ids.is_empty() {
        return Err(ChromaError::Engine(validation_err("ids must not be empty")));
    }
    if req.ids.len() > crate::MAX_BATCH_SIZE {
        return Err(ChromaError::Engine(validation_err(format!(
            "batch size {} exceeds max {}",
            req.ids.len(),
            crate::MAX_BATCH_SIZE
        ))));
    }
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;
    // Chroma sends `embeddings: null` for a metadata-only update, and may send a
    // vector when the caller is updating both.
    let embeddings: Option<Vec<Embedding>> = req
        .embeddings
        .filter(|e| !e.is_empty())
        .map(|e| e.into_iter().map(Into::into).collect());
    let metadatas = convert_metadata_patches(req.metadatas);
    let engine = state.engine.clone();
    let collection = id;
    let ids = req.ids;
    let documents = req.documents;
    crate::write_batch(&state, collection, move || {
        engine.prepare_update(
            &collection,
            &ids,
            embeddings.as_deref(),
            metadatas.as_deref(),
            documents.as_deref(),
        )
    })
    .await?;
    Ok((StatusCode::OK, Json(serde_json::json!({}))))
}

/// POST .../collections/{name}/delete
///
/// Chroma allows `ids`, `where`, or both (intersection). A request with neither
/// is a **no-op** returning `{"deleted": 0}`; it deletes nothing.
pub(crate) async fn delete_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaDeleteRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ChromaError> {
    let has_ids = req.ids.as_ref().is_some_and(|v| !v.is_empty());
    let has_where = req.where_filter.is_some();
    let has_where_document = req.where_document.is_some();
    if let Some(ref ids) = req.ids
        && ids.len() > crate::MAX_BATCH_SIZE
    {
        return Err(ChromaError::Engine(validation_err(format!(
            "batch size {} exceeds max {}",
            ids.len(),
            crate::MAX_BATCH_SIZE
        ))));
    }

    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;

    // A request with neither selector is a no-op in Chroma (`{"deleted": 0}`),
    // not a collection wipe. Rekha used to reject it with 400, which is both a
    // compatibility break and a worse answer than doing nothing.
    if !has_ids && !has_where && !has_where_document {
        return Ok((StatusCode::OK, Json(serde_json::json!({ "deleted": 0 }))));
    }

    let deleted = match (req.ids, req.where_filter, req.where_document) {
        // Exactly one selector, ids only: Chroma reports the number of ids
        // submitted, including ids that did not exist, so the count describes
        // the request rather than stored state.
        (Some(ids), None, None) => {
            let n = ids.len();
            let engine = state.engine.clone();
            crate::write_batch(&state, id, move || engine.prepare_delete(&id, &ids)).await?;
            n
        }
        // Every other shape is a *selection*: ids and/or filters, intersected.
        // Selection and deletion happen inside one collection lock (the
        // `prepare` closure runs under it), so a concurrent write cannot slip
        // between the filter evaluation and the delete. The count is the number
        // of records actually removed.
        (ids, filter, document) => {
            let engine = state.engine.clone();
            crate::write_batch(&state, id, move || {
                engine.prepare_delete_where(&id, ids.as_deref(), filter.as_ref(), document.as_ref())
            })
            .await?
        }
    };

    Ok((
        StatusCode::OK,
        Json(serde_json::json!({ "deleted": deleted })),
    ))
}

/// POST .../collections/{name}/query
pub(crate) async fn query_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaQueryRequest>,
) -> Result<Json<ChromaQueryResponse>, ChromaError> {
    if req.query_embeddings.is_empty() {
        return Err(ChromaError::Engine(validation_err(
            "query_embeddings must not be empty",
        )));
    }
    if req.query_embeddings.len() > crate::MAX_QUERY_EMBEDDINGS {
        return Err(ChromaError::Engine(validation_err(format!(
            "query_embeddings len {} exceeds max {}",
            req.query_embeddings.len(),
            crate::MAX_QUERY_EMBEDDINGS
        ))));
    }
    if req.n_results == 0 || req.n_results > crate::MAX_N_RESULTS {
        return Err(ChromaError::Engine(validation_err(format!(
            "n_results {} out of range 1..={}",
            req.n_results,
            crate::MAX_N_RESULTS
        ))));
    }
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;

    // Resolve the projection *before* searching so unknown fields fail without
    // doing work, and so the engine knows whether vectors are needed.
    let include = resolve_include(
        req.include.as_deref(),
        // Chroma's query defaults: metadatas, documents, distances.
        &["documents", "metadatas", "distances"],
    );
    let include_refs: Vec<&str> = include.iter().map(String::as_str).collect();
    validate_include(&include_refs).map_err(ChromaError::Engine)?;

    let include_metadatas = include_refs.contains(&"metadatas");
    let include_documents = include_refs.contains(&"documents");
    let include_distances = include_refs.contains(&"distances");
    let include_embeddings = include_refs.contains(&"embeddings");
    let include_uris = include_refs.contains(&"uris");

    let query_embeddings: Vec<Embedding> = req
        .query_embeddings
        .iter()
        .cloned()
        .map(Into::into)
        .collect();
    let opts = rekha_engine::QueryOptions {
        ef: 0,
        where_filter: req.where_filter.clone(),
        where_document: req.where_document.clone(),
        oversampling: 4,
    };
    // One call, one collection lock: every query vector and every projected
    // field below comes from the same read view. Fetching embeddings afterwards
    // via `get_records` (the previous approach) could pair a distance from one
    // vector version with an embedding from another.
    //
    // A query is CPU-bound (scan, distance, rank) and runs on the blocking pool
    // so a large query cannot starve Raft heartbeats on the async workers.
    let engine = state.engine.clone();
    let all_results = crate::offload(move || {
        engine.query_many(
            &id,
            &query_embeddings,
            req.n_results,
            &opts,
            include_embeddings,
        )
    })
    .await?;

    let mut ids = Vec::with_capacity(all_results.len());
    let mut distances = include_distances.then(Vec::new);
    let mut metadatas = include_metadatas.then(Vec::new);
    let mut documents = include_documents.then(Vec::new);
    let mut embeddings_out: Option<Vec<Vec<Vec<f32>>>> = include_embeddings.then(Vec::new);
    let mut uris_out: Option<Vec<Vec<Option<String>>>> = include_uris.then(Vec::new);

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
        if let Some(ref mut embs) = embeddings_out {
            embs.push(
                results
                    .iter()
                    .map(|r| {
                        r.embedding
                            .as_ref()
                            .map(|e| e.iter().copied().collect())
                            .unwrap_or_default()
                    })
                    .collect(),
            );
        }
        if let Some(ref mut uris) = uris_out {
            uris.push(results.iter().map(|_| None).collect());
        }
    }

    Ok(Json(ChromaQueryResponse {
        ids,
        distances,
        metadatas,
        documents,
        embeddings: embeddings_out,
        uris: uris_out,
        include,
    }))
}

/// Fields a Chroma client may request via `include`.
const CHROMA_INCLUDE_FIELDS: [&str; 5] =
    ["embeddings", "metadatas", "documents", "distances", "uris"];

/// Reject unknown `include` fields instead of silently returning a subset.
fn validate_include(include: &[&str]) -> Result<(), EngineError> {
    for field in include {
        if !CHROMA_INCLUDE_FIELDS.contains(field) {
            return Err(validation_err(format!(
                "unsupported include field `{field}` (supported: {})",
                CHROMA_INCLUDE_FIELDS.join(", ")
            )));
        }
    }
    Ok(())
}

/// POST .../collections/{name}/get
///
/// Chroma's `get` defaults to `metadatas` + `documents` when `include` is
/// omitted, supports filters without ids, and paginates with `limit`/`offset`.
pub(crate) async fn get_records_chroma(
    State(state): State<Arc<AppState>>,
    Path((tenant, database, name)): Path<(String, String, String)>,
    Json(req): Json<ChromaGetRequest>,
) -> Result<Json<ChromaGetResponse>, ChromaError> {
    let id = resolve_collection(&state.engine, &tenant, &database, &name)?;

    // `get` defaults to metadatas + documents; an explicit `[]` means ids only.
    let include = resolve_include(req.include.as_deref(), &["documents", "metadatas"]);
    let include_refs: Vec<&str> = include.iter().map(String::as_str).collect();
    validate_include(&include_refs).map_err(ChromaError::Engine)?;

    let include_embeddings = include_refs.contains(&"embeddings");
    let include_metadatas = include_refs.contains(&"metadatas");
    let include_documents = include_refs.contains(&"documents");
    let include_uris = include_refs.contains(&"uris");

    let engine = state.engine.clone();
    let ids = req.ids;
    let where_filter = req.where_filter;
    let where_document = req.where_document;
    let limit = req.limit;
    let offset = req.offset.unwrap_or(0);
    // A filtered `get` scans the collection; keep it off the async workers.
    let records = crate::offload(move || {
        engine.get_records(
            &id,
            ids.as_deref(),
            where_filter.as_ref(),
            where_document.as_ref(),
            limit,
            offset,
        )
    })
    .await?;

    Ok(Json(ChromaGetResponse {
        ids: records.iter().map(|r| r.id.clone()).collect(),
        embeddings: include_embeddings.then(|| {
            records
                .iter()
                .map(|r| {
                    r.embedding
                        .as_ref()
                        .map(|e| e.iter().copied().collect())
                        .unwrap_or_default()
                })
                .collect()
        }),
        metadatas: include_metadatas.then(|| {
            records
                .iter()
                .map(|r| flatten_metadata(&r.metadata))
                .collect()
        }),
        documents: include_documents.then(|| records.iter().map(|r| r.document.clone()).collect()),
        // Rekha has no URI concept; Chroma returns null for records without one.
        uris: include_uris.then(|| records.iter().map(|_| None).collect()),
        include,
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
