use crate::prelude::*;

// ---------------------------------------------------------------------------
// Request / Response DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(default = "default_distance")]
    pub distance: Distance,
    #[serde(default)]
    pub partitioned: Option<bool>,
}

pub(crate) fn default_distance() -> Distance {
    Distance::L2
}

#[derive(Serialize)]
pub struct CollectionResponse {
    pub id: Uuid,
    pub name: String,
    pub dimension: usize,
    pub distance: Distance,
    /// Live record count, or `null` when no replica could answer.
    ///
    /// `null` rather than `0`: a list that reports zero for a collection it
    /// could not reach is a wrong answer that looks like a right one.
    pub count: Option<u64>,
    /// The shard this collection was placed on, if the deployment is sharded.
    ///
    /// Exposed because it is the answer to "where is my data", which an
    /// operator otherwise has to reconstruct by hashing — the thing this field
    /// exists to stop being the source of truth.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shard: Option<u32>,
}

#[derive(Deserialize)]
pub struct AddRequest {
    pub ids: Vec<Id>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct UpdateRequest {
    pub ids: Vec<Id>,
    /// Optional: a metadata-only update omits this (or sends null).
    #[serde(default)]
    pub embeddings: Option<Vec<Embedding>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct UpsertRequest {
    pub ids: Vec<Id>,
    pub embeddings: Vec<Vec<f32>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Deserialize)]
pub struct DeleteRequest {
    pub ids: Vec<Id>,
}

#[derive(Deserialize)]
pub struct QueryRequest {
    pub query_embeddings: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    pub n_results: usize,
    pub where_filter: Option<WhereFilter>,
    /// Chroma's document-text eligibility filter. Absent means unfiltered.
    #[serde(default)]
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    #[serde(default)]
    pub include: Vec<String>,
}

pub(crate) fn default_k() -> usize {
    10
}

#[derive(Serialize)]
pub struct QueryResponse {
    pub ids: Vec<Vec<String>>,
    pub distances: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Vec<Option<Metadata>>>>,
    pub documents: Option<Vec<Vec<Option<String>>>>,
}

#[derive(Deserialize)]
pub struct GetRequest {
    pub ids: Option<Vec<Id>>,
    pub where_filter: Option<WhereFilter>,
    #[serde(default)]
    pub where_document: Option<rekha_core::filter::WhereDocument>,
    #[serde(default)]
    pub include: Vec<String>,
}

#[derive(Serialize)]
pub struct GetResponse {
    pub ids: Vec<String>,
    pub embeddings: Option<Vec<Vec<f32>>>,
    pub metadatas: Option<Vec<Option<Metadata>>>,
    pub documents: Option<Vec<Option<String>>>,
}

#[derive(Serialize)]
pub struct CountResponse {
    pub count: u64,
}
