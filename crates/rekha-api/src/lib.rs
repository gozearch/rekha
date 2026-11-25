#![deny(clippy::unwrap_used, clippy::expect_used)]

//! RekhaDB HTTP API — Chroma-compatible REST server.
//!
//! The server is split by responsibility rather than by handler count:
//!
//! - [`state`] owns the process-wide [`AppState`] every handler receives.
//! - [`routes`] assembles the public and internal routers.
//! - [`handlers`] implements the client-facing endpoints.
//! - [`internal`] implements the peer-facing Raft/shard RPCs.
//! - [`shard`] holds the shard routing and two-phase-commit dispatch.
//! - [`collection_ops`] holds the write path that talks to consensus.
//! - [`error`], [`validation`], [`models`], [`authz`] are the shared seams.

pub mod auth_routes;
pub mod chroma;
pub mod leadership;
pub mod middleware;

mod admin;
mod authz;
mod collection_ops;
mod error;
mod handlers;
mod internal;
mod models;
mod namespace;
mod routes;
mod runtime;
mod shard;
mod state;
mod stats;
mod validation;

/// Imports shared by every module in this crate.
///
/// The crate was one 5,000-line file before it was split; keeping the common
/// imports in one place means a new module starts with two `use` lines rather
/// than a copy of the previous module's header.
pub(crate) mod prelude {
    pub use std::collections::HashMap;
    pub use std::sync::Arc;
    pub use std::sync::atomic::{AtomicBool, Ordering};

    pub use axum::extract::{Extension, Path, State};
    pub use axum::http::StatusCode;
    pub use axum::response::IntoResponse;
    pub use axum::routing::{get, post};
    pub use axum::{Json, Router};
    pub use openraft::raft::{
        AppendEntriesRequest, AppendEntriesResponse, VoteRequest, VoteResponse,
    };
    pub use rekha_cluster::RaftTypeConfig;
    pub use rekha_cluster::raft_types::ClusterOperation;
    pub use rekha_core::config::CollectionConfig;
    pub use rekha_core::filter::WhereFilter;
    pub use rekha_core::types::{Distance, Embedding, Id, Metadata};
    pub use rekha_engine::{Engine, EngineError, QueryOptions};
    pub use serde::{Deserialize, Serialize};
    pub use uuid::Uuid;

    pub use crate::state::AppState;
}

// The crate root stays the single import surface for the modules, which keeps
// call sites unchanged from the pre-split file. Public API is re-exported
// deliberately; everything else is crate-visible only.
pub use routes::{internal_router, public_router, router};
pub use state::{AppState, ServerConfig};
pub use validation::{MAX_BATCH_SIZE, validate_embedding_slice};

pub(crate) use admin::*;
pub(crate) use authz::*;
pub(crate) use collection_ops::*;
pub(crate) use error::*;
pub(crate) use handlers::collections::*;
pub(crate) use handlers::health::*;
pub(crate) use handlers::records::*;
pub(crate) use internal::*;
pub(crate) use models::*;
pub(crate) use namespace::*;
pub(crate) use runtime::*;
pub(crate) use shard::*;
pub(crate) use stats::*;
pub(crate) use validation::*;
