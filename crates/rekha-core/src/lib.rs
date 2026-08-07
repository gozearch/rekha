//! RekhaDB core types.
//!
//! These are the shared vocabulary for the rest of the workspace: embeddings,
//! ids, metadata, filters, config, cluster primitives, and WAL operations.
//!
//! Design principle: **the WAL is the source of truth; indexes are derived.**
//! [`op::Operation`] records are self-contained and replayable, [`cluster`]
//! defines the epoch/clock machinery used to order and fence writes, and
//! [`filter::WhereFilter`] implements Chroma-compatible metadata filtering.

pub mod cluster;
pub mod config;
pub mod filter;
pub mod op;
pub mod types;
