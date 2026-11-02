//! Object-store abstraction.
//!
//! This is the **compute/storage separation seam**: vector blocks, segment
//! files, and snapshots live on an object store behind [`Storage`], never
//! inside the catalog. Local filesystem today; S3/GCS (via the `object_store`
//! crate) in the cluster phase. The trait is deliberately small, sync, and
//! `Send + Sync` so the engine can wrap calls in `spawn_blocking`.
//!
//! Keys are opaque forward-slash paths (e.g. `col/<uuid>/segments/seg-0.bin`).
//! Implementations decide how to map a key onto real storage, but must not
//! allow a key to escape its namespace (see [`LocalStorage`](crate::LocalStorage)
//! for the strict sanitization we require).

use std::io;

use thiserror::Error;

/// Errors returned by an object store.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The underlying filesystem / object store failed.
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// The key is malformed (absolute path, `..` traversal, empty segment, ...).
    #[error("invalid object key `{0}`")]
    InvalidKey(String),
    /// The object does not exist where an error (rather than `None`) is
    /// expected — e.g. `delete` of a missing key.
    #[error("object `{0}` not found")]
    NotFound(String),
}

/// Blob storage for vector blocks and segments.
pub trait Storage: Send + Sync {
    /// Write `bytes` at `key`, atomically replacing any existing object.
    fn put(&self, key: &str, bytes: &[u8]) -> Result<(), StorageError>;

    /// Read the object at `key`. Returns `Ok(None)` if the object does not exist.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, StorageError>;

    /// List every object key (relative to the storage root) under `prefix`.
    /// An empty prefix lists the whole namespace.
    fn list(&self, prefix: &str) -> Result<Vec<String>, StorageError>;

    /// Remove the object at `key`. Errors with [`StorageError::NotFound`] if it
    /// does not exist.
    fn delete(&self, key: &str) -> Result<(), StorageError>;

    /// Whether an object exists at `key`.
    fn exists(&self, key: &str) -> Result<bool, StorageError>;
}
