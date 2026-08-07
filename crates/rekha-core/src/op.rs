//! WAL operation records.
//!
//! Every write is expressed as an [`Operation`]. Records are fully
//! self-contained (they carry their complete payload), so any fragment of the
//! log can be shipped to another replica or replayed after a crash.
//!
//! Design: **the WAL is the source of truth; indexes are derived.** Indexes are
//! rebuilt by replaying operations from this log, so an operation must never
//! reference out-of-band state.

use serde::{Deserialize, Serialize};

use crate::types::{Document, Embedding, Id, Metadata};

/// A single WAL write operation. Self-contained and replayable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Insert a brand-new record with a fresh embedding.
    Add {
        id: Id,
        #[serde(with = "crate::types::embedding_serde")]
        embedding: Embedding,
        metadata: Option<Metadata>,
        document: Option<Document>,
    },
    /// Update the metadata/document of an existing record. No embedding change
    /// (a changed embedding is an [`Operation::Upsert`]).
    Update {
        id: Id,
        metadata: Option<Metadata>,
        document: Option<Document>,
    },
    /// Remove a record by id (soft delete at the index layer).
    Delete { id: Id },
    /// Add-or-update by id: insert if absent, otherwise replace the embedding,
    /// metadata, and document.
    Upsert {
        id: Id,
        #[serde(with = "crate::types::embedding_serde")]
        embedding: Embedding,
        metadata: Option<Metadata>,
        document: Option<Document>,
    },
}

impl Operation {
    /// The record id this operation acts on.
    pub fn id(&self) -> &str {
        match self {
            Operation::Add { id, .. }
            | Operation::Update { id, .. }
            | Operation::Delete { id }
            | Operation::Upsert { id, .. } => id,
        }
    }

    /// Canonical operation name: `"add"`, `"update"`, `"delete"`, `"upsert"`.
    pub fn kind(&self) -> &'static str {
        match self {
            Operation::Add { .. } => "add",
            Operation::Update { .. } => "update",
            Operation::Delete { .. } => "delete",
            Operation::Upsert { .. } => "upsert",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(v: Vec<f32>) -> Embedding {
        v.into()
    }

    #[test]
    fn operation_id_and_kind() {
        let add = Operation::Add {
            id: "a".into(),
            embedding: embedding(vec![1.0]),
            metadata: None,
            document: None,
        };
        assert_eq!(add.id(), "a");
        assert_eq!(add.kind(), "add");

        let update = Operation::Update {
            id: "b".into(),
            metadata: None,
            document: None,
        };
        assert_eq!(update.id(), "b");
        assert_eq!(update.kind(), "update");

        let delete = Operation::Delete { id: "c".into() };
        assert_eq!(delete.id(), "c");
        assert_eq!(delete.kind(), "delete");

        let upsert = Operation::Upsert {
            id: "d".into(),
            embedding: embedding(vec![2.0]),
            metadata: None,
            document: None,
        };
        assert_eq!(upsert.id(), "d");
        assert_eq!(upsert.kind(), "upsert");
    }

    #[test]
    fn operation_serde_roundtrip() {
        let mut meta = crate::types::Metadata::new();
        meta.insert("tag".into(), crate::types::MetadataValue::Str("x".into()));
        let op = Operation::Add {
            id: "id-1".into(),
            embedding: embedding(vec![1.0, 2.0, 3.0]),
            metadata: Some(meta),
            document: Some("hello".into()),
        };
        let json = serde_json::to_string(&op).unwrap();
        let back: Operation = serde_json::from_str(&json).unwrap();
        assert_eq!(op, back);
    }
}
