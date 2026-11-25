use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Maximum records accepted in a single write or delete batch.
/// Advertised by /pre-flight-checks and enforced by the write handlers.
pub const MAX_BATCH_SIZE: usize = 10_000;
pub(crate) const MAX_QUERY_EMBEDDINGS: usize = 100;
pub(crate) const MAX_N_RESULTS: usize = 1_000;
pub(crate) const MAX_ID_LENGTH: usize = 512;
pub(crate) const MAX_NAME_LENGTH: usize = 512;
pub(crate) const MAX_DIMENSION: usize = 65_536;

pub(crate) fn validation_err(msg: impl Into<String>) -> EngineError {
    EngineError::Validation(msg.into())
}

pub(crate) fn validate_name(name: &str) -> Result<(), EngineError> {
    if name.is_empty() {
        return Err(validation_err("name must not be empty"));
    }
    if name.len() > MAX_NAME_LENGTH {
        return Err(validation_err(format!(
            "name exceeds max length {MAX_NAME_LENGTH}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_ids(ids: &[Id]) -> Result<(), EngineError> {
    if ids.is_empty() {
        return Err(validation_err("ids must not be empty"));
    }
    if ids.len() > MAX_BATCH_SIZE {
        return Err(validation_err(format!(
            "batch size {} exceeds max {MAX_BATCH_SIZE}",
            ids.len()
        )));
    }
    for id in ids {
        if id.is_empty() {
            return Err(validation_err("id must not be empty"));
        }
        if id.len() > MAX_ID_LENGTH {
            return Err(validation_err(format!(
                "id exceeds max length {MAX_ID_LENGTH}: {id}"
            )));
        }
    }
    Ok(())
}

pub fn validate_embedding_slice(emb: &[f32]) -> Result<(), EngineError> {
    if emb.is_empty() {
        return Err(validation_err("embedding must not be empty"));
    }
    if emb.len() > MAX_DIMENSION {
        return Err(validation_err(format!(
            "embedding dimension {} exceeds max {MAX_DIMENSION}",
            emb.len()
        )));
    }
    if !emb.iter().all(|v| v.is_finite()) {
        return Err(validation_err(
            "embedding contains non-finite value (NaN or Infinity)",
        ));
    }
    Ok(())
}

pub(crate) fn validate_embeddings(embeddings: &[Vec<f32>]) -> Result<(), EngineError> {
    if embeddings.is_empty() {
        return Err(validation_err("embeddings must not be empty"));
    }
    for emb in embeddings {
        validate_embedding_slice(emb)?;
    }
    Ok(())
}

pub(crate) fn validate_add_request(req: &AddRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    validate_embeddings(&req.embeddings)?;
    if req.ids.len() != req.embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            req.embeddings.len()
        )));
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

pub(crate) fn validate_upsert_request(req: &UpsertRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    validate_embeddings(&req.embeddings)?;
    if req.ids.len() != req.embeddings.len() {
        return Err(validation_err(format!(
            "ids len {} != embeddings len {}",
            req.ids.len(),
            req.embeddings.len()
        )));
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

pub(crate) fn validate_query_request(req: &QueryRequest) -> Result<(), EngineError> {
    if req.query_embeddings.is_empty() {
        return Err(validation_err("query_embeddings must not be empty"));
    }
    if req.query_embeddings.len() > MAX_QUERY_EMBEDDINGS {
        return Err(validation_err(format!(
            "query_embeddings len {} exceeds max {MAX_QUERY_EMBEDDINGS}",
            req.query_embeddings.len()
        )));
    }
    if req.n_results == 0 {
        return Err(validation_err("n_results must be >= 1"));
    }
    if req.n_results > MAX_N_RESULTS {
        return Err(validation_err(format!(
            "n_results {} exceeds max {MAX_N_RESULTS}",
            req.n_results
        )));
    }
    for emb in &req.query_embeddings {
        validate_embedding_slice(emb)?;
    }
    Ok(())
}

pub(crate) fn validate_delete_request(req: &DeleteRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)
}

pub(crate) fn validate_update_request(req: &UpdateRequest) -> Result<(), EngineError> {
    validate_ids(&req.ids)?;
    if let Some(ref embs) = req.embeddings {
        if embs.len() != req.ids.len() {
            return Err(validation_err(format!(
                "embeddings len {} != ids len {}",
                embs.len(),
                req.ids.len()
            )));
        }
        for emb in embs {
            validate_embedding_slice(emb.as_ref())?;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(n: usize) -> Vec<Id> {
        (0..n).map(|i| format!("id-{i}")).collect()
    }

    #[test]
    fn names_are_bounded_and_non_empty() {
        assert!(validate_name("ok").is_ok());
        assert!(matches!(validate_name(""), Err(EngineError::Validation(_))));
        assert!(validate_name(&"n".repeat(MAX_NAME_LENGTH + 1)).is_err());
    }

    #[test]
    fn ids_are_bounded_non_empty_and_unique_enough() {
        assert!(validate_ids(&ids(2)).is_ok());
        assert!(validate_ids(&[]).is_err());
        assert!(validate_ids(&vec!["x".to_owned(); MAX_BATCH_SIZE + 1]).is_err());
        assert!(validate_ids(&[String::new()]).is_err());
        assert!(validate_ids(&["i".repeat(MAX_ID_LENGTH + 1)]).is_err());
    }

    #[test]
    fn embeddings_must_be_non_empty_finite_and_within_the_dimension_limit() {
        assert!(validate_embedding_slice(&[1.0, 2.0]).is_ok());
        assert!(validate_embedding_slice(&[]).is_err());
        assert!(validate_embedding_slice(&[f32::NAN]).is_err());
        assert!(validate_embedding_slice(&[f32::INFINITY]).is_err());
        assert!(validate_embedding_slice(&vec![0.0; MAX_DIMENSION + 1]).is_err());
        assert!(validate_embeddings(&[]).is_err());
    }

    #[test]
    fn add_requests_reject_every_length_mismatch() {
        let base = || AddRequest {
            ids: ids(2),
            embeddings: vec![vec![1.0], vec![2.0]],
            metadatas: None,
            documents: None,
        };
        assert!(validate_add_request(&base()).is_ok());

        let mut short = base();
        short.embeddings.pop();
        assert!(validate_add_request(&short).is_err());

        let mut meta = base();
        meta.metadatas = Some(vec![None]);
        assert!(validate_add_request(&meta).is_err());

        let mut docs = base();
        docs.documents = Some(vec![None]);
        assert!(validate_add_request(&docs).is_err());
    }

    #[test]
    fn upsert_requests_reject_every_length_mismatch() {
        let base = || UpsertRequest {
            ids: ids(2),
            embeddings: vec![vec![1.0], vec![2.0]],
            metadatas: None,
            documents: None,
        };
        assert!(validate_upsert_request(&base()).is_ok());

        let mut short = base();
        short.embeddings.pop();
        assert!(validate_upsert_request(&short).is_err());

        let mut meta = base();
        meta.metadatas = Some(vec![None, None, None]);
        assert!(validate_upsert_request(&meta).is_err());

        let mut docs = base();
        docs.documents = Some(vec![None]);
        assert!(validate_upsert_request(&docs).is_err());
    }

    #[test]
    fn query_requests_bound_vectors_and_k() {
        let mut req = QueryRequest {
            query_embeddings: vec![vec![1.0]],
            n_results: 1,
            where_filter: None,
            where_document: None,
            include: Vec::new(),
        };
        assert!(validate_query_request(&req).is_ok());

        req.query_embeddings.clear();
        assert!(validate_query_request(&req).is_err());

        req.query_embeddings = vec![vec![1.0]; MAX_QUERY_EMBEDDINGS + 1];
        assert!(validate_query_request(&req).is_err());

        req.query_embeddings = vec![vec![1.0]];
        req.n_results = 0;
        assert!(validate_query_request(&req).is_err());

        req.n_results = MAX_N_RESULTS + 1;
        assert!(validate_query_request(&req).is_err());

        req.n_results = 1;
        req.query_embeddings = vec![vec![f32::NAN]];
        assert!(validate_query_request(&req).is_err());
    }

    #[test]
    fn delete_and_update_requests_share_the_id_rules() {
        assert!(validate_delete_request(&DeleteRequest { ids: ids(1) }).is_ok());
        assert!(validate_delete_request(&DeleteRequest { ids: Vec::new() }).is_err());

        let mut update = UpdateRequest {
            ids: ids(2),
            embeddings: None,
            metadatas: None,
            documents: None,
        };
        assert!(validate_update_request(&update).is_ok());

        update.embeddings = Some(vec![Embedding::from(vec![1.0])]);
        assert!(
            validate_update_request(&update).is_err(),
            "a partial embedding list must be refused, not applied to a subset"
        );

        update.embeddings = Some(vec![
            Embedding::from(vec![1.0]),
            Embedding::from(vec![f32::NAN]),
        ]);
        assert!(validate_update_request(&update).is_err());

        update.embeddings = None;
        update.metadatas = Some(vec![None]);
        assert!(validate_update_request(&update).is_err());

        update.metadatas = None;
        update.documents = Some(vec![None]);
        assert!(validate_update_request(&update).is_err());
    }
}
