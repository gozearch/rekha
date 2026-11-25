use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Error mapping (newtype wrapper to satisfy orphan rules)
// ---------------------------------------------------------------------------

pub(crate) enum ApiError {
    Engine(EngineError),
    /// This node may not perform the write. `leader` names the node that can,
    /// so a client or proxy can retry against it.
    NotLeader {
        leader: Option<u64>,
    },
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        match self {
            ApiError::Engine(e) => {
                let status = engine_error_status(&e);
                let message = e.to_string();
                (status, Json(serde_json::json!({ "error": message }))).into_response()
            }
            ApiError::NotLeader { leader } => {
                let mut headers = axum::http::HeaderMap::new();
                if let Some(id) = leader
                    && let Ok(value) = axum::http::HeaderValue::from_str(&id.to_string())
                {
                    headers.insert(
                        axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                        value,
                    );
                }
                headers.insert(
                    axum::http::header::RETRY_AFTER,
                    axum::http::HeaderValue::from_static("1"),
                );
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    headers,
                    Json(serde_json::json!({
                        "error": "this node is not the leader for this write; retry against the leader",
                        "leader": leader,
                    })),
                )
                    .into_response()
            }
        }
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError::Engine(e)
    }
}
/// Shared helper: map EngineError variants to HTTP status.
pub(crate) fn engine_error_status(err: &EngineError) -> StatusCode {
    match err {
        EngineError::CollectionNotFound(_) => StatusCode::NOT_FOUND,
        // A catalog identity collision is the caller's conflict, not a server
        // fault. Chroma answers a duplicate collection name with 409; returning
        // 500 made a retriable race look like a crash.
        EngineError::Catalog(rekha_storage::CatalogError::AlreadyExists) => StatusCode::CONFLICT,
        EngineError::Catalog(rekha_storage::CatalogError::NotFound) => StatusCode::NOT_FOUND,
        EngineError::Catalog(rekha_storage::CatalogError::InvalidArgument(_)) => {
            StatusCode::BAD_REQUEST
        }
        EngineError::Validation(_) => StatusCode::BAD_REQUEST,
        EngineError::Forbidden(_) => StatusCode::FORBIDDEN,
        // Retryable, and never a number: a count that cannot be fetched is not
        // zero.
        EngineError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_error_maps_to_the_status_a_client_can_act_on() {
        let cases = [
            (
                EngineError::CollectionNotFound("x".into()),
                StatusCode::NOT_FOUND,
            ),
            (
                EngineError::Catalog(rekha_storage::CatalogError::AlreadyExists),
                StatusCode::CONFLICT,
            ),
            (
                EngineError::Catalog(rekha_storage::CatalogError::NotFound),
                StatusCode::NOT_FOUND,
            ),
            (
                EngineError::Validation("bad".into()),
                StatusCode::BAD_REQUEST,
            ),
            (EngineError::Forbidden("no".into()), StatusCode::FORBIDDEN),
            (
                EngineError::Unavailable("later".into()),
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                EngineError::Io(std::io::Error::other("disk")),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];
        for (error, status) in cases {
            assert_eq!(engine_error_status(&error), status, "{error:?}");
        }
    }

    #[test]
    fn a_not_leader_answer_names_the_leader_and_asks_for_a_retry() {
        let response = ApiError::NotLeader { leader: Some(7) }.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(leadership::LEADER_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("7")
        );
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );

        // No known leader still answers a retryable status, just without the
        // hint header.
        let response = ApiError::NotLeader { leader: None }.into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().get(leadership::LEADER_HEADER).is_none());
    }

    #[test]
    fn an_engine_error_body_carries_the_message() {
        let response = ApiError::Engine(EngineError::Validation("nope".into())).into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
