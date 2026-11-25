use crate::prelude::*;
use crate::*;

// ---------------------------------------------------------------------------
// Tenants and databases
// ---------------------------------------------------------------------------

/// A namespace request that could not be served.
///
/// Separate from `ApiError` because these are not collection-scoped: the
/// failure is about whether a namespace exists, not about the data in it. It is
/// a struct rather than a tuple because a refusal also carries routing headers,
/// and a 136-byte tuple in every `Result` trips `result_large_err`.
#[derive(Debug)]
pub(crate) struct NamespaceError {
    status: StatusCode,
    message: String,
    /// The node that can accept this write, when this one cannot.
    leader: Option<u64>,
}

impl NamespaceError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            leader: None,
        }
    }
}

impl IntoResponse for NamespaceError {
    fn into_response(self) -> axum::response::Response {
        let mut headers = axum::http::HeaderMap::new();
        if let Some(leader) = self.leader
            && let Ok(value) = axum::http::HeaderValue::from_str(&leader.to_string())
        {
            headers.insert(
                axum::http::HeaderName::from_static(leadership::LEADER_HEADER),
                value,
            );
            headers.insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("1"),
            );
        }
        let mut body = serde_json::json!({ "error": self.message });
        if let Some(leader) = self.leader {
            body["leader"] = serde_json::json!(leader);
        }
        (self.status, headers, Json(body)).into_response()
    }
}

#[derive(Deserialize)]
pub(crate) struct NamespaceRequest {
    name: String,
}

pub(crate) fn namespace_error(status: StatusCode, message: impl Into<String>) -> NamespaceError {
    NamespaceError::new(status, message)
}

/// The same `503` + `x-rekha-leader` contract the collection path uses.
///
/// A node whose cached leadership is stale appends to a log it does not lead,
/// and `client_write` says so instead of forwarding. Reporting a bare error
/// would hide the one fact the caller needs: another node can accept this write.
pub(crate) fn not_leader(leader: Option<u64>) -> NamespaceError {
    NamespaceError {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message: "this node is not the metadata leader; retry against the leader".into(),
        leader,
    }
}

/// Propose a metadata operation to the log.
///
/// Returns `Ok(false)` on a single-node deployment, where the caller applies the
/// change to its own catalog and the log has nothing to say about it.
pub(crate) async fn replicate_metadata(
    state: &AppState,
    op: ClusterOperation,
) -> Result<bool, NamespaceError> {
    let Some(raft) = state.raft.as_ref() else {
        if state.shards.is_some() {
            // A clustered node with no metadata group has not been admitted yet.
            // Writing its local catalog would create a namespace exactly one
            // node knows about.
            return Err(namespace_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "this node has not joined a cluster yet; retry once it is admitted",
            ));
        }
        return Ok(false);
    };
    use openraft::error::{ClientWriteError, RaftError};
    match raft.client_write(op).await {
        Ok(_) => Ok(true),
        Err(RaftError::APIError(ClientWriteError::ForwardToLeader(forward))) => {
            Err(not_leader(forward.leader_id))
        }
        Err(e) => Err(namespace_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("the metadata log refused this write: {e}"),
        )),
    }
}

pub(crate) fn missing_namespace(what: &str) -> NamespaceError {
    namespace_error(StatusCode::NOT_FOUND, format!("{what} does not exist"))
}

pub(crate) fn catalog_unavailable(e: rekha_storage::CatalogError) -> NamespaceError {
    namespace_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// Validate a tenant or database name and trim it.
pub(crate) fn namespace_name(raw: &str) -> Result<String, NamespaceError> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(namespace_error(
            StatusCode::BAD_REQUEST,
            "name must not be empty",
        ));
    }
    validate_name(name).map_err(|e| {
        let status = engine_error_status(&e);
        namespace_error(status, e.to_string())
    })?;
    Ok(name.to_string())
}

/// Every database that exists under `tenant`.
///
/// A namespace exists if it was created, or if a collection already lives in
/// it. Both are catalog facts, so this is a union of two reads of one store
/// rather than a second source of truth — but it is a union because a
/// collection created before its database row existed must not make its own
/// database invisible. Reporting an empty list for a namespace that holds data
/// is the same class of lie as a create that persists nothing.
pub(crate) fn databases_of(
    catalog: &dyn rekha_storage::Catalog,
    tenant: &str,
) -> Result<std::collections::BTreeSet<String>, rekha_storage::CatalogError> {
    let mut names = std::collections::BTreeSet::new();
    for (owner, database) in catalog.list_databases()? {
        if owner == tenant {
            names.insert(database);
        }
    }
    for record in catalog.list_collections_all()? {
        if record.config.tenant == tenant {
            names.insert(record.config.database);
        }
    }
    Ok(names)
}

pub(crate) fn tenant_exists(
    catalog: &dyn rekha_storage::Catalog,
    tenant: &str,
) -> Result<bool, rekha_storage::CatalogError> {
    Ok(catalog.list_tenants()?.iter().any(|known| known == tenant)
        || !databases_of(catalog, tenant)?.is_empty())
}

/// Chroma's client requires a database id; the catalog keys by name.
pub(crate) fn database_id(tenant: &str, name: &str) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_OID, format!("{tenant}/{name}").as_bytes())
}

/// GET /api/v2/tenants/{name} — does this tenant exist?
pub(crate) async fn get_tenant(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    // The configured namespace always exists: the server serves it, whether or
    // not a row has been written yet.
    if name != state.tenant
        && !tenant_exists(catalog.as_ref(), &name).map_err(catalog_unavailable)?
    {
        return Err(missing_namespace(&format!("tenant {name}")));
    }
    Ok(Json(serde_json::json!({ "name": name })))
}

/// POST /api/v2/tenants — create a tenant.
///
/// Idempotent, and the response is sent only once the namespace exists. In
/// cluster mode the row is a log entry, so a tenant created here is visible on
/// every node instead of only on the one that served the request.
pub(crate) async fn create_tenant(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NamespaceRequest>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let name = namespace_name(&req.name)?;
    if !replicate_metadata(
        &state,
        ClusterOperation::CreateTenant { name: name.clone() },
    )
    .await?
    {
        match state.engine.catalog().create_tenant(&name) {
            Ok(()) | Err(rekha_storage::CatalogError::AlreadyExists) => {}
            Err(e) => return Err(catalog_unavailable(e)),
        }
    }
    Ok(Json(serde_json::json!({ "name": name })))
}

/// GET /api/v2/tenants/{tenant}/databases — list databases under a tenant.
pub(crate) async fn list_databases(
    State(state): State<Arc<AppState>>,
    Path(tenant): Path<String>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    if tenant != state.tenant
        && !tenant_exists(catalog.as_ref(), &tenant).map_err(catalog_unavailable)?
    {
        return Err(missing_namespace(&format!("tenant {tenant}")));
    }
    let mut names = databases_of(catalog.as_ref(), &tenant).map_err(catalog_unavailable)?;
    if tenant == state.tenant {
        names.insert(state.database.clone());
    }
    let rows: Vec<serde_json::Value> = names
        .into_iter()
        .map(|name| {
            serde_json::json!({
                "id": database_id(&tenant, &name),
                "name": name,
                "tenant": tenant,
            })
        })
        .collect();
    Ok(Json(serde_json::json!(rows)))
}

/// GET /api/v2/tenants/{tenant}/databases/{name} — does this database exist?
pub(crate) async fn get_database(
    State(state): State<Arc<AppState>>,
    Path((tenant, name)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let catalog = state.engine.catalog();
    let exists = (tenant == state.tenant && name == state.database)
        || databases_of(catalog.as_ref(), &tenant)
            .map_err(catalog_unavailable)?
            .contains(&name);
    if !exists {
        return Err(missing_namespace(&format!("database {tenant}/{name}")));
    }
    Ok(Json(serde_json::json!({
        "id": database_id(&tenant, &name),
        "name": name,
        "tenant": tenant,
    })))
}

/// POST /api/v2/tenants/{tenant}/databases — create a database.
///
/// A database is created inside an existing tenant; a missing tenant is a 404,
/// never an implicit create. Idempotent, and replicated in cluster mode for the
/// same reason as the tenant itself.
pub(crate) async fn create_database(
    State(state): State<Arc<AppState>>,
    Path(tenant): Path<String>,
    Json(req): Json<NamespaceRequest>,
) -> Result<Json<serde_json::Value>, NamespaceError> {
    let name = namespace_name(&req.name)?;
    let catalog = state.engine.catalog();
    if !tenant_exists(catalog.as_ref(), &tenant).map_err(catalog_unavailable)? {
        return Err(missing_namespace(&format!("tenant {tenant}")));
    }
    match replicate_metadata(
        &state,
        ClusterOperation::CreateDatabase {
            tenant: tenant.clone(),
            name: name.clone(),
        },
    )
    .await
    {
        Ok(true) => {}
        Ok(false) => match catalog.create_database(&tenant, &name) {
            Ok(()) | Err(rekha_storage::CatalogError::AlreadyExists) => {}
            Err(e) => return Err(catalog_unavailable(e)),
        },
        Err(e) => return Err(e),
    }
    Ok(Json(serde_json::json!({ "name": name, "tenant": tenant })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_namespace_refusal_names_the_leader_and_asks_for_a_retry() {
        let response = not_leader(Some(4)).into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            response
                .headers()
                .get(leadership::LEADER_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("4")
        );
        assert_eq!(
            response
                .headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1")
        );
    }

    #[test]
    fn a_namespace_refusal_without_a_known_leader_still_asks_for_a_retry() {
        let response = not_leader(None).into_response();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().get(leadership::LEADER_HEADER).is_none());
    }

    #[test]
    fn namespace_names_are_trimmed_and_validated() {
        assert_eq!(namespace_name(" acme ").unwrap(), "acme");
        assert_eq!(
            namespace_name("   ").unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            namespace_name(&"n".repeat(513)).unwrap_err().status,
            StatusCode::BAD_REQUEST
        );
    }

    #[test]
    fn catalog_and_lookup_failures_map_to_the_status_an_operator_expects() {
        assert_eq!(
            catalog_unavailable(rekha_storage::CatalogError::NotFound).status,
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(missing_namespace("tenant").status, StatusCode::NOT_FOUND);
        assert!(
            missing_namespace("tenant")
                .into_response()
                .status()
                .is_client_error()
        );
    }
}
