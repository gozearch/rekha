//! Request authentication middleware.

use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use rekha_core::auth::{RouteAction, has_permission, hash_api_key_with_pepper, route_action};

/// Authenticated request context attached to request extensions after successful auth.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub user_name: String,
    pub permissions: Vec<rekha_core::auth::Permission>,
}

/// The identity the configured shared key authenticates as.
///
/// Built in one place because two paths reach it — the legacy single-key
/// deployment, and the window in which a clustered node has applied the
/// bootstrap user but not yet its key row. Two constructions would be two
/// permission sets that can drift.
pub(crate) fn shared_key_context() -> AuthContext {
    AuthContext {
        user_id: "legacy".into(),
        user_name: "legacy".into(),
        permissions: vec![rekha_core::auth::Permission {
            id: "legacy-wide".into(),
            action: "*".into(),
            tenant: None,
            database: None,
        }],
    }
}

/// Public RBAC auth middleware.
///
/// 1. Extract API key from `x-chroma-token` / `x-api-key` / `Authorization: Bearer`.
/// 2. If RBAC users exist, require a valid RBAC key (lookup via catalog).
/// 3. Else if legacy `api_key` is set, check via constant-time compare and treat as superadmin.
/// 4. Else (no RBAC, no legacy key) allow through (bootstrap / open mode).
/// 5. For matched routes, check `route_action` against the caller's permissions.
pub async fn auth(
    mut request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let state = request
        .extensions()
        .get::<std::sync::Arc<super::AppState>>()
        .cloned()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    let path = request.uri().path().to_owned();
    let method = request.method().to_string();
    let is_public = matches!(
        path.as_str(),
        "/health"
            | "/ready"
            | "/api/v2/heartbeat"
            | "/api/v2/version"
            | "/api/v2/pre-flight-checks"
    );
    if is_public {
        return Ok(next.run(request).await);
    }

    let headers: HeaderMap = request.headers().clone();
    let provided = headers
        .get("x-chroma-token")
        .or_else(|| headers.get("x-api-key"))
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| {
                    if s.to_ascii_lowercase().starts_with("bearer ") {
                        Some(s[7..].trim().to_string())
                    } else {
                        None
                    }
                })
        });

    let catalog = state.engine.catalog();
    // A failed read is not "there are no users". Treating it as one fell
    // through to the legacy shared-key path, so a storage error was a
    // fail-open. Ask once, and refuse the request if the answer cannot be
    // read.
    let users_exist = match catalog.list_users() {
        Ok(users) => !users.is_empty(),
        Err(_) => return Err(StatusCode::INTERNAL_SERVER_ERROR),
    };

    let auth_ctx = if users_exist {
        let token = provided.ok_or(StatusCode::UNAUTHORIZED)?;
        let hash = hash_api_key_with_pepper(&token, &state.pepper);
        // Deliberately no permission cache. A cache here has to be
        // invalidated on every user, key and role change -- and those
        // changes are log entries that apply on every node -- while the
        // version this replaces was invalidated nowhere: a deleted user kept
        // working until the entry's TTL expired. The reads it saved are
        // in-process table lookups on the store this request already reads
        // to check collection access.
        let looked_up = catalog
            .lookup_api_key(&hash)
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
        match looked_up {
            Some((user_id, _meta)) => {
                let user = catalog
                    .get_user(&user_id)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
                    .ok_or(StatusCode::UNAUTHORIZED)?;
                if !user.enabled {
                    return Err(StatusCode::UNAUTHORIZED);
                }
                let perms = catalog
                    .get_user_permissions(&user_id)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                AuthContext {
                    user_id: user.id.clone(),
                    user_name: user.name.clone(),
                    permissions: perms,
                }
            }
            // One apply writes the bootstrap user, its key and its role, so a
            // reader can arrive after the user row lands and before the key
            // row does. The honest answer for that instant is not "401 for a
            // cluster that is still starting". The environment key is the
            // credential the bootstrap hashes, so accepting *it* here keeps
            // the node authenticable, and the constant-time comparison
            // against the configured secret means no other credential is
            // widened.
            None if state.api_key.as_deref().is_some_and(|expected| {
                token.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1
            }) =>
            {
                shared_key_context()
            }
            None => return Err(StatusCode::UNAUTHORIZED),
        }
    } else if let Some(ref expected) = state.api_key {
        let token = provided.ok_or(StatusCode::UNAUTHORIZED)?;
        let authorized = token.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1;
        if !authorized {
            return Err(StatusCode::UNAUTHORIZED);
        }
        shared_key_context()
    } else {
        if !state.allow_open_mode {
            return Err(StatusCode::UNAUTHORIZED);
        }
        AuthContext {
            user_id: "anonymous".into(),
            user_name: "anonymous".into(),
            permissions: vec![rekha_core::auth::Permission {
                id: "open".into(),
                action: "*".into(),
                tenant: None,
                database: None,
            }],
        }
    };

    // Authorization fails closed. The previous form checked permissions
    // `if let Some(..)` and skipped everything else, so a route the mapping
    // did not recognise was allowed for any authenticated caller — which is
    // how `/optimize` was unprotected until it was patched by hand.
    match route_action(method.as_str(), &path) {
        RouteAction::Public | RouteAction::AuthenticatedOnly => {}
        RouteAction::Permission {
            action,
            tenant,
            database,
        } => {
            // Native v2 collection routes carry no tenant in the path; the
            // handler resolves the collection's tenant and re-checks with
            // `check_collection_access`. Here the gate is coarse on purpose,
            // so a tenant-scoped operator is not denied before that check.
            let is_native = tenant.is_none()
                && database.is_none()
                && (action.starts_with("collection.") || action.starts_with("record."))
                && !path.starts_with("/api/v2/auth/");
            let allowed = if is_native {
                auth_ctx
                    .permissions
                    .iter()
                    .any(|p| p.action == "*" || p.action == action)
            } else {
                has_permission(
                    &auth_ctx.permissions,
                    &action,
                    tenant.as_deref(),
                    database.as_deref(),
                )
            };
            if !allowed {
                return Err(StatusCode::FORBIDDEN);
            }
        }
        RouteAction::Unmatched => {
            // Not a route this build knows. Deny rather than skip, and do it
            // before the router so an unlisted path cannot reach a handler
            // that forgot to authorise itself.
            return Err(StatusCode::NOT_FOUND);
        }
    }

    request.extensions_mut().insert(auth_ctx);
    Ok(next.run(request).await)
}

pub async fn internal_auth(
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let state = request
        .extensions()
        .get::<std::sync::Arc<super::AppState>>()
        .cloned()
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;

    // Fail closed. Internal routes drive Raft and WAL shipping; leaving them
    // open when no secret is configured would expose cluster administration
    // on any listener that mounts them. An *empty* secret is treated as
    // unset for the same reason: it would authenticate a request that sends
    // no header at all.
    let Some(ref expected) = state.cluster_secret else {
        return Err(StatusCode::UNAUTHORIZED);
    };
    if expected.is_empty() {
        return Err(StatusCode::UNAUTHORIZED);
    }
    let headers: HeaderMap = request.headers().clone();
    let provided = headers
        .get("x-rekha-cluster-secret")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let authorized = provided.as_bytes().ct_eq(expected.as_bytes()).unwrap_u8() == 1;
    if !authorized {
        return Err(StatusCode::UNAUTHORIZED);
    }
    Ok(next.run(request).await)
}
