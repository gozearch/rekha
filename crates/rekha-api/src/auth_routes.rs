//! RBAC management API handlers.

use std::sync::Arc;

use axum::extract::{Extension, Path, State};
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use rekha_cluster::raft_types::ClusterOperation;
use rekha_core::auth::{
    KeyMeta, Permission, RoleRecord, UserRecord, generate_api_key, has_permission,
    hash_api_key_with_pepper,
};
use rekha_storage::CatalogError;

use crate::AppState;
use crate::middleware::AuthContext;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn catalog_status(e: CatalogError) -> (StatusCode, Json<serde_json::Value>) {
    let (code, msg) = match &e {
        CatalogError::NotFound => (StatusCode::NOT_FOUND, e.to_string()),
        CatalogError::AlreadyExists => (StatusCode::CONFLICT, e.to_string()),
        CatalogError::InvalidArgument(m) => (StatusCode::BAD_REQUEST, m.clone()),
        CatalogError::Storage(m) => (StatusCode::INTERNAL_SERVER_ERROR, m.to_string()),
        CatalogError::Serialization(m) => (StatusCode::INTERNAL_SERVER_ERROR, m.clone()),
    };
    (code, Json(serde_json::json!({ "error": msg })))
}

fn require(auth: &AuthContext, action: &str) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    if has_permission(&auth.permissions, action, None, None) {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(serde_json::json!({ "error": format!("missing permission {action}") })),
        ))
    }
}

// ---------------------------------------------------------------------------
// Request / response DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateUserRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct CreateUserResponse {
    pub id: String,
    pub name: String,
    pub api_key: String,
}

#[derive(Serialize)]
pub struct UserResponse {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub created_at_ms: u64,
}

impl From<UserRecord> for UserResponse {
    fn from(u: UserRecord) -> Self {
        Self {
            id: u.id,
            name: u.name,
            enabled: u.enabled,
            created_at_ms: u.created_at_ms,
        }
    }
}

#[derive(Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
}

#[derive(Serialize)]
pub struct CreateKeyResponse {
    pub api_key: String,
    pub key_hash_hex: String,
}

#[derive(Deserialize)]
pub struct CreateRoleRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct RoleResponse {
    pub id: String,
    pub name: String,
    pub description: String,
    pub built_in: bool,
}

impl From<RoleRecord> for RoleResponse {
    fn from(r: RoleRecord) -> Self {
        Self {
            id: r.id,
            name: r.name,
            description: r.description,
            built_in: r.built_in,
        }
    }
}

#[derive(Deserialize)]
pub struct AddPermissionRequest {
    pub action: String,
    pub tenant: Option<String>,
    pub database: Option<String>,
}

#[derive(Deserialize)]
pub struct AssignRoleRequest {
    pub role_id: String,
}

// ---------------------------------------------------------------------------
// Handlers — Users
// ---------------------------------------------------------------------------

async fn create_user(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<CreateUserRequest>,
) -> Result<Json<CreateUserResponse>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.create")?;
    if req.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "name must not be empty" })),
        ));
    }
    let id = uuid::Uuid::new_v4().to_string();
    let user = UserRecord {
        id: id.clone(),
        name: req.name.clone(),
        enabled: true,
        created_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    };
    let api_key = generate_api_key();
    let hash = hash_api_key_with_pepper(&api_key, &state.pepper);
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::CreateUser {
            user: user.clone(),
            key_hash: hash,
            key_label: "default".into(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .create_user(&user)
            .map_err(catalog_status)?;
        state
            .engine
            .catalog()
            .create_api_key(&user.id, &hash, "default")
            .map_err(catalog_status)?;
    }
    Ok(Json(CreateUserResponse {
        id,
        name: req.name,
        api_key,
    }))
}

async fn list_users(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<UserResponse>>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.list")?;
    let users = state
        .engine
        .catalog()
        .list_users()
        .map_err(catalog_status)?;
    Ok(Json(users.into_iter().map(UserResponse::from).collect()))
}

async fn get_user(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<UserResponse>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.list")?;
    let user = state
        .engine
        .catalog()
        .get_user(&id)
        .map_err(catalog_status)?
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "user not found" })),
            )
        })?;
    Ok(Json(UserResponse::from(user)))
}

async fn delete_user(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.delete")?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::DeleteUser {
            user_id: id.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .delete_user(&id)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

async fn create_api_key(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(user_id): Path<String>,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<CreateKeyResponse>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.manage_keys")?;
    if req.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "key name must not be empty" })),
        ));
    }
    let api_key = generate_api_key();
    let hash = hash_api_key_with_pepper(&api_key, &state.pepper);
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::CreateApiKey {
            user_id: user_id.clone(),
            key_hash: hash,
            label: req.name.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .create_api_key(&user_id, &hash, &req.name)
            .map_err(catalog_status)?;
    }
    let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
    Ok(Json(CreateKeyResponse {
        api_key,
        key_hash_hex: hex,
    }))
}

async fn delete_api_key(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((user_id, hash_hex)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.manage_keys")?;
    let hash = hex_to_hash(&hash_hex)?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::DeleteApiKey {
            user_id: user_id.clone(),
            key_hash: hash,
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .delete_api_key(&user_id, &hash)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

fn hex_to_hash(s: &str) -> Result<[u8; 32], (StatusCode, Json<serde_json::Value>)> {
    if s.len() != 64 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "key hash must be 64 hex chars" })),
        ));
    }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hex = std::str::from_utf8(chunk).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid hex" })),
            )
        })?;
        out[i] = u8::from_str_radix(hex, 16).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "invalid hex" })),
            )
        })?;
    }
    Ok(out)
}

async fn list_user_keys(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<KeyMeta>>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.manage_keys")?;
    let keys = state
        .engine
        .catalog()
        .list_user_keys(&user_id)
        .map_err(catalog_status)?;
    Ok(Json(keys))
}

// ---------------------------------------------------------------------------
// Handlers — Roles
// ---------------------------------------------------------------------------

async fn create_role(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<CreateRoleRequest>,
) -> Result<Json<RoleResponse>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.create")?;
    if req.name.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "role name must not be empty" })),
        ));
    }
    let role = RoleRecord {
        id: uuid::Uuid::new_v4().to_string(),
        name: req.name,
        description: req.description.unwrap_or_default(),
        built_in: false,
    };
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::CreateRole { role: role.clone() };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .create_role(&role)
            .map_err(catalog_status)?;
    }
    Ok(Json(RoleResponse::from(role)))
}

async fn list_roles(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<RoleResponse>>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.assign")?;
    let roles = state
        .engine
        .catalog()
        .list_roles()
        .map_err(catalog_status)?;
    Ok(Json(roles.into_iter().map(RoleResponse::from).collect()))
}

async fn delete_role(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.delete")?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::DeleteRole {
            role_id: id.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .delete_role(&id)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

async fn add_permission(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(role_id): Path<String>,
    Json(req): Json<AddPermissionRequest>,
) -> Result<Json<Permission>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.manage_permissions")?;
    if req.action.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "action must not be empty" })),
        ));
    }
    let perm = Permission {
        id: uuid::Uuid::new_v4().to_string(),
        action: req.action,
        tenant: req.tenant,
        database: req.database,
    };
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::AddPermission {
            role_id: role_id.clone(),
            permission: perm.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .add_permission(&role_id, &perm)
            .map_err(catalog_status)?;
    }
    Ok(Json(perm))
}

async fn remove_permission(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((role_id, perm_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.manage_permissions")?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::RemovePermission {
            role_id: role_id.clone(),
            perm_id: perm_id.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .remove_permission(&role_id, &perm_id)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

async fn get_role_permissions(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(role_id): Path<String>,
) -> Result<Json<Vec<Permission>>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.assign")?;
    let perms = state
        .engine
        .catalog()
        .get_role_permissions(&role_id)
        .map_err(catalog_status)?;
    Ok(Json(perms))
}

// ---------------------------------------------------------------------------
// Handlers — Assignments
// ---------------------------------------------------------------------------

async fn assign_role(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(user_id): Path<String>,
    Json(req): Json<AssignRoleRequest>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.assign")?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::AssignRole {
            user_id: user_id.clone(),
            role_id: req.role_id.clone(),
            assigned_by: auth.user_id.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .assign_role(&user_id, &req.role_id, &auth.user_id)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

async fn unassign_role(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((user_id, role_id)): Path<(String, String)>,
) -> Result<StatusCode, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "role.assign")?;
    if let Some(ref raft) = state.raft {
        let op = ClusterOperation::UnassignRole {
            user_id: user_id.clone(),
            role_id: role_id.clone(),
        };
        raft.client_write(op).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;
    } else {
        state
            .engine
            .catalog()
            .unassign_role(&user_id, &role_id)
            .map_err(catalog_status)?;
    }
    Ok(StatusCode::OK)
}

async fn get_user_roles(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(user_id): Path<String>,
) -> Result<Json<Vec<RoleRecord>>, (StatusCode, Json<serde_json::Value>)> {
    require(&auth, "user.list")?;
    let roles = state
        .engine
        .catalog()
        .get_user_roles(&user_id)
        .map_err(catalog_status)?;
    Ok(Json(roles.into_iter().map(|(r, _)| r).collect()))
}

// ---------------------------------------------------------------------------
// Whoami
// ---------------------------------------------------------------------------

async fn whoami(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Json<serde_json::Value> {
    // Return current user + permissions
    let user = state
        .engine
        .catalog()
        .get_user(&auth.user_id)
        .ok()
        .flatten();
    Json(serde_json::json!({
        "user_id": auth.user_id,
        "user_name": auth.user_name,
        "enabled": user.map(|u| u.enabled).unwrap_or(true),
        "permissions": auth.permissions,
    }))
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn auth_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/v2/auth/users", post(create_user).get(list_users))
        .route("/api/v2/auth/users/{id}", get(get_user).delete(delete_user))
        .route(
            "/api/v2/auth/users/{id}/keys",
            post(create_api_key).get(list_user_keys),
        )
        .route(
            "/api/v2/auth/users/{id}/keys/{hash}",
            delete(delete_api_key),
        )
        .route(
            "/api/v2/auth/users/{id}/roles",
            get(get_user_roles).post(assign_role),
        )
        .route(
            "/api/v2/auth/users/{id}/roles/{role_id}",
            delete(unassign_role),
        )
        .route("/api/v2/auth/roles", post(create_role).get(list_roles))
        .route("/api/v2/auth/roles/{id}", delete(delete_role))
        .route(
            "/api/v2/auth/roles/{id}/permissions",
            post(add_permission).get(get_role_permissions),
        )
        .route(
            "/api/v2/auth/roles/{id}/permissions/{perm_id}",
            delete(remove_permission),
        )
        .route("/api/v2/auth/whoami", get(whoami))
}
