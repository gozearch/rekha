//! RBAC types: users, API keys, roles, permissions, and assignments.
//!
//! The RBAC model is **resource-scoped**: a permission is an `(action,
//! tenant?, database?)` tuple. A user holds one or more roles; each role
//! carries a set of permissions. Authorization checks whether any of the
//! caller's permissions match the required action and the resource's
//! tenant/database scope.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Hashing helpers (HMAC-SHA256) — used for API key storage.
// ---------------------------------------------------------------------------

/// Compute the HMAC-SHA256 of an API key using the server pepper.
///
/// Keys are never stored in plaintext; only the HMAC output is persisted.
/// The pepper MUST be kept secret — it is the second factor that makes a
/// stolen database dump useless without the server's env var.
///
/// If `pepper` is empty, the function falls back to plain SHA-256 (still
/// deterministic and indexable, but without the pepper boundary). This path
/// is only for tests that do not configure a pepper.
pub fn hash_api_key_with_pepper(key: &str, pepper: &[u8]) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    if pepper.is_empty() {
        use sha2::Digest;
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        let out = hasher.finalize();
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&out);
        return buf;
    }
    // HMAC accepts a key of any length by construction, so this cannot fail;
    // there is no input to validate and nothing to return an error to.
    #[allow(clippy::expect_used)]
    let mut mac = Hmac::<Sha256>::new_from_slice(pepper).expect("HMAC accepts any key length");
    mac.update(key.as_bytes());
    let out = mac.finalize().into_bytes();
    let mut buf = [0u8; 32];
    buf.copy_from_slice(&out);
    buf
}

/// Resolve the pepper from the standard env var.
pub fn pepper_from_env() -> Vec<u8> {
    std::env::var("REKHA_KEY_PEPPER")
        .map(|s| s.into_bytes())
        .unwrap_or_default()
}

/// Generate a random API key string with a `rekha_sk_` prefix.
pub fn generate_api_key() -> String {
    let id = uuid::Uuid::new_v4().to_string().replace('-', "");
    format!("rekha_sk_{id}")
}

// ---------------------------------------------------------------------------
// Core RBAC records
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserRecord {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyMeta {
    /// Human label for the key (e.g. "ci-key").
    pub name: String,
    /// 32-byte hash of the API key.
    pub key_hash: [u8; 32],
    pub created_at_ms: u64,
    pub last_used_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleRecord {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Built-in roles cannot be deleted.
    pub built_in: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permission {
    pub id: String,
    /// Action string, e.g. "collection.create", "record.query", or "*" for all.
    pub action: String,
    /// If `None`, the permission applies to all tenants.
    pub tenant: Option<String>,
    /// If `None`, the permission applies to all databases within the tenant
    /// scope. Ignored when `tenant` is `None`.
    pub database: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleAssignment {
    pub user_id: String,
    pub role_id: String,
    pub assigned_at_ms: u64,
    pub assigned_by: String,
}

// ---------------------------------------------------------------------------
// Built-in roles
// ---------------------------------------------------------------------------

pub const ROLE_SUPERADMIN: &str = "superadmin";
pub const ROLE_TENANT_ADMIN: &str = "tenant_admin";
pub const ROLE_WRITER: &str = "writer";
pub const ROLE_READER: &str = "reader";

/// All built-in role names.
pub const BUILTIN_ROLES: &[&str] = &[ROLE_SUPERADMIN, ROLE_TENANT_ADMIN, ROLE_WRITER, ROLE_READER];

/// Deterministic identifier for a built-in role.
///
/// Built-in roles are materialised from this table on every node rather than
/// proposed through the log, and a snapshot carries the rows. A random UUID per
/// node would make the same role a different row everywhere, so an `AssignRole`
/// entry committed by the leader names an id that exists on the leader and
/// nowhere else — and every follower drops the assignment as "role not found".
/// Deriving the id from the name makes one role one row on every node.
pub fn builtin_role_id(name: &str) -> String {
    name_based_uuid(&format!("rekha/role/{name}"))
}

/// A UUID derived from a name.
///
/// Same shape and intent as a v5 UUID, built from SHA-256 because that digest is
/// already a dependency here and SHA-1 is not. The version and variant bits are
/// set so the string parses as a well-formed UUID on the client side.
fn name_based_uuid(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(input.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    uuid::Uuid::from_bytes(bytes).to_string()
}

/// Deterministic identifier for one permission of a built-in role, for the same
/// reason: `RemovePermission` names a permission id, and it has to mean the same
/// thing on every node.
fn builtin_permission_id(
    role: &str,
    action: &str,
    tenant: Option<&str>,
    database: Option<&str>,
) -> String {
    name_based_uuid(&format!(
        "rekha/permission/{role}/{action}/{}/{}",
        tenant.unwrap_or("*"),
        database.unwrap_or("*")
    ))
}

/// Return the default permissions for a built-in role.
///
/// `tenant` and `database` are used for tenant/database-scoped roles; they
/// are `None` for `superadmin`.
pub fn builtin_permissions(
    role: &str,
    tenant: Option<&str>,
    database: Option<&str>,
) -> Vec<Permission> {
    let mk = |action: &str| Permission {
        id: builtin_permission_id(role, action, tenant, database),
        action: action.to_string(),
        tenant: tenant.map(|t| t.to_owned()),
        database: database.map(|d| d.to_owned()),
    };
    match role {
        ROLE_SUPERADMIN => vec![mk("*")],
        ROLE_TENANT_ADMIN => vec![
            mk("collection.create"),
            mk("collection.delete"),
            mk("collection.list"),
            mk("collection.get"),
            mk("record.add"),
            mk("record.upsert"),
            mk("record.update"),
            mk("record.delete"),
            mk("record.query"),
            mk("record.get"),
            mk("record.count"),
            mk("tenant.create"),
            mk("tenant.delete"),
            mk("database.create"),
            mk("database.delete"),
            mk("tenant.get"),
            mk("database.list"),
            mk("database.get"),
        ],
        ROLE_WRITER => vec![
            mk("collection.create"),
            mk("collection.delete"),
            mk("collection.list"),
            mk("collection.get"),
            mk("record.add"),
            mk("record.upsert"),
            mk("record.update"),
            mk("record.delete"),
            mk("record.query"),
            mk("record.get"),
            mk("record.count"),
        ],
        ROLE_READER => vec![
            mk("collection.list"),
            mk("collection.get"),
            mk("record.query"),
            mk("record.get"),
            mk("record.count"),
            mk("tenant.get"),
            mk("database.list"),
            mk("database.get"),
        ],
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Permission matching
// ---------------------------------------------------------------------------

/// Whether `perm` grants `required_action` on `resource`.
///
/// - `*` matches every action.
/// - A permission with `tenant = None` matches any tenant.
/// - A permission with `tenant = Some(t)` but `database = None` matches any
///   database within tenant `t`.
/// - Otherwise both tenant and database must match.
pub fn permission_matches(
    perm: &Permission,
    required_action: &str,
    resource_tenant: Option<&str>,
    resource_database: Option<&str>,
) -> bool {
    if perm.action != "*" && perm.action != required_action {
        return false;
    }
    match (&perm.tenant, resource_tenant) {
        (None, _) => true,
        (Some(pt), Some(rt)) if pt == rt => match (&perm.database, resource_database) {
            (None, _) => true,
            (Some(pd), Some(rd)) if pd == rd => true,
            _ => false,
        },
        _ => false,
    }
}

/// Whether any permission in `perms` grants `required_action` on the resource.
pub fn has_permission(
    perms: &[Permission],
    required_action: &str,
    resource_tenant: Option<&str>,
    resource_database: Option<&str>,
) -> bool {
    perms
        .iter()
        .any(|p| permission_matches(p, required_action, resource_tenant, resource_database))
}

// ---------------------------------------------------------------------------
// Route → action mapping
// ---------------------------------------------------------------------------

/// What authorization a route requires.
///
/// The important variant is [`RouteAction::Unmatched`]. The previous signature
/// returned `Option`, and the middleware only checked permissions `if let
/// Some(..)`, so **any route this function did not recognise was allowed for
/// every authenticated caller**. `/optimize` was one such route until it was
/// patched by hand. An allow-list that fails open is not an allow-list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteAction {
    /// No credentials required: liveness and readiness only.
    Public,
    /// Credentials required, no specific permission (identity endpoints).
    AuthenticatedOnly,
    /// Credentials required plus this action, scoped to tenant/database when
    /// the path names them.
    Permission {
        action: String,
        tenant: Option<String>,
        database: Option<String>,
    },
    /// Not a route this build knows. Denied rather than skipped.
    Unmatched,
}

/// Map an HTTP `(method, path)` pair to the authorization it requires.
///
/// **Paths are parsed by segment, never by substring.** The earlier version
/// asked `path.contains("/query")` over the whole path, and the collection name
/// is part of the path: a collection named `get` turned `POST .../get/add` into
/// a `record.get` check, so a read-only permission authorised a write, and a
/// collection named `update` let `record.update` authorise a delete.
pub fn route_action(method: &str, path: &str) -> RouteAction {
    // HEAD is served by the GET route in axum; classify it as such so a HEAD
    // probe does not read as an unknown path.
    let method = if method == "HEAD" { "GET" } else { method };

    if matches!(
        (method, path),
        ("GET", "/health")
            | ("GET", "/ready")
            | ("GET", "/api/v2/heartbeat")
            | ("GET", "/api/v2/version")
            | ("GET", "/api/v2/pre-flight-checks")
    ) {
        return RouteAction::Public;
    }

    if path == "/api/v2/auth/whoami" || path == "/api/v2/auth/identity" {
        return RouteAction::AuthenticatedOnly;
    }

    if let Some(found) = chroma_route(method, path) {
        return found;
    }
    if let Some(found) = native_route(method, path) {
        return found;
    }
    RouteAction::Unmatched
}

fn action(action: &str) -> RouteAction {
    RouteAction::Permission {
        action: action.to_string(),
        tenant: None,
        database: None,
    }
}

fn scoped(action: &str, tenant: Option<String>, database: Option<String>) -> RouteAction {
    RouteAction::Permission {
        action: action.to_string(),
        tenant,
        database,
    }
}

/// `/api/v2/tenants/{t}/databases/{d}/...` — Chroma compatibility surface.
///
/// The shape is fixed, so the collection name sits at a known index and is
/// never consulted. Segment counts are explicit for the same reason.
fn chroma_route(method: &str, path: &str) -> Option<RouteAction> {
    if !path.starts_with("/api/v2/tenants/") {
        return None;
    }
    let parts: Vec<&str> = path.trim_matches('/').split('/').collect();
    // ["api","v2","tenants",T,"databases",D,...]
    if parts.len() < 4 || parts[0] != "api" || parts[1] != "v2" || parts[2] != "tenants" {
        return None;
    }
    let tenant = parts[3].to_string();

    // /api/v2/tenants/{t}
    if parts.len() == 4 {
        return Some(match method {
            "GET" => scoped("tenant.get", Some(tenant), None),
            "POST" | "PUT" => scoped("tenant.create", Some(tenant), None),
            "DELETE" => scoped("tenant.delete", Some(tenant), None),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    if parts[4] != "databases" {
        return None;
    }
    // /api/v2/tenants/{t}/databases
    if parts.len() == 5 {
        return Some(match method {
            "GET" => scoped("database.list", Some(tenant), None),
            "POST" => scoped("database.create", Some(tenant), None),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    let database = parts[5].to_string();
    // /api/v2/tenants/{t}/databases/{d}
    if parts.len() == 6 {
        return Some(match method {
            "GET" => scoped("database.get", Some(tenant), Some(database)),
            "DELETE" => scoped("database.delete", Some(tenant), Some(database)),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    if parts[6] != "collections" {
        return None;
    }
    // /api/v2/tenants/{t}/databases/{d}/collections
    if parts.len() == 7 {
        return Some(match method {
            "GET" => scoped("collection.list", Some(tenant), Some(database)),
            "POST" => scoped("collection.create", Some(tenant), Some(database)),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    // .../collections/{name} — parts[7] is the name and is deliberately unused
    // for classification.
    let _name = parts[7];
    if parts.len() == 8 {
        return Some(match method {
            "GET" => scoped("collection.get", Some(tenant), Some(database)),
            "PUT" | "POST" => scoped("collection.update", Some(tenant), Some(database)),
            "DELETE" => scoped("collection.delete", Some(tenant), Some(database)),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    if parts.len() == 9 {
        let action = match parts[8] {
            "add" => "record.add",
            "upsert" => "record.upsert",
            "update" => "record.update",
            "delete" => "record.delete",
            "query" => "record.query",
            "get" => "record.get",
            "count" => "record.count",
            "optimize" => "collection.optimize",
            _ => return Some(RouteAction::Unmatched),
        };
        return Some(match method {
            "POST" if action != "record.count" => scoped(action, Some(tenant), Some(database)),
            "GET" if action == "record.count" => scoped(action, Some(tenant), Some(database)),
            _ => return Some(RouteAction::Unmatched),
        });
    }
    Some(RouteAction::Unmatched)
}

/// Native `/api/v2/...` routes. Collection ids are UUIDs, so splitting on the
/// single slash after the id is unambiguous.
fn native_route(method: &str, path: &str) -> Option<RouteAction> {
    if path == "/api/v2/collections" {
        return Some(action(match method {
            "POST" => "collection.create",
            "GET" => "collection.list",
            _ => return Some(RouteAction::Unmatched),
        }));
    }
    if path == "/api/v2/tenants" {
        return Some(action(match method {
            "POST" => "tenant.create",
            _ => return Some(RouteAction::Unmatched),
        }));
    }
    // `/api/v2/tenants/{tenant}` and `/api/v2/tenants/{tenant}/databases[...]`.
    // The collection surface under the same prefix is classified by
    // `chroma_route` first, so it never reaches here.
    if let Some(rest) = path.strip_prefix("/api/v2/tenants/") {
        let segments: Vec<&str> = rest.split('/').collect();
        return Some(match (method, segments.as_slice()) {
            ("GET", [tenant]) => scoped("tenant.get", Some((*tenant).to_string()), None),
            ("GET", [tenant, "databases"]) => {
                scoped("database.list", Some((*tenant).to_string()), None)
            }
            ("POST", [tenant, "databases"]) => {
                scoped("database.create", Some((*tenant).to_string()), None)
            }
            ("GET", [tenant, "databases", database]) => scoped(
                "database.get",
                Some((*tenant).to_string()),
                Some((*database).to_string()),
            ),
            _ => RouteAction::Unmatched,
        });
    }
    if path.starts_with("/api/v2/auth/") {
        return Some(auth_route(method, path));
    }

    let rest = path.strip_prefix("/api/v2/collections/")?;
    match rest.split_once('/') {
        // /api/v2/collections/{id}
        None => Some(action(match method {
            "GET" => "collection.get",
            "DELETE" => "collection.delete",
            _ => return Some(RouteAction::Unmatched),
        })),
        // /api/v2/collections/{id}/{operation}
        Some((_id, operation)) => {
            let act = match operation {
                "add" => "record.add",
                "upsert" => "record.upsert",
                "update" => "record.update",
                "delete" => "record.delete",
                "query" => "record.query",
                "get" => "record.get",
                "count" => "record.count",
                "optimize" => "collection.optimize",
                _ => return Some(RouteAction::Unmatched),
            };
            // Every native collection operation is a POST, including `count`.
            // Only the Chroma surface uses GET for count.
            if method != "POST" {
                return Some(RouteAction::Unmatched);
            }
            Some(action(act))
        }
    }
}

/// RBAC administration routes.
fn auth_route(method: &str, path: &str) -> RouteAction {
    let act = match (method, path) {
        ("POST", "/api/v2/auth/users") => "user.create",
        ("GET", "/api/v2/auth/users") => "user.list",
        ("DELETE", p)
            if p.starts_with("/api/v2/auth/users/")
                && !p.contains("/roles")
                && !p.contains("/keys") =>
        {
            "user.delete"
        }
        ("GET", p) if p.starts_with("/api/v2/auth/users/") => "user.list",
        ("POST", "/api/v2/auth/roles") => "role.create",
        ("GET", "/api/v2/auth/roles") => "role.list",
        ("DELETE", p) if p.starts_with("/api/v2/auth/roles/") && !p.contains("/permissions") => {
            "role.delete"
        }
        ("GET", p) if p.starts_with("/api/v2/auth/roles/") => "role.list",
        ("POST", p) if p.contains("/permissions") => "role.manage_permissions",
        ("DELETE", p) if p.contains("/permissions") => "role.manage_permissions",
        ("POST", p) if p.contains("/roles") => "role.assign",
        ("DELETE", p) if p.contains("/roles") => "role.assign",
        ("POST", p) if p.contains("/keys") => "user.manage_keys",
        ("DELETE", p) if p.contains("/keys") => "user.manage_keys",
        _ => return RouteAction::Unmatched,
    };
    action(act)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_global_matches_all() {
        let p = Permission {
            id: "1".into(),
            action: "record.query".into(),
            tenant: None,
            database: None,
        };
        assert!(permission_matches(&p, "record.query", Some("t"), Some("d")));
        assert!(permission_matches(&p, "record.query", None, None));
        assert!(!permission_matches(&p, "record.add", Some("t"), Some("d")));
    }

    #[test]
    fn permission_tenant_scoped() {
        let p = Permission {
            id: "1".into(),
            action: "record.query".into(),
            tenant: Some("acme".into()),
            database: None,
        };
        assert!(permission_matches(
            &p,
            "record.query",
            Some("acme"),
            Some("prod")
        ));
        assert!(!permission_matches(
            &p,
            "record.query",
            Some("other"),
            Some("prod")
        ));
        assert!(!permission_matches(&p, "record.query", None, None));
    }

    #[test]
    fn permission_database_scoped() {
        let p = Permission {
            id: "1".into(),
            action: "record.query".into(),
            tenant: Some("acme".into()),
            database: Some("prod".into()),
        };
        assert!(permission_matches(
            &p,
            "record.query",
            Some("acme"),
            Some("prod")
        ));
        assert!(!permission_matches(
            &p,
            "record.query",
            Some("acme"),
            Some("staging")
        ));
        assert!(!permission_matches(
            &p,
            "record.query",
            Some("other"),
            Some("prod")
        ));
    }

    #[test]
    fn permission_wildcard_action() {
        let p = Permission {
            id: "1".into(),
            action: "*".into(),
            tenant: None,
            database: None,
        };
        assert!(permission_matches(&p, "record.query", Some("t"), Some("d")));
        assert!(permission_matches(&p, "anything", None, None));
    }

    #[test]
    fn route_action_health_is_public() {
        assert_eq!(route_action("GET", "/health"), RouteAction::Public);
        assert_eq!(route_action("GET", "/ready"), RouteAction::Public);
        assert_eq!(
            route_action("GET", "/api/v2/heartbeat"),
            RouteAction::Public
        );
    }

    /// The defect this mapping replaced: the collection *name* is part of the
    /// path, and the old classifier asked `path.contains("/query")` over the
    /// whole path. A collection named `get` made an add require only
    /// `record.get`, and one named `update` made a delete require only
    /// `record.update`.
    #[test]
    fn a_collection_name_never_chooses_the_required_permission() {
        for (name, operation, expected) in [
            ("get", "add", "record.add"),
            ("query", "add", "record.add"),
            ("update", "delete", "record.delete"),
            ("count", "upsert", "record.upsert"),
            ("delete", "query", "record.query"),
            ("add", "optimize", "collection.optimize"),
        ] {
            let path =
                format!("/api/v2/tenants/acme/databases/prod/collections/{name}/{operation}");
            let RouteAction::Permission { action: found, .. } = route_action("POST", &path) else {
                panic!("{path} should require a permission");
            };
            assert_eq!(found, expected, "misclassified `{path}`");
        }
    }

    /// An unrecognised route is denied, not skipped. Skipping was how `/optimize`
    /// went unprotected until it was patched by hand.
    #[test]
    fn unknown_routes_are_denied_rather_than_skipped() {
        assert_eq!(
            route_action("DELETE", "/api/v2/nonsense"),
            RouteAction::Unmatched
        );
        assert_eq!(
            route_action("POST", "/api/v2/collections/abc/unknown-op"),
            RouteAction::Unmatched
        );
        assert_eq!(
            route_action(
                "POST",
                "/api/v2/tenants/t/databases/d/collections/c/nonsense"
            ),
            RouteAction::Unmatched
        );
        // A method that does not exist on a real route is denied too.
        assert_eq!(
            route_action("PUT", "/api/v2/collections/abc/add"),
            RouteAction::Unmatched
        );
    }

    /// Tenant and database administration used to return `None` from the Chroma
    /// branch, which meant any authenticated key could create or delete them.
    #[test]
    fn tenant_and_database_routes_require_a_permission() {
        for (method, path, expected) in [
            ("POST", "/api/v2/tenants", "tenant.create"),
            ("GET", "/api/v2/tenants/acme", "tenant.get"),
            ("GET", "/api/v2/tenants/acme/databases", "database.list"),
            ("POST", "/api/v2/tenants/acme/databases", "database.create"),
            ("GET", "/api/v2/tenants/acme/databases/prod", "database.get"),
            (
                "GET",
                "/api/v2/tenants/acme/databases/prod/collections",
                "collection.list",
            ),
            (
                "POST",
                "/api/v2/tenants/acme/databases/prod/collections",
                "collection.create",
            ),
        ] {
            let RouteAction::Permission { action: found, .. } = route_action(method, path) else {
                panic!("{method} {path} should require a permission");
            };
            assert_eq!(found, expected, "{method} {path}");
        }
    }

    #[test]
    fn head_is_classified_as_get() {
        assert_eq!(route_action("HEAD", "/health"), RouteAction::Public);
        assert_eq!(
            route_action("HEAD", "/api/v2/tenants/t/databases/d/collections/c/count"),
            scoped("record.count", Some("t".into()), Some("d".into()))
        );
    }

    #[test]
    fn identity_endpoints_need_credentials_but_no_permission() {
        assert_eq!(
            route_action("GET", "/api/v2/auth/whoami"),
            RouteAction::AuthenticatedOnly
        );
        // The Chroma client probes this on connect.
        assert_eq!(
            route_action("GET", "/api/v2/auth/identity"),
            RouteAction::AuthenticatedOnly
        );
    }

    #[test]
    fn route_action_collection_create() {
        assert_eq!(
            route_action("POST", "/api/v2/collections"),
            action("collection.create")
        );
    }

    #[test]
    fn route_action_record_query() {
        assert_eq!(
            route_action("POST", "/api/v2/collections/abc/query"),
            action("record.query")
        );
    }

    #[test]
    fn route_action_chroma_scoped() {
        let RouteAction::Permission {
            action: found,
            tenant,
            database,
        } = route_action(
            "POST",
            "/api/v2/tenants/acme/databases/prod/collections/my_coll/query",
        )
        else {
            panic!("expected a permission check");
        };
        assert_eq!(found, "record.query");
        assert_eq!(tenant, Some("acme".to_string()));
        assert_eq!(database, Some("prod".to_string()));
    }

    #[test]
    fn builtin_roles_have_expected_actions() {
        let perms = builtin_permissions(ROLE_READER, Some("t"), Some("d"));
        assert!(perms.iter().any(|p| p.action == "record.query"));
        assert!(!perms.iter().any(|p| p.action == "record.add"));

        let perms = builtin_permissions(ROLE_SUPERADMIN, None, None);
        assert!(perms.iter().any(|p| p.action == "*"));
    }

    #[test]
    fn builtin_ids_are_stable_across_processes_and_distinct_per_role() {
        // Built-in rows are materialised on every node instead of being written
        // through the log, so these ids are the only thing that makes the same
        // role the same row everywhere. A random id would make a committed
        // `AssignRole` reference a role that exists on the leader alone.
        assert_eq!(
            builtin_role_id(ROLE_SUPERADMIN),
            builtin_role_id(ROLE_SUPERADMIN)
        );
        assert_ne!(
            builtin_role_id(ROLE_SUPERADMIN),
            builtin_role_id(ROLE_READER)
        );
        assert!(
            uuid::Uuid::parse_str(&builtin_role_id(ROLE_WRITER)).is_ok(),
            "the id is used as a key and returned to clients, so it must parse"
        );

        let first = builtin_permissions(ROLE_READER, None, None);
        let second = builtin_permissions(ROLE_READER, None, None);
        let ids = |perms: &[Permission]| perms.iter().map(|p| p.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&first), ids(&second));

        // A tenant-scoped grant is a different permission, not a copy of the
        // global one, or removing the scoped grant would remove the global.
        let scoped = builtin_permissions(ROLE_READER, Some("acme"), None);
        assert_ne!(first[0].id, scoped[0].id);
    }

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_api_key_with_pepper("hello", &[]);
        let h2 = hash_api_key_with_pepper("hello", &[]);
        assert_eq!(h1, h2);
        assert_ne!(h1, hash_api_key_with_pepper("world", &[]));
    }

    #[test]
    fn generate_key_has_prefix() {
        let k = generate_api_key();
        assert!(k.starts_with("rekha_sk_"));
    }
}
