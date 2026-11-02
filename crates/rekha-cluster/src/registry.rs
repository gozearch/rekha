//! The catalog's durable contents, so a snapshot can carry them.
//!
//! A node that joins a cluster after the leader has compacted its log receives a
//! snapshot instead of the entries. Its state machine arrives with the placement
//! map, but the *catalog* — tenants, databases, collection rows, users, roles —
//! lives in redb and was, until this module, not part of any snapshot. The node
//! could answer nothing: routing reads a collection's row, and authorization
//! reads the caller's role.
//!
//! So a metadata-group snapshot carries this alongside the state machine, and
//! installing it rebuilds the catalog before the node accepts a request.
//!
//! # Scope
//!
//! Only the metadata group captures a registry. A shard group's snapshot carries
//! the data of its own collections, and the engine rebuilds those rows from the
//! archive itself — copying the registry into every shard snapshot would ship
//! user records to nodes that do not need them.

use rekha_core::auth::{KeyMeta, Permission, RoleAssignment, RoleRecord, UserRecord};
use rekha_storage::{Catalog, CollectionRecord};

/// Everything the metadata group needs to make a fresh node answerable.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RegistrySnapshot {
    /// Tenant names.
    pub tenants: Vec<String>,
    /// `(tenant, database)` pairs.
    pub databases: Vec<(String, String)>,
    /// Collection rows, including the shard each was placed on.
    pub collections: Vec<CollectionRecord>,
    /// Users.
    pub users: Vec<UserRecord>,
    /// `(user_id, key)` for every API key.
    pub keys: Vec<(String, KeyMeta)>,
    /// Roles.
    pub roles: Vec<RoleRecord>,
    /// `(role_id, permissions)`.
    pub permissions: Vec<(String, Vec<Permission>)>,
    /// Role assignments, including their audit fields.
    pub assignments: Vec<RoleAssignment>,
}

impl RegistrySnapshot {
    /// Read the current catalog into a snapshot.
    ///
    /// # Errors
    ///
    /// Returns the catalog's error rendered as text: this runs while building a
    /// Raft snapshot, where the only useful thing to do with a failure is to
    /// refuse to produce a snapshot that would restore a node into a partial
    /// registry.
    pub fn capture(catalog: &dyn Catalog) -> Result<Self, String> {
        let mut keys = Vec::new();
        let mut permissions = Vec::new();

        let users = catalog.list_users().map_err(|e| e.to_string())?;
        for user in &users {
            for key in catalog
                .list_user_keys(&user.id)
                .map_err(|e| e.to_string())?
            {
                keys.push((user.id.clone(), key));
            }
        }

        let roles = catalog.list_roles().map_err(|e| e.to_string())?;
        for role in &roles {
            let perms = catalog
                .get_role_permissions(&role.id)
                .map_err(|e| e.to_string())?;
            permissions.push((role.id.clone(), perms));
        }

        Ok(Self {
            tenants: catalog.list_tenants().map_err(|e| e.to_string())?,
            databases: catalog.list_databases().map_err(|e| e.to_string())?,
            collections: catalog.list_collections_all().map_err(|e| e.to_string())?,
            users,
            keys,
            roles,
            permissions,
            assignments: catalog.list_assignments().map_err(|e| e.to_string())?,
        })
    }

    /// Write this registry into a catalog.
    ///
    /// Idempotent: every write here is "create if absent", because the receiver
    /// may already hold some of these rows from entries it applied before the
    /// snapshot overtook it. A conflict is not an error — it is the normal case.
    ///
    /// # Errors
    ///
    /// Only a real storage failure is returned. `AlreadyExists` is swallowed for
    /// the reason above.
    pub fn restore(&self, catalog: &dyn Catalog) -> Result<(), String> {
        for tenant in &self.tenants {
            ignore_exists(catalog.create_tenant(tenant).map_err(|e| e.to_string()))?;
        }
        for (tenant, database) in &self.databases {
            ignore_exists(
                catalog
                    .create_database(tenant, database)
                    .map_err(|e| e.to_string()),
            )?;
        }
        for record in &self.collections {
            ignore_exists(
                catalog
                    .record_collection(&record.config, record.shard)
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
            )?;
        }
        // Roles before users' assignments, users before their keys.
        for role in &self.roles {
            ignore_exists(catalog.create_role(role).map_err(|e| e.to_string()))?;
        }
        for (role_id, perms) in &self.permissions {
            for perm in perms {
                ignore_exists(
                    catalog
                        .add_permission(role_id, perm)
                        .map_err(|e| e.to_string()),
                )?;
            }
        }
        for user in &self.users {
            ignore_exists(catalog.create_user(user).map_err(|e| e.to_string()))?;
        }
        for (user_id, key) in &self.keys {
            ignore_exists(
                catalog
                    .create_api_key(user_id, &key.key_hash, &key.name)
                    .map_err(|e| e.to_string()),
            )?;
        }
        for assignment in &self.assignments {
            ignore_exists(
                catalog
                    .assign_role(
                        &assignment.user_id,
                        &assignment.role_id,
                        &assignment.assigned_by,
                    )
                    .map_err(|e| e.to_string()),
            )?;
        }
        Ok(())
    }
}

/// Treat `already exists` as success, and anything else as a failure.
fn ignore_exists(result: Result<(), String>) -> Result<(), String> {
    match result {
        Ok(()) => Ok(()),
        Err(e) if e.contains("already exists") => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rekha_core::config::CollectionConfig;
    use rekha_core::types::Distance;
    use rekha_storage::RedbCatalog;

    fn catalog(dir: &std::path::Path) -> std::sync::Arc<RedbCatalog> {
        std::sync::Arc::new(RedbCatalog::open(dir.join("catalog.redb")).expect("catalog"))
    }

    /// The property this module exists for: capture, restore into an empty
    /// catalog, and get the same registry back.
    ///
    /// Compared as the facts authorization depends on, not as bytes. Three
    /// fields are regenerated by the receiver rather than carried: a
    /// permission's `id` and an assignment's `assigned_at_ms` (both node-local
    /// handles created when the row is written) and a key's `created_at_ms`.
    /// None of them is consulted when a request is authorized — `has_permission`
    /// matches on action, tenant and database — so carrying them would be
    /// copying a timestamp to make two nodes agree about something nobody reads.
    #[test]
    fn a_registry_survives_a_round_trip_into_an_empty_catalog() {
        let source_dir = tempfile::tempdir().expect("tempdir");
        let source = catalog(source_dir.path());
        populate(source.as_ref());

        let captured = RegistrySnapshot::capture(source.as_ref()).expect("capture");

        let target_dir = tempfile::tempdir().expect("tempdir");
        let target = catalog(target_dir.path());
        captured.restore(target.as_ref()).expect("restore");

        let after = RegistrySnapshot::capture(target.as_ref()).expect("recapture");
        assert_eq!(
            facts(&after),
            facts(&captured),
            "a restored registry must carry the same authorization facts"
        );
    }

    /// Restoring into a catalog that already holds some of it must not fail:
    /// that is the normal case for a node that applied entries before the
    /// snapshot overtook it.
    #[test]
    fn restoring_twice_is_not_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let catalog = catalog(dir.path());
        let snapshot = RegistrySnapshot {
            tenants: vec!["acme".into()],
            ..Default::default()
        };
        snapshot.restore(catalog.as_ref()).expect("first");
        snapshot.restore(catalog.as_ref()).expect("second");
        assert_eq!(
            catalog.list_tenants().expect("tenants"),
            vec!["acme".to_owned()]
        );
    }

    /// Create a catalog with one of everything.
    fn populate(catalog: &dyn Catalog) {
        catalog.create_tenant("acme").expect("tenant");
        catalog.create_database("acme", "prod").expect("database");
        let mut config = CollectionConfig::new("docs".into(), 3, Distance::L2);
        config.tenant = "acme".into();
        config.database = "prod".into();
        catalog
            .record_collection(&config, Some(rekha_core::cluster::ShardId(5)))
            .expect("collection");

        catalog.ensure_builtin_roles().expect("roles");
        let user = rekha_core::auth::UserRecord {
            id: "u1".into(),
            name: "operator".into(),
            enabled: true,
            created_at_ms: 1,
        };
        catalog.create_user(&user).expect("user");
        catalog.create_api_key("u1", &[7u8; 32], "ci").expect("key");
        // Built-in roles are created under generated ids; `ROLE_SUPERADMIN` is
        // the name.
        let superadmin = catalog
            .get_role_by_name(rekha_core::auth::ROLE_SUPERADMIN)
            .expect("lookup")
            .expect("built-in role");
        catalog
            .assign_role("u1", &superadmin.id, "test")
            .expect("assign");
    }

    /// What authorization reads of a permission: its action and its scope.
    type RoleActions = Vec<(String, Option<String>, Option<String>)>;

    /// The subset of a registry that authorization reads, as comparable text.
    fn facts(snapshot: &RegistrySnapshot) -> serde_json::Value {
        let mut permissions: Vec<(String, RoleActions)> = snapshot
            .permissions
            .iter()
            .map(|(role, perms)| {
                let mut actions: RoleActions = perms
                    .iter()
                    .map(|p| (p.action.clone(), p.tenant.clone(), p.database.clone()))
                    .collect();
                actions.sort();
                (role.clone(), actions)
            })
            .collect();
        permissions.sort();

        let mut keys: Vec<(String, String, [u8; 32])> = snapshot
            .keys
            .iter()
            .map(|(user, key)| (user.clone(), key.name.clone(), key.key_hash))
            .collect();
        keys.sort();

        let mut assignments: Vec<(String, String, String)> = snapshot
            .assignments
            .iter()
            .map(|a| (a.user_id.clone(), a.role_id.clone(), a.assigned_by.clone()))
            .collect();
        assignments.sort();

        serde_json::json!({
            "tenants": snapshot.tenants,
            "databases": snapshot.databases,
            "collections": snapshot.collections.iter().map(|c| {
                serde_json::json!({
                    "id": c.config.id,
                    "name": c.config.name,
                    "tenant": c.config.tenant,
                    "database": c.config.database,
                    "dimension": c.config.dimension,
                    "space": c.config.space.name(),
                    "shard": c.shard,
                })
            }).collect::<Vec<_>>(),
            "users": snapshot.users.iter().map(|u| {
                serde_json::json!({ "id": u.id, "name": u.name, "enabled": u.enabled })
            }).collect::<Vec<_>>(),
            "roles": snapshot.roles.iter().map(|r| {
                serde_json::json!({ "id": r.id, "name": r.name, "built_in": r.built_in })
            }).collect::<Vec<_>>(),
            "permissions": permissions,
            "keys": keys,
            "assignments": assignments,
        })
    }
}
