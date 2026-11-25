use crate::prelude::*;

/// Whether `perms` allows `action` at this tenant/database scope.
///
/// The three checks are the permission ladder the model defines: a grant at
/// the exact namespace, a grant at the tenant level, and a global grant. They
/// were written out at each call site; keeping the ladder in one place means a
/// new route cannot implement two of the three and look authorised.
pub(crate) fn require_scope_permission(
    perms: &[rekha_core::auth::Permission],
    action: &str,
    tenant: &str,
    database: &str,
) -> Result<(), EngineError> {
    let allowed = rekha_core::auth::has_permission(perms, action, Some(tenant), Some(database))
        || rekha_core::auth::has_permission(perms, action, Some(tenant), None)
        || rekha_core::auth::has_permission(perms, action, None, None);
    if allowed {
        Ok(())
    } else {
        Err(EngineError::Forbidden(format!(
            "missing permission {action} on {tenant}/{database}"
        )))
    }
}

/// Helper to check per-collection access for native v2 routes.
/// Loads the collection's tenant/database and verifies the caller's permissions.
pub(crate) fn check_collection_access(
    catalog: &dyn rekha_storage::Catalog,
    perms: &[rekha_core::auth::Permission],
    action: &str,
    collection_id: &uuid::Uuid,
) -> Result<(), EngineError> {
    let record = catalog
        .get_collection_by_id(collection_id)
        .map_err(EngineError::Catalog)?
        .ok_or_else(|| EngineError::CollectionNotFound(collection_id.to_string()))?;
    require_scope_permission(
        perms,
        action,
        record.config.tenant.as_str(),
        record.config.database.as_str(),
    )
}
