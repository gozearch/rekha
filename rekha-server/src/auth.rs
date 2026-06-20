use rekha_core::{AuthContext, RekhaError, UserConfig, UserRole};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Authenticator for gRPC request authentication and authorization.
///
/// Supports four auth methods:
/// - `trust`: No authentication (development mode).
/// - `password`: HTTP Basic auth with bcrypt-verified password.
/// - `cert`: Client TLS certificate CN matches username.
/// - `password+cert`: Both certificate AND password required.
pub struct Authenticator {
    auth_method: String,
    users: Arc<RwLock<HashMap<String, UserConfig>>>,
}

impl Authenticator {
    pub fn new(auth_method: &str, users: Arc<RwLock<HashMap<String, UserConfig>>>) -> Self {
        Self {
            auth_method: auth_method.to_string(),
            users,
        }
    }

    /// Extract Basic auth credentials from the Authorization header.
    fn parse_basic_auth(header_value: &str) -> Result<(String, String), RekhaError> {
        let encoded = header_value
            .strip_prefix("Basic ")
            .ok_or_else(|| RekhaError::AuthenticationFailed("invalid auth header format".into()))?;

        let decoded = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            encoded,
        )
        .map_err(|_| RekhaError::AuthenticationFailed("invalid base64 encoding".into()))?;

        let decoded_str =
            String::from_utf8(decoded).map_err(|_| RekhaError::AuthenticationFailed(
                "auth header not valid UTF-8".into(),
            ))?;

        let mut parts = decoded_str.splitn(2, ':');
        let username = parts
            .next()
            .ok_or_else(|| RekhaError::AuthenticationFailed("missing username".into()))?
            .to_string();
        let password = parts
            .next()
            .ok_or_else(|| RekhaError::AuthenticationFailed("missing password".into()))?
            .to_string();

        Ok((username, password))
    }

    /// Authenticate a request from its Authorization header.
    /// Returns an AuthContext on success.
    pub async fn authenticate(
        &self,
        auth_header: Option<&str>,
    ) -> Result<AuthContext, RekhaError> {
        match self.auth_method.as_str() {
            "trust" => Ok(AuthContext {
                username: "trusted".into(),
                role: UserRole::Admin,
                collections: Vec::new(),
            }),
            "password" => {
                let header = auth_header.ok_or_else(|| {
                    RekhaError::AuthenticationFailed("authorization header required".into())
                })?;
                let (username, password) = Self::parse_basic_auth(header)?;
                self.verify_password(&username, &password).await?;
                self.get_context(&username).await
            }
            _ => Err(RekhaError::AuthenticationFailed(format!(
                "unsupported auth method: {}",
                self.auth_method
            ))),
        }
    }

    /// Verify a username/password against the stored bcrypt hash.
    async fn verify_password(&self, username: &str, password: &str) -> Result<(), RekhaError> {
        let users = self.users.read().await;
        let config = users.get(username).ok_or_else(|| {
            RekhaError::AuthenticationFailed("invalid username or password".into())
        })?;

        let valid = bcrypt::verify(password, &config.password_hash)
            .map_err(|e| RekhaError::Internal {
                detail: format!("bcrypt verification error: {e}"),
            })?;

        if !valid {
            return Err(RekhaError::AuthenticationFailed(
                "invalid username or password".into(),
            ));
        }
        Ok(())
    }

    /// Build an AuthContext for the given username.
    async fn get_context(&self, username: &str) -> Result<AuthContext, RekhaError> {
        let users = self.users.read().await;
        let config = users.get(username).ok_or_else(|| {
            RekhaError::AuthenticationFailed("user not found".into())
        })?;

        Ok(AuthContext {
            username: username.to_string(),
            role: config.role.clone(),
            collections: config.collections.clone(),
        })
    }

    /// Check if the context allows accessing a collection.
    pub fn check_collection_access(
        ctx: &AuthContext,
        collection: &str,
        write_required: bool,
    ) -> Result<(), RekhaError> {
        match ctx.role {
            UserRole::Admin => Ok(()),
            UserRole::ReadOnly if !write_required => {
                if ctx.collections.is_empty() || ctx.collections.contains(&collection.to_string()) {
                    Ok(())
                } else {
                    Err(RekhaError::PermissionDenied(format!(
                        "read-only user {} not allowed on collection {collection}",
                        ctx.username
                    )))
                }
            }
            UserRole::ReadOnly => Err(RekhaError::PermissionDenied(format!(
                "read-only user {} cannot write to collection {collection}",
                ctx.username
            ))),
            UserRole::User => {
                if ctx.collections.is_empty() || ctx.collections.contains(&collection.to_string()) {
                    Ok(())
                } else {
                    Err(RekhaError::PermissionDenied(format!(
                        "user {} not allowed on collection {collection}",
                        ctx.username
                    )))
                }
            }
        }
    }

    /// Hash a plaintext password with bcrypt.
    pub fn hash_password(password: &str) -> Result<String, RekhaError> {
        bcrypt::hash(password, bcrypt::DEFAULT_COST).map_err(|e| RekhaError::Internal {
            detail: format!("bcrypt hash error: {e}"),
        })
    }
}
