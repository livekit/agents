//! Where agentstore is: every worker (Cloud included) reaches it through LiveKit's public API,
//! threads and databases with `livekit-api`'s `ThreadClient` and `DatabaseClient` (project `p_`,
//! the token's), the data plane (`/v0/agentstore/db`) and the streams (`/v0/agentstore/streams`)
//! under the same base URL. Every request carries a token minted from the project's key pair.

use std::sync::Arc;

use livekit_api::services::databases::DatabaseClient;
use livekit_api::services::{ServerError, ServiceError, public_api_url};
use livekit_protocol::agent_db;
use lk_agents_host_derive::Host;
use serde::{Deserialize, Serialize};

use crate::StorageError;
use crate::threads::{duration_of, seconds_of};

/// The data plane under the base URL, a WebSocket.
pub const DB_PATH: &str = "/v0/agentstore/db";
/// The answer streams under the base URL.
pub const STREAMS_PATH: &str = "/v0/agentstore/streams";

/// The public API of `LIVEKIT_URL`'s project.
pub fn agentstore_url_from_env() -> Result<String, StorageError> {
    match std::env::var("LIVEKIT_URL") {
        Ok(u) if !u.is_empty() => Ok(public_api_url(&u)),
        _ => Err(StorageError::Invalid("agentstore needs LIVEKIT_URL".into())),
    }
}

/// A `livekit-api` failure as a storage error: `not_found` is `NotFound`, `unauthenticated` and
/// `permission_denied` `Unauthorized`, a lost request and the retryable codes `Unavailable`,
/// anything else `Server` with its code.
pub(crate) fn error(e: ServiceError) -> StorageError {
    match e {
        ServiceError::Twirp(ServerError::Request(e)) => StorageError::Unavailable(e.to_string()),
        ServiceError::Twirp(ServerError::Twirp(c)) => match c.code.as_str() {
            "not_found" => StorageError::NotFound(c.msg),
            "unauthenticated" | "permission_denied" => StorageError::Unauthorized(c.msg),
            "unavailable" | "deadline_exceeded" | "resource_exhausted" => {
                StorageError::Unavailable(c.msg)
            }
            _ => StorageError::Server { code: c.code, message: c.msg },
        },
        ServiceError::Twirp(e) => StorageError::Decode(e.to_string()),
        other => StorageError::Invalid(other.to_string()),
    }
}

/// A database made through `DatabaseService.CreateDatabase`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Host)]
#[python(pydantic, module = "storage", config = "frozen=True")]
pub struct DatabaseInfo {
    /// `DB_..`.
    pub database_id: String,
    /// Unix seconds; `0` for never.
    pub expires_at: i64,
}

/// Create, get and delete a database. A thread's database is made with the thread; this is for
/// development and tests.
#[derive(Clone)]
pub struct Databases {
    client: Arc<DatabaseClient>,
}

impl Databases {
    pub(crate) fn new(client: DatabaseClient) -> Self {
        Self { client: Arc::new(client) }
    }

    /// A database in the project's data region, `ttl_seconds` 0 for never.
    pub async fn create_database(&self, ttl_seconds: i64) -> Result<DatabaseInfo, StorageError> {
        let req = agent_db::CreateRequest { ttl: duration_of(ttl_seconds) };
        let r = self.client.create(req).await.map_err(error)?;
        Ok(DatabaseInfo { database_id: r.database_id, expires_at: seconds_of(r.expires_at) })
    }

    pub async fn get_database(
        &self,
        database_id: &str,
    ) -> Result<agent_db::AgentDatabase, StorageError> {
        self.client.get(database_id).await.map_err(error)
    }

    pub async fn delete_database(&self, database_id: &str) -> Result<(), StorageError> {
        self.client.delete(database_id).await.map_err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_service_error_keeps_its_code() {
        use livekit_api::services::ServerErrorCode;
        let twirp = |code: &str| {
            ServiceError::Twirp(ServerError::Twirp(ServerErrorCode {
                code: code.into(),
                msg: "m".into(),
                meta: Default::default(),
            }))
        };
        assert!(matches!(error(twirp("not_found")), StorageError::NotFound(_)));
        assert!(matches!(error(twirp("permission_denied")), StorageError::Unauthorized(_)));
        assert!(matches!(error(twirp("unavailable")), StorageError::Unavailable(_)));
        assert!(matches!(
            error(twirp("already_exists")),
            StorageError::Server { code, .. } if code == "already_exists"
        ));
    }
}
