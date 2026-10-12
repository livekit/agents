//! The worker's agentstore credentials (plan 13.6): HS256 JWTs signed with the project's API key
//! and secret, a one hour lifetime, minted per dial so a token never expires mid-outage. agent-db
//! takes `agent.databaseAdmin`; the threads service takes only an `agentThread` grant, here one
//! covering every thread of the project.
//!
//! Signed here rather than with `livekit-token`, whose `Claims` model `video`/`sip` and not the
//! `agent` container (`protocol/auth/grants.go`'s `ClaimGrants.Agent`).

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use hmac::{Hmac, Mac as _};
use sha2::Sha256;

use crate::StorageError;

/// Plan 13.4: a one hour TTL.
pub const TOKEN_TTL: Duration = Duration::from_secs(3600);

/// The API key and secret, held for the life of the client; a token is minted on every dial.
#[derive(Clone)]
pub struct TokenMinter {
    api_key: String,
    api_secret: String,
    threads: bool,
}

impl std::fmt::Debug for TokenMinter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenMinter").field("api_key", &self.api_key).finish_non_exhaustive()
    }
}

impl TokenMinter {
    pub fn new(
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let (api_key, api_secret) = (api_key.into(), api_secret.into());
        if api_key.is_empty() || api_secret.is_empty() {
            return Err(StorageError::Invalid(
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required".into(),
            ));
        }
        Ok(Self { api_key, api_secret, threads: false })
    }

    /// `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`.
    pub fn from_env() -> Result<Self, StorageError> {
        Self::new(
            std::env::var("LIVEKIT_API_KEY").unwrap_or_default(),
            std::env::var("LIVEKIT_API_SECRET").unwrap_or_default(),
        )
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// The same key, minting the threads service's grant instead of agent-db's.
    pub fn for_threads(&self) -> Self {
        Self { threads: true, ..self.clone() }
    }

    /// A fresh token: `iss` the key, `nbf` now, `exp` now + [`TOKEN_TTL`], and the grant.
    pub fn mint(&self) -> Result<String, StorageError> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| StorageError::Invalid(e.to_string()))?
            .as_secs();
        self.mint_at(now)
    }

    fn mint_at(&self, now: u64) -> Result<String, StorageError> {
        let mut claims = serde_json::json!({
            "iss": self.api_key,
            "nbf": now,
            "exp": now + TOKEN_TTL.as_secs(),
        });
        if self.threads {
            claims["agentThread"] = serde_json::json!({
                "all": true, "list": true, "create": true, "read": true, "write": true, "delete": true,
            });
        } else {
            claims["agent"] = serde_json::json!({"databaseAdmin": true});
        }
        let head = URL_SAFE_NO_PAD.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let body = URL_SAFE_NO_PAD.encode(claims.to_string());
        let signing_input = format!("{head}.{body}");
        let mut mac = Hmac::<Sha256>::new_from_slice(self.api_secret.as_bytes())
            .map_err(|e| StorageError::Invalid(e.to_string()))?;
        mac.update(signing_input.as_bytes());
        let sig = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());
        Ok(format!("{signing_input}.{sig}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_token_carries_the_database_admin_grant_and_verifies() {
        let minter = TokenMinter::new("devkey", "secret").unwrap();
        let token = minter.mint_at(1_700_000_000).unwrap();
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);
        let header: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[0]).unwrap()).unwrap();
        assert_eq!(header["alg"], "HS256");
        let claims: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(parts[1]).unwrap()).unwrap();
        assert_eq!(claims["iss"], "devkey");
        assert_eq!(claims["nbf"], 1_700_000_000u64);
        assert_eq!(claims["exp"], 1_700_003_600u64);
        assert_eq!(claims["agent"]["databaseAdmin"], true);
        let mut mac = Hmac::<Sha256>::new_from_slice(b"secret").unwrap();
        mac.update(format!("{}.{}", parts[0], parts[1]).as_bytes());
        mac.verify_slice(&URL_SAFE_NO_PAD.decode(parts[2]).unwrap()).unwrap();
    }

    #[test]
    fn the_threads_token_carries_only_a_project_wide_thread_grant() {
        let minter = TokenMinter::new("devkey", "secret").unwrap().for_threads();
        let token = minter.mint_at(1_700_000_000).unwrap();
        let body = token.split('.').nth(1).unwrap();
        let claims: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(body).unwrap()).unwrap();
        assert_eq!(claims["agentThread"]["all"], true);
        assert!(claims.get("agent").is_none());
    }

    #[test]
    fn empty_credentials_are_refused() {
        assert!(matches!(TokenMinter::new("", "x"), Err(StorageError::Invalid(_))));
    }
}
