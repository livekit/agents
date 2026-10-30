//! Every credential the framework signs, with LiveKit's token crate: the gateway's, observability's,
//! the agent database's, the worker's registration and the room join tokens.

use std::time::Duration;

use livekit_token::{
    AccessToken, AccessTokenError, AgentGrants, Claims, InferenceGrants, ObservabilityGrants,
    TokenVerifier, VideoGrants,
};
use lk_agents_abi::errors::ApiError;

// Token extension preserves claims modelled by `livekit-token`; unknown claims are dropped.

/// Lifetime of a refreshed job token.
pub const RELOAD_EXTENSION: Duration = Duration::from_secs(3600);

/// How long an agent database token lives.
pub const DATABASE_TOKEN_TTL: Duration = Duration::from_secs(3600);

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("api key and api secret are required")]
    MissingCredentials,
    #[error("{0}")]
    Token(#[from] AccessTokenError),
}

fn require(api_key: &str, api_secret: &str) -> Result<(), TokenError> {
    if api_key.is_empty() || api_secret.is_empty() {
        return Err(TokenError::MissingCredentials);
    }
    Ok(())
}

/// The worker's registration credential; re-minted before every connect attempt so a token never
/// expires mid-outage.
pub fn worker_token(api_key: &str, api_secret: &str) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    Ok(AccessToken::with_api_key(api_key, api_secret)
        .with_grants(VideoGrants { agent: true, ..Default::default() })
        .to_jwt()?)
}

/// A participant's join token for `room`: it publishes, subscribes and updates its own
/// metadata. `agent` joins it as an agent (`kind="agent"` and the agent grant).
pub fn room_join_token(
    api_key: &str,
    api_secret: &str,
    identity: &str,
    room: &str,
    agent: bool,
    ttl: Option<Duration>,
) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    let mut token = AccessToken::with_api_key(api_key, api_secret)
        .with_identity(identity)
        .with_grants(VideoGrants {
            room_join: true,
            room: room.to_owned(),
            agent,
            can_publish: Some(true),
            can_subscribe: Some(true),
            can_update_own_metadata: Some(true),
            ..Default::default()
        });
    if agent {
        token = token.with_kind("agent");
    }
    if let Some(ttl) = ttl {
        token = token.with_ttl(ttl);
    }
    Ok(token.to_jwt()?)
}

/// Verify an SFU-issued join token and refresh its lifetime to [`RELOAD_EXTENSION`].
/// Preserves all modelled claims and checks the issuer and expiration.
pub fn extend_job_token(
    token: &str,
    api_key: &str,
    api_secret: &str,
) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    let c = TokenVerifier::with_api_key(api_key, api_secret).verify(token)?;
    let mut t = AccessToken::with_api_key(api_key, api_secret)
        .with_ttl(RELOAD_EXTENSION)
        .with_identity(&c.sub)
        .with_name(&c.name)
        .with_kind(&c.kind)
        .with_metadata(&c.metadata)
        .with_sha256(&c.sha256)
        .with_attributes(c.attributes)
        .with_grants(c.video)
        .with_sip_grants(c.sip);
    if let Some(rc) = c.room_config {
        t = t.with_room_config(rc);
    }
    Ok(t.to_jwt()?)
}

/// The `sub` of a token without verifying it. Only for `simulate_job`, where the caller supplied
/// the token and is asking what identity it names.
pub fn token_identity(token: &str) -> Result<String, TokenError> {
    Ok(Claims::from_unverified(token)?.sub)
}

/// The agent database's credential: [`DATABASE_TOKEN_TTL`] long, with `agent.databaseAdmin`.
pub fn database_admin_token(api_key: &str, api_secret: &str) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    Ok(AccessToken::with_api_key(api_key, api_secret)
        .with_ttl(DATABASE_TOKEN_TTL)
        .with_agent_grants(AgentGrants { database_admin: true, ..Default::default() })
        .to_jwt()?)
}

/// How long a gateway token lives; each request signs its own.
pub const INFERENCE_TOKEN_TTL: Duration = Duration::from_secs(600);

/// The LiveKit Inference gateway's bearer: identity `agent`, allowed to perform inference.
pub fn inference_bearer(api_key: &str, api_secret: &str) -> Result<String, ApiError> {
    inference_bearer_with_ttl(api_key, api_secret, INFERENCE_TOKEN_TTL)
}

/// An inference credential with the connection's requested lifetime.
pub fn inference_bearer_with_ttl(
    api_key: &str,
    api_secret: &str,
    ttl: Duration,
) -> Result<String, ApiError> {
    if api_key.is_empty() {
        return Err(ApiError::new(
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable",
        )
        .with_retryable(false));
    }
    if api_secret.is_empty() {
        return Err(ApiError::new(
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable",
        )
        .with_retryable(false));
    }
    AccessToken::with_api_key(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(InferenceGrants::default().with_perform())
        .with_ttl(ttl)
        .to_jwt()
        .map_err(|e| {
            ApiError::new(format!("failed to sign the inference access token: {e}"))
                .with_retryable(false)
        })
}

/// An observability write credential for a framework upload or host SDK exporter.
pub fn observability_bearer(api_key: &str, api_secret: &str) -> Result<String, ApiError> {
    if api_key.is_empty() || api_secret.is_empty() {
        return Err(ApiError::new(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required for observability",
        )
        .with_retryable(false));
    }
    AccessToken::with_api_key(api_key, api_secret)
        .with_observability_grants(ObservabilityGrants::default().with_write())
        .with_ttl(Duration::from_secs(6 * 60 * 60))
        .to_jwt()
        .map_err(|err| {
            ApiError::new(format!("failed to sign observability credential: {err}"))
                .with_retryable(false)
        })
}

/// The join tokens as the host calls them.
pub struct JoinTokens;

impl lk_agents_abi::worker::JoinTokens for JoinTokens {
    fn room_join_token(
        api_key: String,
        api_secret: String,
        identity: String,
        room: String,
        agent: bool,
        ttl: Option<f64>,
    ) -> Result<String, lk_agents_abi::worker::TokenRefused> {
        let ttl = ttl
            .map(|seconds| {
                Duration::try_from_secs_f64(seconds).map_err(|_| {
                    lk_agents_abi::worker::TokenRefused("ttl must be a nonnegative number".into())
                })
            })
            .transpose()?;
        room_join_token(&api_key, &api_secret, &identity, &room, agent, ttl)
            .map_err(|error| lk_agents_abi::worker::TokenRefused(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_token::TokenVerifier;

    #[test]
    fn the_observability_token_only_grants_observability_writes() {
        let token = observability_bearer("key", "secret").unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&token).unwrap();
        assert!(claims.observability.write);
        assert!(!claims.inference.perform);
        assert!(claims.exp > claims.nbf && claims.exp - claims.nbf <= 6 * 60 * 60);
        assert!(observability_bearer("", "secret").is_err());
    }

    // Catches: a gateway token without the inference grant or the `agent` identity, or one the
    // key pair does not verify.
    #[test]
    fn the_gateway_token_carries_the_inference_grant_and_verifies() {
        let token = inference_bearer("key", "secret").unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&token).unwrap();
        assert_eq!(
            (claims.sub.as_str(), claims.iss.as_str(), claims.inference.perform),
            ("agent", "key", true)
        );
        assert!(claims.exp > claims.nbf && claims.exp - claims.nbf <= 600);
        assert!(!inference_bearer("", "secret").unwrap_err().retryable);
        assert!(!inference_bearer("key", "").unwrap_err().retryable);
    }

    mod worker {
        use super::*;
        use std::time::{SystemTime, UNIX_EPOCH};

        const KEY: &str = "APIkey";
        const SECRET: &str = "thiskeyistotallyunsafe";

        fn verify(token: &str) -> Claims {
            TokenVerifier::with_api_key(KEY, SECRET)
                .verify(token)
                .expect("verifies under the api secret")
        }

        // Catches: a worker token without `video.agent`, or minting with a missing credential.
        #[test]
        fn the_worker_token_carries_the_agent_grant() {
            let c = verify(&worker_token(KEY, SECRET).unwrap());
            assert!(c.video.agent, "{c:?}");
            assert_eq!((c.sub.as_str(), c.iss.as_str(), c.kind.as_str()), ("", KEY, ""));
            assert!(!c.video.room_join);
            assert!(matches!(worker_token("", SECRET), Err(TokenError::MissingCredentials)));
            assert!(matches!(worker_token(KEY, ""), Err(TokenError::MissingCredentials)));
        }

        // Catches: the simulated agent joining without `kind="agent"` or without its room.
        #[test]
        fn the_join_token_names_the_agent_and_its_room() {
            let t = room_join_token(KEY, SECRET, "fake-agent-1", "my-room", true, None).unwrap();
            let c = verify(&t);
            assert_eq!((c.sub.as_str(), c.kind.as_str()), ("fake-agent-1", "agent"));
            assert!(c.video.room_join && c.video.agent);
            assert_eq!(c.video.room, "my-room");
            assert_eq!(token_identity(&t).unwrap(), "fake-agent-1");
        }

        // Catches: the reload dropping the SFU's claims, or accepting a foreign-signed original.

        #[test]
        fn a_reloaded_token_keeps_the_sfus_claims_and_moves_exp() {
            let original = AccessToken::with_api_key(KEY, SECRET)
                .with_identity("agent-7")
                .with_kind("agent")
                .with_metadata("sfu-made")
                .with_grants(VideoGrants {
                    room_join: true,
                    room: "r".into(),
                    agent: true,
                    ..Default::default()
                })
                .to_jwt()
                .unwrap();
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as usize;
            let c = verify(&extend_job_token(&original, KEY, SECRET).unwrap());
            assert_eq!(
                (c.sub.as_str(), c.kind.as_str(), c.metadata.as_str()),
                ("agent-7", "agent", "sfu-made")
            );
            assert_eq!(c.video.room, "r");
            assert!(c.video.agent && c.video.room_join);
            assert!(
                (now + 3600 - 5..=now + 3600 + 5).contains(&c.exp),
                "exp {} vs now {now}",
                c.exp
            );
            assert!(extend_job_token(&original, KEY, "wrong").is_err(), "foreign secret accepted");
            assert!(
                extend_job_token(&original, "other-key", SECRET).is_err(),
                "foreign issuer accepted"
            );
            // `exp`/`nbf` enforcement (with a 60 s leeway) is `TokenVerifier`'s, pinned in `livekit-token`.
            assert!(matches!(
                extend_job_token(&original, KEY, ""),
                Err(TokenError::MissingCredentials)
            ));
        }
    }
}
