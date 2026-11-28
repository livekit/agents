use std::time::{Duration, SystemTime};

use livekit_token::{
    AccessToken, AccessTokenError, AgentDatabaseGrants, AgentThreadGrants, Claims, InferenceGrants,
    ObservabilityGrants, TokenVerifier, VideoGrants,
};
use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::worker::TokenRefused;
use serde_json::{Map, Value};

use crate::job_token::{self, INFERENCE_CLAIMS, OBSERVABILITY_CLAIMS};

// Renewal preserves the verified claim document, including future server restrictions.

/// Lifetime of a refreshed job token.
pub const RELOAD_EXTENSION: Duration = Duration::from_secs(3600);

/// How long an agent database token lives.
pub const DATABASE_TOKEN_TTL: Duration = Duration::from_secs(3600);

/// A credential that cannot be signed or read: retrying does not help.
fn refused(error: AccessTokenError) -> ApiError {
    ApiError::new(ErrorKind::Auth, error.to_string())
}

fn require(api_key: &str, api_secret: &str) -> Result<(), ApiError> {
    if api_key.is_empty() || api_secret.is_empty() {
        return Err(ApiError::new(ErrorKind::Auth, "api key and api secret are required"));
    }
    Ok(())
}

/// Refuse lifetimes the signing SDK cannot add to the current wall clock safely.
pub(crate) fn validate_token_ttl(ttl: Duration) -> Result<(), ApiError> {
    if SystemTime::now().checked_add(ttl).is_none() {
        return Err(ApiError::new(ErrorKind::InvalidRequest, "token TTL is out of range"));
    }
    Ok(())
}

/// The worker's registration credential; re-minted before every connect attempt so a token never
/// expires mid-outage.
pub fn worker_token(api_key: &str, api_secret: &str) -> Result<String, ApiError> {
    require(api_key, api_secret)?;
    AccessToken::with_api_key(api_key, api_secret)
        .with_grants(VideoGrants { agent: true, ..Default::default() })
        .to_jwt()
        .map_err(refused)
}

#[derive(Default)]
pub struct RoomJoinOptions {
    pub agent: bool,
    pub ttl: Option<Duration>,
}

/// A participant's join token for `room`: it publishes, subscribes and updates its own
/// metadata. `agent` joins it as an agent (`kind="agent"` and the agent grant).
pub fn room_join_token(
    api_key: &str,
    api_secret: &str,
    identity: &str,
    room: &str,
    options: RoomJoinOptions,
) -> Result<String, ApiError> {
    let RoomJoinOptions { agent, ttl } = options;
    require(api_key, api_secret)?;
    if let Some(ttl) = ttl {
        validate_token_ttl(ttl)?;
    }
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
    token.to_jwt().map_err(refused)
}

/// Verify an SFU-issued join token and refresh its lifetime to [`RELOAD_EXTENSION`].
/// Preserves every signed claim, changing only `nbf` and `exp`, after checking the original
/// signature, issuer and validity window. Future restrictions must not be dropped on reload.
pub fn extend_job_token(token: &str, api_key: &str, api_secret: &str) -> Result<String, ApiError> {
    require(api_key, api_secret)?;
    let verified =
        TokenVerifier::with_api_key(api_key, api_secret).verify(token).map_err(refused)?;
    // Keep the signing SDK's room-join validation when retaining the original document.
    if verified.video.room_join && (verified.sub.is_empty() || verified.video.room.is_empty()) {
        return Err(refused(AccessTokenError::InvalidClaims(
            "token grants room_join but doesn't have an identity or room",
        )));
    }
    // This exact document has already been verified above. Reading it without rebuilding a
    // typed Claims retains nested permissions and claims added by a newer server.
    let mut claims = jsonwebtoken::dangerous::insecure_decode::<
        serde_json::Map<String, serde_json::Value>,
    >(token)
    .map_err(|error| refused(error.into()))?
    .claims;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| ApiError::new(ErrorKind::Auth, "system clock is before the Unix epoch"))?;
    let expires = now
        .checked_add(RELOAD_EXTENSION)
        .ok_or_else(|| ApiError::new(ErrorKind::Auth, "token expiration is out of range"))?;
    claims.insert("nbf".into(), now.as_secs().into());
    claims.insert("exp".into(), expires.as_secs().into());
    jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(api_secret.as_bytes()),
    )
    .map_err(|error| refused(error.into()))
}

/// The `sub` of a token without verifying it. Only for `simulate_job`, where the caller supplied
/// the token and is asking what identity it names.
pub fn token_identity(token: &str) -> Result<String, ApiError> {
    Ok(Claims::from_unverified(token).map_err(refused)?.sub)
}

/// A data-plane token for one database: a thread's (`AT_...`) through its thread grant's `sql`,
/// a standalone one (`DB_...`) through `agentDatabase`.
pub fn database_token(
    api_key: &str,
    api_secret: &str,
    database_id: &str,
) -> Result<String, ApiError> {
    require(api_key, api_secret)?;
    let token = AccessToken::with_api_key(api_key, api_secret).with_ttl(DATABASE_TOKEN_TTL);
    let ids = vec![database_id.to_owned()];
    let token = if database_id.starts_with("AT_") {
        token.with_agent_thread_grants(AgentThreadGrants {
            thread_ids: ids,
            sql: true,
            ..Default::default()
        })
    } else {
        token.with_agent_database_grants(AgentDatabaseGrants {
            database_ids: ids,
            read: true,
            write: true,
            ..Default::default()
        })
    };
    token.to_jwt().map_err(refused)
}

pub const INFERENCE_TOKEN_TTL: Duration = Duration::from_secs(600);

/// Returns true in token mode: a worker token is set, and no API key and no API secret. With a
/// key or a secret, the process runs in key mode, even when a worker token is set.
pub fn token_mode(api_key: &str, api_secret: &str, token: &str) -> bool {
    !token.is_empty() && api_key.is_empty() && api_secret.is_empty()
}

/// Returns true when a client with `api_key` and `api_secret` uses job tokens: this process runs
/// in token mode (`LIVEKIT_TOKEN`) and the client has no key pair of its own.
pub fn uses_job_token(api_key: &str, api_secret: &str) -> bool {
    let token = crate::env::server_setting("LIVEKIT_TOKEN").unwrap_or_default();
    token_mode(api_key, api_secret, &token)
}

/// Returns the bearer for the LiveKit Inference gateway, with identity `agent` and the grant to
/// perform inference.
///
/// In token mode, this is the current inference token of `job`.
pub fn inference_bearer(
    job: Option<&JobKey>,
    api_key: &str,
    api_secret: &str,
) -> Result<String, ApiError> {
    inference_bearer_with_ttl(job, api_key, api_secret, INFERENCE_TOKEN_TTL)
}

/// Returns an inference credential with the lifetime that the connection asks for.
///
/// In token mode, this is the job's inference token, and the server sets its lifetime.
pub fn inference_bearer_with_ttl(
    job: Option<&JobKey>,
    api_key: &str,
    api_secret: &str,
    ttl: Duration,
) -> Result<String, ApiError> {
    if uses_job_token(api_key, api_secret) {
        return job_token::cached(job, INFERENCE_CLAIMS);
    }
    if api_key.is_empty() {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable",
        ));
    }
    if api_secret.is_empty() {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable",
        ));
    }
    validate_token_ttl(ttl)?;
    AccessToken::with_api_key(api_key, api_secret)
        .with_identity("agent")
        .with_inference_grants(InferenceGrants::default().with_perform())
        .with_ttl(ttl)
        .to_jwt()
        .map_err(|e| {
            ApiError::new(
                ErrorKind::Auth,
                format!("failed to sign the inference access token: {e}"),
            )
        })
}

/// Returns an observability write credential for a framework upload or a host SDK exporter.
///
/// In token mode, this is the current telemetry token of `job`.
pub fn observability_bearer(
    job: &JobKey,
    api_key: &str,
    api_secret: &str,
) -> Result<String, ApiError> {
    if uses_job_token(api_key, api_secret) {
        return job_token::cached(Some(job), OBSERVABILITY_CLAIMS);
    }
    observability_key_bearer(api_key, api_secret)
}

/// Signs an observability write credential with the key and secret. For an export that belongs
/// to no job, which has no token in token mode.
pub fn observability_key_bearer(api_key: &str, api_secret: &str) -> Result<String, ApiError> {
    if api_key.is_empty() || api_secret.is_empty() {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required for observability",
        ));
    }
    AccessToken::with_api_key(api_key, api_secret)
        .with_observability_grants(ObservabilityGrants::default().with_write())
        .with_ttl(Duration::from_secs(6 * 60 * 60))
        .to_jwt()
        .map_err(|err| {
            ApiError::new(
                ErrorKind::Auth,
                format!("failed to sign observability credential: {err}"),
            )
        })
}

/// How long a participant token lives when the caller sets no lifetime.
pub const PARTICIPANT_TOKEN_TTL: Duration = Duration::from_secs(6 * 60 * 60);

/// Signs `claims` with the key pair. `claims` is JSON in the form of `CreateJobTokenRequest`:
/// the payload of an access token, with the identity in `identity`.
pub fn sign_claims(
    claims: &str,
    api_key: &str,
    api_secret: &str,
    ttl: Duration,
) -> Result<String, ApiError> {
    require(api_key, api_secret)?;
    validate_token_ttl(ttl)?;
    let invalid =
        |error: serde_json::Error| ApiError::new(ErrorKind::InvalidRequest, error.to_string());
    let mut fields: Map<String, Value> = serde_json::from_str(claims).map_err(invalid)?;
    if let Some(identity) = fields.remove("identity") {
        fields.insert("sub".into(), identity);
    }
    let claims: Claims = serde_json::from_value(Value::Object(fields)).map_err(invalid)?;
    let mut token = AccessToken::with_api_key(api_key, api_secret)
        .with_ttl(ttl)
        .with_identity(&claims.sub)
        .with_name(&claims.name)
        .with_kind(&claims.kind)
        .with_metadata(&claims.metadata)
        .with_attributes(claims.attributes)
        .with_sha256(&claims.sha256)
        .with_grants(claims.video)
        .with_sip_grants(claims.sip)
        .with_agent_grants(claims.agent)
        .with_agent_endpoint_grants(claims.agent_endpoint)
        .with_agent_thread_grants(claims.agent_thread)
        .with_agent_database_grants(claims.agent_database)
        .with_inference_grants(claims.inference)
        .with_observability_grants(claims.observability);
    if let Some(config) = claims.room_config {
        token = token.with_room_config(config);
    }
    token.to_jwt().map_err(refused)
}

/// A token for a participant other than the agent, such as an avatar.
///
/// With a key pair, it is signed here and lives `ttl`. Without one, it is a job token of `job`,
/// which the server issues and gives a lifetime (token mode). For example, an avatar token in
/// key mode is signed with the key; in token mode it comes from the server for the job.
pub async fn participant_token(
    claims: &str,
    api_key: &str,
    api_secret: &str,
    job: Option<&JobKey>,
    ttl: Duration,
) -> Result<String, ApiError> {
    match job {
        Some(job) if api_key.is_empty() && api_secret.is_empty() => {
            job_token::token(job, claims).await
        }
        _ => sign_claims(claims, api_key, api_secret, ttl),
    }
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
        room_join_token(&api_key, &api_secret, &identity, &room, RoomJoinOptions { agent, ttl })
            .map_err(|error| lk_agents_abi::worker::TokenRefused(error.to_string()))
    }

    async fn participant_token(
        claims: String,
        api_key: String,
        api_secret: String,
        job_id: Option<String>,
        ttl: Option<f64>,
    ) -> Result<String, TokenRefused> {
        let ttl = match ttl {
            Some(seconds) => Duration::try_from_secs_f64(seconds)
                .map_err(|_| TokenRefused("ttl must be a nonnegative number".into()))?,
            None => PARTICIPANT_TOKEN_TTL,
        };
        let job = job_id.map(JobKey);
        participant_token(&claims, &api_key, &api_secret, job.as_ref(), ttl)
            .await
            .map_err(|error| TokenRefused(error.message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_token::TokenVerifier;

    #[test]
    fn the_observability_token_only_grants_observability_writes() {
        let token = observability_bearer(&JobKey("AJ_1".into()), "key", "secret").unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&token).unwrap();
        assert!(claims.observability.write);
        assert!(!claims.inference.perform);
        assert!(claims.exp > claims.nbf && claims.exp - claims.nbf <= 6 * 60 * 60);
        assert!(observability_bearer(&JobKey("AJ_1".into()), "", "secret").is_err());
    }

    // Catches: a gateway token without the inference grant or the `agent` identity, or one the
    // key pair does not verify.
    #[test]
    fn the_gateway_token_carries_the_inference_grant_and_verifies() {
        let token = inference_bearer(None, "key", "secret").unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&token).unwrap();
        assert_eq!(
            (claims.sub.as_str(), claims.iss.as_str(), claims.inference.perform),
            ("agent", "key", true)
        );
        assert!(claims.exp > claims.nbf && claims.exp - claims.nbf <= 600);
        assert!(!inference_bearer(None, "", "secret").unwrap_err().retryable);
        assert!(!inference_bearer(None, "key", "").unwrap_err().retryable);
    }

    // Catches a key pair that does not sign a participant token (or loses its identity, kind,
    // attributes or lifetime), a token signed without a key pair, and a job token asked for while
    // a key pair is set.
    #[test]
    fn a_participant_token_is_signed_with_a_key_pair_and_else_comes_from_the_job() {
        let claims = r#"{"identity":"avatar","kind":"agent","attributes":{"a":"b"},"video":{"roomJoin":true,"room":"call"}}"#;
        let job = JobKey("AJ_not_running".into());
        let ttl = Duration::from_secs(60);
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let token = rt.block_on(participant_token(claims, "key", "secret", Some(&job), ttl));
        let signed = TokenVerifier::with_api_key("key", "secret").verify(&token.unwrap()).unwrap();
        assert_eq!((signed.sub.as_str(), signed.kind.as_str()), ("avatar", "agent"));
        assert_eq!((signed.video.room.as_str(), signed.attributes["a"].as_str()), ("call", "b"));
        assert_eq!(signed.exp - signed.nbf, 60);
        let no_job = rt.block_on(participant_token(claims, "", "", Some(&job), ttl)).unwrap_err();
        assert!(no_job.message.contains("token mode"), "{no_job}");
        assert!(rt.block_on(participant_token(claims, "", "", None, ttl)).is_err());
        let no_identity = r#"{"video":{"roomJoin":true,"room":"call"}}"#;
        assert!(sign_claims(no_identity, "key", "secret", ttl).is_err());
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
            assert!(worker_token("", SECRET).is_err_and(|e| !e.retryable));
            assert!(worker_token(KEY, "").is_err_and(|e| !e.retryable));
        }

        // Catches: the simulated agent joining without `kind="agent"` or without its room.
        #[test]
        fn the_join_token_names_the_agent_and_its_room() {
            let t = room_join_token(
                KEY,
                SECRET,
                "fake-agent-1",
                "my-room",
                RoomJoinOptions { agent: true, ..Default::default() },
            )
            .unwrap();
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
            assert!(extend_job_token(&original, KEY, "").is_err_and(|e| !e.retryable));
        }
    }
}

#[cfg(test)]
mod renewal_regressions {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn job_token_renewal_preserves_known_grants_and_future_signed_restrictions() {
        let original = AccessToken::with_api_key("key", "secret")
            .with_identity("agent")
            .with_agent_thread_grants(AgentThreadGrants {
                sql: true,
                thread_ids: vec!["AT_1".into()],
                ..Default::default()
            })
            .with_agent_database_grants(AgentDatabaseGrants {
                read: true,
                database_ids: vec!["DB_1".into()],
                ..Default::default()
            })
            .with_inference_grants(InferenceGrants::default().with_perform())
            .with_observability_grants(ObservabilityGrants::default().with_write())
            .to_jwt()
            .unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&original).unwrap();
        let mut claims = serde_json::to_value(claims).unwrap();
        claims["futureRestriction"] = json!({"allowedModels": ["only-this"], "ceiling": 0});
        claims["video"]["futurePermission"] = json!({"deny": true});
        let original = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(b"secret"),
        )
        .unwrap();
        let renewed = extend_job_token(&original, "key", "secret").unwrap();
        TokenVerifier::with_api_key("key", "secret").verify(&renewed).unwrap();
        let mut renewed =
            jsonwebtoken::dangerous::insecure_decode::<Value>(&renewed).unwrap().claims;
        for field in ["exp", "nbf"] {
            claims.as_object_mut().unwrap().remove(field);
            renewed.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(renewed, claims);
        assert!(extend_job_token(&original, "key", "foreign-secret").is_err());
    }

    #[test]
    fn unrepresentable_token_lifetimes_return_errors_instead_of_panicking() {
        assert!(
            room_join_token(
                "key",
                "secret",
                "agent",
                "room",
                RoomJoinOptions { ttl: Some(Duration::MAX), ..Default::default() }
            )
            .is_err()
        );
        assert!(inference_bearer_with_ttl(None, "key", "secret", Duration::MAX).is_err());
    }
}

#[cfg(test)]
mod validity_window_regressions {
    use super::*;

    #[test]
    fn renewal_does_not_resign_expired_or_not_yet_valid_claims() {
        let original = AccessToken::with_api_key("key", "secret").to_jwt().unwrap();
        let original = TokenVerifier::with_api_key("key", "secret").verify(&original).unwrap();
        for (field, value) in [("exp", 1), ("nbf", original.nbf as u64 + 3600)] {
            let mut claims = serde_json::to_value(&original).unwrap();
            claims[field] = value.into();
            let invalid = jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &claims,
                &jsonwebtoken::EncodingKey::from_secret(b"secret"),
            )
            .unwrap();
            assert!(
                extend_job_token(&invalid, "key", "secret").is_err_and(|error| !error.retryable)
            );
        }
    }
}

#[cfg(test)]
mod join_claim_regressions {
    use super::*;
    use serde_json::json;

    #[test]
    fn renewal_keeps_the_signers_validation_of_room_join_claims() {
        let original = AccessToken::with_api_key("key", "secret").to_jwt().unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&original).unwrap();
        for (identity, room) in [("", "room"), ("agent", "")] {
            let mut claims = serde_json::to_value(&claims).unwrap();
            claims["sub"] = identity.into();
            claims["video"] = json!({"roomJoin": true, "room": room});
            let invalid = jsonwebtoken::encode(
                &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
                &claims,
                &jsonwebtoken::EncodingKey::from_secret(b"secret"),
            )
            .unwrap();
            assert!(extend_job_token(&invalid, "key", "secret").is_err());
        }
    }
}
