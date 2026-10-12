//! Credentials for worker registration, simulated jobs, and job reloads.

use livekit_token::{AccessToken, AccessTokenError, Claims, TokenVerifier, VideoGrants};
use std::time::Duration;

// Token extension preserves claims modelled by `livekit-token`; unknown claims are dropped.

/// Lifetime of a refreshed job token.
pub const RELOAD_EXTENSION: Duration = Duration::from_secs(3600);

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

/// Re-minted before every connect attempt so a token never expires mid-outage.
pub fn worker_token(api_key: &str, api_secret: &str) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    Ok(AccessToken::with_api_key(api_key, api_secret)
        .with_grants(VideoGrants { agent: true, ..Default::default() })
        .to_jwt()?)
}

/// The join token a simulated job's agent participant uses.
#[lk_agents_abi::host(export)]
pub fn agent_join_token(
    api_key: &str,
    api_secret: &str,
    identity: &str,
    room: &str,
) -> Result<String, TokenError> {
    require(api_key, api_secret)?;
    Ok(AccessToken::with_api_key(api_key, api_secret)
        .with_identity(identity)
        .with_kind("agent")
        .with_grants(VideoGrants {
            room_join: true,
            room: room.to_owned(),
            agent: true,
            ..Default::default()
        })
        .to_jwt()?)
}

/// Verify an SFU-issued join token and refresh its lifetime to [`RELOAD_EXTENSION`].
/// Preserves all modelled claims and checks the issuer and expiration.
#[lk_agents_abi::host(export)]
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
#[lk_agents_abi::host(export)]
pub fn token_identity(token: &str) -> Result<String, TokenError> {
    Ok(Claims::from_unverified(token)?.sub)
}

pub fn authorization_header(jwt: &str) -> (String, String) {
    ("Authorization".to_owned(), format!("Bearer {jwt}"))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

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
        let t = agent_join_token(KEY, SECRET, "fake-agent-1", "my-room").unwrap();
        let c = verify(&t);
        assert_eq!((c.sub.as_str(), c.kind.as_str()), ("fake-agent-1", "agent"));
        assert!(c.video.room_join && c.video.agent);
        assert_eq!(c.video.room, "my-room");
        assert_eq!(token_identity(&t).unwrap(), "fake-agent-1");
        assert_eq!(authorization_header("x"), ("Authorization".to_owned(), "Bearer x".to_owned()));
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
        assert!((now + 3600 - 5..=now + 3600 + 5).contains(&c.exp), "exp {} vs now {now}", c.exp);
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
