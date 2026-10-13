//! `inference/_utils.create_access_token` -- the gateway JWT, minted with `jsonwebtoken` over our
//! own claims (`livekit-api`'s `Claims` has no `kind` or `inference`). Unset grants and empty
//! strings are omitted as `Claims.asdict` does, while `sub`/`iss`/`nbf`/`exp` are always present.
//! `identity` appears only as `sub`.

use serde::Serialize;

/// The one grant the inference gateway looks at.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct InferenceGrants {
    pub perform: bool,
}

/// The claim set, exactly as `Claims.asdict()` plus the four the reference merges in afterwards.
/// `perform: false` would not be skipped (`False != ""`), so `perform` is unconditional.
#[derive(Debug, Clone, Serialize)]
struct Claims<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    inference: Option<InferenceGrants>,
    /// `with_identity(...)`, which the reference hardcodes to `"agent"` for inference.
    sub: &'a str,
    /// The API key. `jsonwebtoken` will not set this for us; it is a claim, not a header.
    iss: &'a str,
    nbf: u64,
    exp: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error(
        "api_key is required, either as argument or set LIVEKIT_API_KEY environmental variable"
    )]
    MissingApiKey,
    #[error(
        "api_secret is required, either as argument or set LIVEKIT_API_SECRET environmental variable"
    )]
    MissingApiSecret,
    #[error("failed to sign the inference access token: {0}")]
    Sign(String),
}

/// The reference's default TTL for an inference token: ten minutes (so `chat()` re-mints).
pub const DEFAULT_TTL_SECS: u64 = 600;

/// Mint the gateway JWT. `now_unix` is the caller's clock, in whole seconds as
/// `utctimetuple()` truncates.
pub fn create_access_token(
    api_key: &str,
    api_secret: &str,
    ttl_secs: u64,
    now_unix: u64,
) -> Result<String, TokenError> {
    // The callers' message (naming the env var) rather than `AccessToken.__init__`'s.
    if api_key.is_empty() {
        return Err(TokenError::MissingApiKey);
    }
    if api_secret.is_empty() {
        return Err(TokenError::MissingApiSecret);
    }

    let claims = Claims {
        inference: Some(InferenceGrants { perform: true }),
        sub: "agent",
        iss: api_key,
        nbf: now_unix,
        exp: now_unix + ttl_secs,
    };

    jsonwebtoken::encode(
        &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256),
        &claims,
        &jsonwebtoken::EncodingKey::from_secret(api_secret.as_bytes()),
    )
    .map_err(|e| TokenError::Sign(e.to_string()))
}

/// Seconds since the Unix epoch, for callers with no clock of their own (a JWT `exp` is wall
/// time).
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn payload(token: &str) -> serde_json::Value {
        let part = token.split('.').nth(1).expect("a JWT has three parts");
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(part).unwrap();
        serde_json::from_slice(&raw).unwrap()
    }

    // Catches: the claim set drifting from what the other SDKs mint (exact key set).
    #[test]
    fn the_claim_set_is_exactly_inference_sub_iss_nbf_exp() {
        let token = create_access_token("APIkey", "secret", 600, 1_700_000_000).unwrap();
        let claims = payload(&token);
        let mut keys: Vec<&str> = claims.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["exp", "inference", "iss", "nbf", "sub"]);
        assert_eq!(claims["inference"], serde_json::json!({"perform": true}));
        assert_eq!(claims["sub"], "agent");
        assert_eq!(claims["iss"], "APIkey");
        assert_eq!(claims["nbf"], 1_700_000_000u64);
        assert_eq!(claims["exp"], 1_700_000_600u64);
    }

    // Catches: signing with the api_key, or with an HMAC over the wrong bytes.
    #[test]
    fn the_token_verifies_against_the_api_secret_and_not_the_key() {
        let token = create_access_token("APIkey", "s3cret", 600, 1_700_000_000).unwrap();
        let mut val = jsonwebtoken::Validation::new(jsonwebtoken::Algorithm::HS256);
        val.required_spec_claims.clear();
        val.validate_exp = false;
        assert!(
            jsonwebtoken::decode::<serde_json::Value>(
                &token,
                &jsonwebtoken::DecodingKey::from_secret(b"s3cret"),
                &val
            )
            .is_ok()
        );
        assert!(
            jsonwebtoken::decode::<serde_json::Value>(
                &token,
                &jsonwebtoken::DecodingKey::from_secret(b"APIkey"),
                &val
            )
            .is_err()
        );
    }

    // Catches: minting from empty credentials instead of raising with the env var name.
    #[test]
    fn empty_credentials_are_rejected_before_signing() {
        assert!(matches!(create_access_token("", "s", 600, 0), Err(TokenError::MissingApiKey)));
        assert!(matches!(create_access_token("k", "", 600, 0), Err(TokenError::MissingApiSecret)));
    }
}
