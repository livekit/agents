//! Framework credentials, signed with LiveKit's token crate.

use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use livekit_token::{AccessToken, InferenceGrants};

/// How long a gateway token lives; each request signs its own.
const INFERENCE_TOKEN_TTL: Duration = Duration::from_secs(600);

/// The LiveKit Inference gateway's bearer: identity `agent`, allowed to perform inference.
pub fn inference_bearer(api_key: &str, api_secret: &str) -> Result<String, ApiError> {
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
        .with_ttl(INFERENCE_TOKEN_TTL)
        .to_jwt()
        .map_err(|e| ApiError::new(format!("failed to sign the inference access token: {e}")).with_retryable(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use livekit_token::TokenVerifier;

    // Catches: a gateway token without the inference grant or the `agent` identity, or one the
    // key pair does not verify.
    #[test]
    fn the_gateway_token_carries_the_inference_grant_and_verifies() {
        let token = inference_bearer("key", "secret").unwrap();
        let claims = TokenVerifier::with_api_key("key", "secret").verify(&token).unwrap();
        assert_eq!((claims.sub.as_str(), claims.iss.as_str(), claims.inference.perform), ("agent", "key", true));
        assert!(claims.exp > claims.nbf && claims.exp - claims.nbf <= 600);
        assert!(!inference_bearer("", "secret").unwrap_err().retryable);
    }
}
