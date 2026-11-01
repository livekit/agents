//! Local authority to disclose a caller's thread. A card is never an authorization source;
//! this verifier is used only by delegation, independently of ordinary endpoint authentication.

use std::time::{SystemTime, UNIX_EPOCH};

use livekit_token::TokenVerifier;
use lk_agents_core::a2a::delegation::VerifiedAuthority;

/// Explicit application trust in one destination for one caller's thread.
#[derive(Debug, Clone, Copy)]
pub struct ConfiguredTrust<'a> {
    pub endpoint: &'a str,
    pub thread_id: &'a str,
    pub caller_identity: &'a str,
}

/// Credentials are intentionally not Debug: errors and diagnostics never print a bearer or secret.
pub enum AuthorityAssertion<'a> {
    Grant { project_url: &'a str, api_key: &'a str, api_secret: &'a str, token: &'a str },
    ConfiguredTrust(ConfiguredTrust<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum AuthorityError {
    #[error(
        "delegation authority needs an HTTP(S) endpoint without credentials, query or fragment"
    )]
    InvalidEndpoint,
    #[error("delegation authority needs a nonempty thread and caller identity")]
    MissingCaller,
    #[error("the delegation grant is not signed by this project")]
    InvalidToken,
    #[error("the delegation endpoint is outside the locally configured project origin")]
    WrongProject,
    #[error("the delegation grant is expired or not yet active")]
    InactiveToken,
    #[error("the delegation grant belongs to another caller")]
    WrongCaller,
    #[error("the delegation grant does not allow reading and writing this thread")]
    ThreadDenied,
    #[error("configured trust does not cover this endpoint, thread and caller")]
    TrustMismatch,
}

/// Verify an explicit assertion before sharing any thread ID/history. `now` is sampled by the
/// runtime at the operation boundary. No assertion is inferred from a URL, card or peer metadata.
pub fn verify(
    endpoint: &str,
    caller_thread: &str,
    caller_identity: &str,
    assertion: AuthorityAssertion<'_>,
    now: SystemTime,
) -> Result<VerifiedAuthority, AuthorityError> {
    let destination = checked_endpoint(endpoint)?;
    if caller_thread.trim().is_empty() || caller_identity.trim().is_empty() {
        return Err(AuthorityError::MissingCaller);
    }
    match assertion {
        AuthorityAssertion::Grant { project_url, api_key, api_secret, token } => {
            if checked_endpoint(project_url)?.origin() != destination.origin() {
                return Err(AuthorityError::WrongProject);
            }
            if api_key.is_empty() || api_secret.is_empty() || token.is_empty() {
                return Err(AuthorityError::InvalidToken);
            }
            let claims = TokenVerifier::with_api_key(api_key, api_secret)
                .verify(token)
                .map_err(|_| AuthorityError::InvalidToken)?;
            let now = now
                .duration_since(UNIX_EPOCH)
                .map_err(|_| AuthorityError::InactiveToken)?
                .as_secs();
            // TokenVerifier has JWT clock-skew leeway. Disclosure requires an active grant at
            // the boundary, so an already-expired or not-yet-valid token conveys no authority.
            if u64::try_from(claims.exp).map_or(true, |expires| expires <= now)
                || u64::try_from(claims.nbf).map_or(true, |starts| starts > now)
            {
                return Err(AuthorityError::InactiveToken);
            }
            if claims.sub != caller_identity {
                return Err(AuthorityError::WrongCaller);
            }
            if !claims.agent_thread.can_read(caller_thread)
                || !claims.agent_thread.can_write(caller_thread)
            {
                return Err(AuthorityError::ThreadDenied);
            }
        }
        AuthorityAssertion::ConfiguredTrust(trust) => {
            if checked_endpoint(trust.endpoint)?.as_str().trim_end_matches('/')
                != destination.as_str().trim_end_matches('/')
                || trust.thread_id != caller_thread
                || trust.caller_identity != caller_identity
            {
                return Err(AuthorityError::TrustMismatch);
            }
        }
    }
    // Match the exact endpoint spelling retained by the interaction; canonicalization above is
    // only for comparing a configured destination, never for widening the proof's scope.
    VerifiedAuthority::from_local_verifier(
        endpoint.trim_end_matches('/').to_owned(),
        caller_thread.to_owned(),
        caller_identity.to_owned(),
    )
    .map_err(|_| AuthorityError::MissingCaller)
}

fn checked_endpoint(endpoint: &str) -> Result<url::Url, AuthorityError> {
    let url = super::checked_url(endpoint).map_err(|_| AuthorityError::InvalidEndpoint)?;
    if url.query().is_some() {
        return Err(AuthorityError::InvalidEndpoint);
    }
    Ok(url)
}

#[cfg(test)]
#[path = "authority_tests.rs"]
mod tests;
