//! Job tokens for token mode.
//!
//! In token mode the process has no API key and no API secret. It has a worker token
//! (`LIVEKIT_TOKEN`) instead. The worker uses the worker token only to register. A job gets every
//! other token from the LiveKit server. The job asks its worker, and the worker sends a
//! `CreateJobTokenRequest`.
//!
//! While a job runs, it keeps two job tokens current in the background: one for LiveKit
//! Inference and one for telemetry. [`inference_token`] and [`observability_token`] read them.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::{Clock, Ts};
use parking_lot::Mutex;

/// The claims of the job token for LiveKit Inference.
pub const INFERENCE_CLAIMS: &str = r#"{"identity":"agent","inference":{"perform":true}}"#;
/// The claims of the job token for telemetry.
pub const OBSERVABILITY_CLAIMS: &str = r#"{"observability":{"write":true}}"#;

/// The wait after a failed renewal, and the shortest wait between two renewals.
///
/// The floor stops a busy loop when the server sends an expiry that is already past.
pub const RENEW_FLOOR: Duration = Duration::from_secs(5);

/// Returns true in token mode: a worker token is set, and no API key and no API secret are set.
///
/// When a key or a secret is set, the process runs in key mode, even if a worker token is set.
pub fn token_mode(api_key: &str, api_secret: &str, token: &str) -> bool {
    !token.is_empty() && api_key.is_empty() && api_secret.is_empty()
}

/// Builds the claims for a new join token after a reload, from the job's old join token.
///
/// The old token is decoded without a signature check, because the worker has no secret.
/// That is safe: the result is only a request, and the server decides what to sign.
///
/// Changes to the old claims:
/// - `sub` is renamed to `identity`, the name that a job token request uses.
/// - `iss`, `exp` and `nbf` are removed, because the server sets them.
/// - `sha256` and `video.agent` are removed, because the server refuses a request with them.
///
/// All other claims are kept, including claims that this SDK does not know.
pub fn reload_claims(token: &str) -> Result<String, ApiError> {
    let mut claims = jsonwebtoken::dangerous::insecure_decode::<
        serde_json::Map<String, serde_json::Value>,
    >(token)
    .map_err(|error| ApiError::new(ErrorKind::Auth, error.to_string()))?
    .claims;
    for name in ["iss", "exp", "nbf", "sha256"] {
        claims.remove(name);
    }
    if let Some(sub) = claims.remove("sub").filter(|sub| sub.as_str() != Some("")) {
        claims.entry("identity").or_insert(sub);
    }
    if let Some(video) = claims.get_mut("video").and_then(serde_json::Value::as_object_mut) {
        video.remove("agent");
    }
    Ok(serde_json::Value::Object(claims).to_string())
}

/// A job token that the server signed (`CreateJobTokenResponse`).
#[derive(Clone, PartialEq)]
pub struct Issued {
    pub token: String,
    /// Unix seconds.
    pub expires_at: i64,
}

impl std::fmt::Debug for Issued {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Issued").field("expires_at", &self.expires_at).finish_non_exhaustive()
    }
}

/// Asks the server for a job token with `claims`.
///
/// `claims` is JSON with the same fields as the payload of an access token. The implementation
/// limits how long it waits for the server.
#[async_trait::async_trait]
pub trait Issue: Send + Sync {
    async fn issue(&self, claims: &str) -> Result<Issued, ApiError>;
}

struct Cached {
    token: String,
    /// The middle of the token's life. From here on, the next call asks for a new token.
    renew_at: Ts,
    /// The token is not used after this time.
    expires_at: Ts,
}

/// The job tokens of one job. There is one cached token per set of claims. A token is renewed
/// when half of its life has passed.
pub struct JobTokens {
    issue: Arc<dyn Issue>,
    clock: Arc<dyn Clock>,
    cache: Mutex<BTreeMap<String, Cached>>,
}

impl JobTokens {
    pub fn new(issue: Arc<dyn Issue>, clock: Arc<dyn Clock>) -> Self {
        Self { issue, clock, cache: Mutex::default() }
    }

    /// Returns a token with `claims`. Before the middle of its life, this is the cached token.
    /// After that, the server issues a new one.
    pub async fn token(&self, claims: &str) -> Result<String, ApiError> {
        let now = self.clock.now();
        if let Some(cached) = self.cache.lock().get(claims).filter(|c| now < c.renew_at) {
            return Ok(cached.token.clone());
        }
        let issued = self.issue.issue(claims).await?;
        let now = self.clock.now();
        let life = issued.expires_at as f64 - lk_agents_abi::unix_secs(self.clock.wall());
        let life = lk_agents_abi::secs(life);
        let cached = Cached {
            token: issued.token.clone(),
            renew_at: now.saturating_add(life / 2),
            expires_at: now.saturating_add(life),
        };
        self.cache.lock().insert(claims.to_owned(), cached);
        Ok(issued.token)
    }

    /// Returns the last token issued with `claims`, or `None` once it expired.
    pub fn cached(&self, claims: &str) -> Option<String> {
        let now = self.clock.now();
        self.cache.lock().get(claims).filter(|c| now < c.expires_at).map(|c| c.token.clone())
    }

    /// Renews each token in `claims` that is due.
    ///
    /// Returns the wait until the next token is due, and never less than [`RENEW_FLOOR`]. After a
    /// failure, it logs the failure and returns [`RENEW_FLOOR`].
    pub async fn renew(&self, claims: &[&str], attempt: &mut u32) -> Duration {
        let results =
            futures_util::future::join_all(claims.iter().map(|claims| self.token(claims))).await;
        if let Some(error) = results.into_iter().find_map(Result::err) {
            *attempt += 1;
            // This is a warning, not a debug line. If renewals keep failing, the tokens expire,
            // and the job loses inference and telemetry. These lines explain why.
            lk_agents_abi::types::retry::retrying(
                log::Level::Warn,
                "job token renewal",
                *attempt,
                RENEW_FLOOR,
                &error,
            );
            return RENEW_FLOOR;
        }
        *attempt = 0;
        let now = self.clock.now();
        let cache = self.cache.lock();
        let next = claims
            .iter()
            .filter_map(|claims| cache.get(*claims))
            .map(|cached| cached.renew_at.saturating_duration_since(now))
            .min()
            .unwrap_or_default();
        next.max(RENEW_FLOOR)
    }

    /// Renews the tokens with `claims` in a loop until the future is dropped. The first renewal
    /// is after `delay`.
    pub async fn keep_current(&self, mut delay: Duration, claims: &[&str]) {
        let mut attempt = 0;
        loop {
            self.clock.wait_until(self.clock.now().saturating_add(delay)).await;
            delay = self.renew(claims, &mut attempt).await;
        }
    }
}

/// The running jobs of this process that have job tokens, oldest first.
type Jobs = Vec<(JobKey, Arc<JobTokens>)>;

fn jobs() -> &'static Mutex<Jobs> {
    static JOBS: OnceLock<Mutex<Jobs>> = OnceLock::new();
    JOBS.get_or_init(Mutex::default)
}

/// Registers the job tokens of `job`.
pub fn open(job: JobKey, tokens: Arc<JobTokens>) {
    let mut jobs = jobs().lock();
    jobs.retain(|(key, _)| *key != job);
    jobs.push((job, tokens));
}

/// Removes the job tokens of `job`. The job calls this after it ended and sent its last
/// telemetry.
pub fn close(job: &JobKey) {
    jobs().lock().retain(|(key, _)| key != job);
}

/// Finds the job tokens of `job`, and only of `job`.
///
/// Jobs that run as threads share one process. A lookup never returns another job's tokens. For
/// example, jobs A and B run in one process. A call from job A finds job A's entry, and never job
/// B's.
fn tokens_of(job: &JobKey) -> Option<Arc<JobTokens>> {
    jobs().lock().iter().find(|(key, _)| key == job).map(|(_, tokens)| tokens.clone())
}

/// Returns a job token with `claims` for `job`, from the cache or from the server.
///
/// Fails when `job` is not a running job of this process.
pub async fn token(job: &JobKey, claims: &str) -> Result<String, ApiError> {
    let Some(tokens) = tokens_of(job) else {
        return Err(ApiError::new(
            ErrorKind::Auth,
            "in token mode (LIVEKIT_TOKEN without an API key and secret) a server API call needs a \
             running job: the job gets its tokens from the LiveKit server",
        ));
    };
    tokens.token(claims).await
}

/// Returns the current inference token of `job`.
///
/// Fails when there is no running job. `job` is `None` for a text run, or for a model that is
/// called outside a `JobContext`.
pub fn inference_token(job: Option<&JobKey>) -> Result<String, ApiError> {
    job.and_then(tokens_of).and_then(|tokens| tokens.cached(INFERENCE_CLAIMS)).ok_or_else(|| {
        ApiError::new(
            ErrorKind::Auth,
            "LiveKit Inference in token mode (LIVEKIT_TOKEN without an API key and secret) needs \
             a running job: the job gets its inference token from the LiveKit server",
        )
    })
}

/// Returns the current telemetry token of `job`.
pub fn observability_token(job: &JobKey) -> Result<String, ApiError> {
    tokens_of(job).and_then(|tokens| tokens.cached(OBSERVABILITY_CLAIMS)).ok_or_else(|| {
        ApiError::new(
            ErrorKind::Auth,
            "no observability token: in token mode (LIVEKIT_TOKEN) the job gets it from the \
             LiveKit server",
        )
    })
}

#[cfg(test)]
#[path = "job_token_tests.rs"]
mod tests;
