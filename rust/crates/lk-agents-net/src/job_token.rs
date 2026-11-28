//! Job tokens, for token mode.
//!
//! In token mode the process has no API key and secret, only a worker token (`LIVEKIT_TOKEN`).
//! The worker registers with it. A job gets every other token from the LiveKit server: it asks
//! its worker, and the worker sends a `CreateJobTokenRequest`.
//!
//! A running job registers its [`JobTokens`] here, by job id. Code that runs for a job finds them
//! with the job's key, and never finds another job's tokens.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::{BoxFuture, Clock, Ts};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// The claims of the job token for LiveKit Inference.
pub const INFERENCE_CLAIMS: &str = r#"{"identity":"agent","inference":{"perform":true}}"#;
/// The claims of the job token for telemetry.
pub const OBSERVABILITY_CLAIMS: &str = r#"{"observability":{"write":true}}"#;

/// The wait after a failed renewal, and the shortest wait between two renewals. The floor stops
/// a busy loop when the server sends an expiry that is already past.
pub const RENEW_FLOOR: Duration = Duration::from_secs(5);

/// A token that the server issued for a job (`CreateJobTokenResponse`).
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct JobToken {
    pub token: String,
    /// Unix seconds.
    pub expires_at: i64,
}

impl std::fmt::Debug for JobToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobToken").field("expires_at", &self.expires_at).finish_non_exhaustive()
    }
}

/// Asks the server for a token with these claims (JSON, the payload of an access token).
pub type Fetch =
    Box<dyn Fn(String) -> BoxFuture<'static, Result<JobToken, ApiError>> + Send + Sync>;

struct Cached {
    token: String,
    /// The middle of the token's life: from here on, the next call asks for a new one.
    renew_at: Ts,
    expires_at: Ts,
}

/// The tokens of one job: one per set of claims, each kept until half of its life has passed.
pub struct JobTokens {
    fetch: Fetch,
    clock: Arc<dyn Clock>,
    cache: Mutex<HashMap<String, Cached>>,
}

impl JobTokens {
    pub fn new(fetch: Fetch, clock: Arc<dyn Clock>) -> Self {
        Self { fetch, clock, cache: Mutex::default() }
    }

    /// Returns a token with `claims`: the cached one before the middle of its life, else a new
    /// one from the server.
    pub async fn token(&self, claims: &str) -> Result<String, ApiError> {
        let now = self.clock.now();
        if let Some(cached) = self.cache.lock().get(claims).filter(|c| now < c.renew_at) {
            return Ok(cached.token.clone());
        }
        let issued = (self.fetch)(claims.to_owned()).await?;
        let life = issued.expires_at as f64 - lk_agents_abi::unix_secs(self.clock.wall());
        let life = lk_agents_abi::secs(life);
        let now = self.clock.now();
        let cached = Cached {
            token: issued.token.clone(),
            renew_at: now.saturating_add(life / 2),
            expires_at: now.saturating_add(life),
        };
        self.cache.lock().insert(claims.to_owned(), cached);
        Ok(issued.token)
    }

    /// Returns the last token with `claims`, unless it expired.
    pub fn cached(&self, claims: &str) -> Option<String> {
        let now = self.clock.now();
        self.cache.lock().get(claims).filter(|c| now < c.expires_at).map(|c| c.token.clone())
    }

    /// Gets each token in `claims` that is due. Returns the wait until the next one is due, and
    /// never less than [`RENEW_FLOOR`]. A failure is logged, and the wait is [`RENEW_FLOOR`].
    pub async fn renew(&self, claims: &[&str]) -> Duration {
        for claims in claims {
            if let Err(error) = self.token(claims).await {
                // A warning: when renewals keep failing, the job loses inference and telemetry.
                log::warn!(target: lk_agents_abi::LOG_TARGET, error:% = error; "failed to renew a job token, retrying");
                return RENEW_FLOOR;
            }
        }
        let now = self.clock.now();
        let cache = self.cache.lock();
        let next = claims.iter().filter_map(|claims| cache.get(*claims)).map(|c| c.renew_at).min();
        next.map_or(RENEW_FLOOR, |at| at.saturating_duration_since(now).max(RENEW_FLOOR))
    }
}

/// The running jobs of this process that have job tokens. Jobs that run as threads share one
/// process, so the map is keyed by job.
static JOBS: LazyLock<Mutex<HashMap<JobKey, Arc<JobTokens>>>> = LazyLock::new(Mutex::default);

/// Registers the tokens of `job`, while it runs.
pub fn open(job: JobKey, tokens: Arc<JobTokens>) {
    JOBS.lock().insert(job, tokens);
}

/// Removes the tokens of `job`, after it ended and sent its last telemetry.
pub fn close(job: &JobKey) {
    JOBS.lock().remove(job);
}

fn no_job() -> ApiError {
    ApiError::new(
        ErrorKind::Auth,
        "no LiveKit API key and secret, and no running job in token mode (LIVEKIT_TOKEN): a job \
         gets its tokens from the LiveKit server",
    )
}

/// Returns a token with `claims` for `job`, from the cache or from the server. Fails when `job`
/// is not a running token mode job of this process.
pub async fn token(job: &JobKey, claims: &str) -> Result<String, ApiError> {
    let tokens = JOBS.lock().get(job).cloned().ok_or_else(no_job)?;
    tokens.token(claims).await
}

/// Returns the current token with `claims` that `job` keeps renewed ([`INFERENCE_CLAIMS`] or
/// [`OBSERVABILITY_CLAIMS`]). `job` is `None` outside a job, such as in a text run.
pub fn cached(job: Option<&JobKey>, claims: &str) -> Result<String, ApiError> {
    let jobs = JOBS.lock();
    let tokens = job.and_then(|job| jobs.get(job)).ok_or_else(no_job)?;
    tokens.cached(claims).ok_or_else(|| {
        ApiError::new(ErrorKind::Auth, "the job has no current token from the LiveKit server")
    })
}

/// Builds the claims of a kept job's new join token after a reload, from its old join token.
///
/// The old token is decoded without a signature check: the worker has no secret, and the result
/// is only a request that the server checks. `sub` becomes `identity`, the signer's fields
/// (`iss`, `exp`, `nbf`) go, and so do `sha256` and `video.agent`, which a job token never
/// carries. The other claims stay.
pub fn reload_claims(token: &str) -> Result<String, ApiError> {
    use serde_json::{Map, Value};
    let mut claims = jsonwebtoken::dangerous::insecure_decode::<Map<String, Value>>(token)
        .map_err(|error| ApiError::new(ErrorKind::Auth, error.to_string()))?
        .claims;
    for name in ["iss", "exp", "nbf", "sha256"] {
        claims.remove(name);
    }
    if let Some(sub) = claims.remove("sub").filter(|sub| sub.as_str() != Some("")) {
        claims.entry("identity").or_insert(sub);
    }
    if let Some(video) = claims.get_mut("video").and_then(Value::as_object_mut) {
        video.remove("agent");
    }
    Ok(Value::Object(claims).to_string())
}

#[cfg(test)]
#[path = "job_token_tests.rs"]
mod tests;
