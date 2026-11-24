use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::{Clock, VirtualClock};
use parking_lot::Mutex;

use super::*;

/// A fake server. It answers each request with a new token that expires `life` seconds after the
/// clock's wall time. While `refusing` is set, it fails each request.
struct Server {
    clock: Arc<VirtualClock>,
    life: Mutex<i64>,
    refusing: Mutex<bool>,
    asked: Mutex<Vec<String>>,
}

impl Server {
    fn new(clock: Arc<VirtualClock>, life: i64) -> Arc<Self> {
        Arc::new(Self {
            clock,
            life: Mutex::new(life),
            refusing: Mutex::new(false),
            asked: Mutex::default(),
        })
    }
}

#[async_trait::async_trait]
impl Issue for Server {
    async fn issue(&self, claims: &str) -> Result<Issued, ApiError> {
        let mut asked = self.asked.lock();
        asked.push(claims.to_owned());
        if *self.refusing.lock() {
            return Err(ApiError::new(ErrorKind::Auth, "refused"));
        }
        let wall = lk_agents_abi::unix_secs(self.clock.wall()) as i64;
        Ok(Issued { token: format!("token-{}", asked.len()), expires_at: wall + *self.life.lock() })
    }
}

fn job_tokens(life: i64) -> (Arc<VirtualClock>, Arc<Server>, JobTokens) {
    let clock = Arc::new(VirtualClock::new());
    let server = Server::new(clock.clone(), life);
    let tokens = JobTokens::new(server.clone(), clock.clone());
    (clock, server, tokens)
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
}

// Checks each mix of key, secret and worker token. Catches token mode without a worker token, or
// token mode when a key or a secret is set.
#[test]
fn token_mode_is_a_token_without_a_key_or_a_secret() {
    assert!(token_mode("", "", "t"));
    assert!(!token_mode("", "", ""));
    assert!(!token_mode("k", "", "t"));
    assert!(!token_mode("", "s", "t"));
    assert!(!token_mode("k", "s", "t"));
}

// One job asks for tokens while the clock moves forward. Catches these bugs:
// - a second server request for claims that are already cached;
// - a cached token that is still used after half of its life;
// - one cached token shared by two different sets of claims;
// - a cached token returned after it expired.
#[test]
fn a_token_is_cached_per_claims_until_half_its_life_and_never_used_expired() {
    let (clock, server, tokens) = job_tokens(600);
    block_on(async {
        assert_eq!(tokens.token(INFERENCE_CLAIMS).await.unwrap(), "token-1");
        assert_eq!(tokens.token(INFERENCE_CLAIMS).await.unwrap(), "token-1", "cached");
        assert_eq!(tokens.token(OBSERVABILITY_CLAIMS).await.unwrap(), "token-2");
        clock.advance(Duration::from_secs(301));
        assert_eq!(tokens.cached(INFERENCE_CLAIMS).as_deref(), Some("token-1"), "still valid");
        assert_eq!(tokens.token(INFERENCE_CLAIMS).await.unwrap(), "token-3", "renewed");
        clock.advance(Duration::from_secs(600));
        assert_eq!(tokens.cached(INFERENCE_CLAIMS), None, "expired");
        assert_eq!(tokens.cached(OBSERVABILITY_CLAIMS), None, "expired");
    });
    assert_eq!(server.asked.lock().len(), 3);
}

// Renews against a normal server, a server that sends expired tokens, and a server that fails
// once. Catches a renewal that waits longer than half of a token's life, a busy loop on an
// expired token, and a failure that stops the renewal or retries with no wait.
#[test]
fn renewal_waits_half_a_life_with_a_floor_and_retries_a_failure() {
    let claims = [INFERENCE_CLAIMS, OBSERVABILITY_CLAIMS];
    let (_, server, tokens) = job_tokens(600);
    let mut attempt = 0;
    // The server sends the expiry in whole seconds. So half of the life is a bit less than
    // 300 s.
    let half_life =
        |wait: Duration| wait > Duration::from_secs(299) && wait <= Duration::from_secs(300);
    assert!(half_life(block_on(tokens.renew(&claims, &mut attempt))));

    let (_, server_expired, expired) = job_tokens(-60);
    assert_eq!(block_on(expired.renew(&claims, &mut attempt)), RENEW_FLOOR);
    assert_eq!(expired.cached(INFERENCE_CLAIMS), None);
    assert_eq!(server_expired.asked.lock().len(), 2);

    let (_, refusing, failing) = job_tokens(600);
    *refusing.refusing.lock() = true;
    assert_eq!(block_on(failing.renew(&claims, &mut attempt)), RENEW_FLOOR);
    assert_eq!(attempt, 1);
    *refusing.refusing.lock() = false;
    assert!(half_life(block_on(failing.renew(&claims, &mut attempt))));
    assert_eq!(attempt, 0);
    assert_eq!(server.asked.lock().len(), 2);
}

// Runs the background renewal on a virtual clock. Catches a renewal that never runs, or that
// runs before its delay.
#[test]
fn keep_current_renews_at_each_due_time() {
    let (clock, server, tokens) = job_tokens(20);
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    rt.block_on(async {
        let claims = [INFERENCE_CLAIMS];
        let mut renewing = Box::pin(tokens.keep_current(Duration::from_secs(10), &claims));
        let poll = |renewing: &mut std::pin::Pin<Box<_>>| {
            futures_util::FutureExt::now_or_never(renewing).is_none()
        };
        assert!(poll(&mut renewing));
        assert!(server.asked.lock().is_empty(), "not before its delay");
        clock.advance(Duration::from_secs(10));
        assert!(poll(&mut renewing));
        assert_eq!(tokens.cached(INFERENCE_CLAIMS).as_deref(), Some("token-1"));
        clock.advance(Duration::from_secs(10));
        assert!(poll(&mut renewing));
        assert_eq!(tokens.cached(INFERENCE_CLAIMS).as_deref(), Some("token-2"));
    });
}

/// The job registry is shared by the whole process, so the tests that register jobs take this
/// lock and run one at a time.
static REGISTRY: Mutex<()> = parking_lot::const_mutex(());

// Two jobs are registered, and only the first job has a telemetry token. Catches a lookup that
// returns another job's token, a token for a caller with no job, or a token after its job
// closed.
#[test]
fn a_job_reads_only_its_own_tokens_until_it_closes() {
    let _registry = REGISTRY.lock();
    let (first, second) = (JobKey("AJ_token_1".into()), JobKey("AJ_token_2".into()));
    let (_, _, a) = job_tokens(600);
    let (_, _, b) = job_tokens(600);
    let (a, b) = (Arc::new(a), Arc::new(b));
    block_on(async {
        a.token(INFERENCE_CLAIMS).await.unwrap();
        a.token(OBSERVABILITY_CLAIMS).await.unwrap();
        b.token(INFERENCE_CLAIMS).await.unwrap();
    });
    let error = inference_token(Some(&first)).unwrap_err();
    assert!(error.message.contains("needs a running job"), "{error}");
    open(first.clone(), a.clone());
    open(second.clone(), b.clone());
    assert_eq!(inference_token(Some(&first)).unwrap(), a.cached(INFERENCE_CLAIMS).unwrap());
    assert_eq!(inference_token(Some(&second)).unwrap(), b.cached(INFERENCE_CLAIMS).unwrap());
    // The second job has no telemetry token. It must not get the first job's.
    assert!(observability_token(&second).is_err());
    assert!(inference_token(Some(&JobKey("AJ_other".into()))).is_err(), "a job not running");
    let error = inference_token(None).unwrap_err();
    assert!(error.message.contains("needs a running job"), "no job, no token: {error}");
    close(&first);
    close(&second);
    assert!(observability_token(&first).is_err());
}

// Two jobs are registered, and the first job asks for a server API token. Catches a token that
// comes from the second job, which is the newest job.
#[test]
fn a_jobs_api_token_comes_from_its_own_issuer() {
    let _registry = REGISTRY.lock();
    let (first, second) = (JobKey("AJ_api_1".into()), JobKey("AJ_api_2".into()));
    let (_, first_server, a) = job_tokens(600);
    let (_, second_server, b) = job_tokens(600);
    open(first.clone(), Arc::new(a));
    open(second.clone(), Arc::new(b));
    let claims = r#"{"video":{"roomCreate":true}}"#;
    block_on(async {
        assert_eq!(token(&first, claims).await.unwrap(), "token-1");
        assert_eq!(token(&first, claims).await.unwrap(), "token-1", "cached");
        let error = token(&JobKey("AJ_other".into()), claims).await.unwrap_err();
        assert!(error.message.contains("needs a running job"), "{error}");
    });
    assert_eq!(*first_server.asked.lock(), [claims]);
    assert!(second_server.asked.lock().is_empty(), "another job's issuer was asked");
    close(&first);
    close(&second);
}

// Builds the reload claims from a join token with the agent grant and a `sha256`. Catches a
// request that keeps `video.agent`, `sha256`, `sub` or the signer's fields, or that drops the
// other claims that the SFU set.
#[test]
fn reload_claims_keep_the_jobs_claims_without_the_agent_grant_sha256_or_signing_fields() {
    use livekit_token::{AccessToken, VideoGrants};
    let original = AccessToken::with_api_key("key", "secret-secret-secret-secret-secret")
        .with_identity("agent-7")
        .with_kind("agent")
        .with_metadata("sfu-made")
        .with_sha256("hash")
        .with_grants(VideoGrants {
            room_join: true,
            room: "r".into(),
            agent: true,
            ..Default::default()
        })
        .to_jwt()
        .unwrap();
    let claims = reload_claims(&original).unwrap();
    let claims: serde_json::Value = serde_json::from_str(&claims).unwrap();
    assert_eq!(claims["identity"], "agent-7");
    assert_eq!(claims["kind"], "agent");
    assert_eq!(claims["metadata"], "sfu-made");
    assert_eq!(claims["video"]["roomJoin"], true);
    assert_eq!(claims["video"]["room"], "r");
    for gone in ["sub", "iss", "exp", "nbf", "sha256"] {
        assert!(claims.get(gone).is_none(), "{gone} kept: {claims}");
    }
    assert!(claims["video"].get("agent").is_none(), "{claims}");
    assert!(reload_claims("not a jwt").is_err());
}
