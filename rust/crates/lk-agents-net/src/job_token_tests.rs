use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::{ApiError, ErrorKind};
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_abi::{Clock, VirtualClock};
use parking_lot::Mutex;

use super::*;
use crate::token::token_mode;

/// A fake server: each request gets a new token that expires `life` seconds after the clock's
/// wall time. While `refusing` is set, it refuses each request.
#[derive(Default)]
struct Server {
    life: Mutex<i64>,
    refusing: Mutex<bool>,
    asked: Mutex<Vec<String>>,
}

fn job_tokens(life: i64) -> (Arc<VirtualClock>, Arc<Server>, JobTokens) {
    let clock = Arc::new(VirtualClock::new());
    let server = Arc::new(Server { life: Mutex::new(life), ..Default::default() });
    let (wall, answering) = (clock.clone(), server.clone());
    let fetch: Fetch = Box::new(move |claims| {
        let mut asked = answering.asked.lock();
        asked.push(claims);
        let answer = if *answering.refusing.lock() {
            Err(ApiError::new(ErrorKind::Auth, "refused"))
        } else {
            let now = lk_agents_abi::unix_secs(wall.wall()) as i64;
            let token = format!("token-{}", asked.len());
            Ok(JobToken { token, expires_at: now + *answering.life.lock() })
        };
        Box::pin(std::future::ready(answer))
    });
    (clock.clone(), server, JobTokens::new(fetch, clock))
}

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
}

// Catches token mode without a worker token, or token mode when a key or a secret is set.
#[test]
fn token_mode_is_a_token_without_a_key_or_a_secret() {
    assert!(token_mode("", "", "t"));
    assert!(!token_mode("", "", ""));
    assert!(!token_mode("k", "", "t"));
    assert!(!token_mode("", "s", "t"));
    assert!(!token_mode("k", "s", "t"));
}

// Catches a second request for claims already cached, a token used after half of its life, one
// token shared by two sets of claims, and a token returned after it expired.
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
    });
    assert_eq!(server.asked.lock().len(), 3);
}

// Catches a renewal that waits longer than half of a token's life, a busy loop on a token that
// is already expired, and a failure that is not retried after the floor.
#[test]
fn renewal_waits_half_a_life_with_a_floor_and_retries_a_failure() {
    let claims = [INFERENCE_CLAIMS, OBSERVABILITY_CLAIMS];
    // The expiry is in whole seconds, so half of the life is a bit under 300 s.
    let half_life =
        |wait: Duration| wait > Duration::from_secs(299) && wait <= Duration::from_secs(300);
    let (_, _, tokens) = job_tokens(600);
    assert!(half_life(block_on(tokens.renew(&claims))));

    let (_, _, expired) = job_tokens(-60);
    assert_eq!(block_on(expired.renew(&claims)), RENEW_FLOOR);
    assert_eq!(expired.cached(INFERENCE_CLAIMS), None);

    let (_, server, failing) = job_tokens(600);
    *server.refusing.lock() = true;
    assert_eq!(block_on(failing.renew(&claims)), RENEW_FLOOR);
    *server.refusing.lock() = false;
    assert!(half_life(block_on(failing.renew(&claims))));
}

/// The registry is shared by the whole process: the tests that use it run one at a time.
static REGISTRY: Mutex<()> = parking_lot::const_mutex(());

// Two jobs are registered, and only the first has a telemetry token. Catches a lookup that
// returns another job's token, a token for a caller with no job, a token from another job's
// server, or a token after its job closed.
#[test]
fn a_job_reads_only_its_own_tokens_until_it_closes() {
    let _registry = REGISTRY.lock();
    let (first, second) = (JobKey("AJ_token_1".into()), JobKey("AJ_token_2".into()));
    let (_, first_server, a) = job_tokens(600);
    let (_, second_server, b) = job_tokens(600);
    let (a, b) = (Arc::new(a), Arc::new(b));
    block_on(async {
        a.token(INFERENCE_CLAIMS).await.unwrap();
        a.token(OBSERVABILITY_CLAIMS).await.unwrap();
        b.token(INFERENCE_CLAIMS).await.unwrap();
    });
    assert!(cached(Some(&first), INFERENCE_CLAIMS).is_err(), "not registered yet");
    open(first.clone(), a.clone());
    open(second.clone(), b.clone());
    assert_eq!(cached(Some(&first), INFERENCE_CLAIMS).unwrap(), "token-1");
    assert_eq!(cached(Some(&second), INFERENCE_CLAIMS).unwrap(), "token-1");
    assert!(cached(Some(&second), OBSERVABILITY_CLAIMS).is_err(), "never the first job's");
    assert!(cached(None, INFERENCE_CLAIMS).is_err(), "no job, no token");
    let claims = r#"{"video":{"roomCreate":true}}"#;
    assert_eq!(block_on(token(&first, claims)).unwrap(), "token-3");
    assert!(block_on(token(&JobKey("AJ_other".into()), claims)).is_err());
    assert_eq!(first_server.asked.lock().last().map(String::as_str), Some(claims));
    assert_eq!(second_server.asked.lock().len(), 1, "another job's server was asked");
    close(&first);
    close(&second);
    assert!(cached(Some(&first), OBSERVABILITY_CLAIMS).is_err());
}

// Catches reload claims that keep `video.agent`, `sha256`, `sub` or the signer's fields, or
// that drop the other claims the SFU set.
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
    let claims: serde_json::Value =
        serde_json::from_str(&reload_claims(&original).unwrap()).unwrap();
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
