//! Inference in token mode.
//!
//! These tests run in their own test binary. They register server settings with a worker token
//! and no key or secret, and those settings apply to every test in the binary.
#![allow(clippy::unwrap_used, reason = "integration test")]
#![allow(clippy::disallowed_methods, reason = "tests build their own runtime")]

use std::sync::{Arc, OnceLock};

use lk_agents_abi::inference::InferenceCredentials as _;
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_inference::credentials::Credentials;
use lk_agents_net::job_token::{
    self, Fetch, INFERENCE_CLAIMS, JobToken, JobTokens, OBSERVABILITY_CLAIMS,
};

const WORKER_TOKEN: &str = "the-worker-token";
const URL: &str = "wss://project.livekit.cloud";

/// Registers the token mode settings once. The key and secret are registered as empty, so they
/// do not fall back to the environment.
fn serve_token_mode() {
    static SETTINGS: OnceLock<lk_agents_net::env::ServerSettings> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        lk_agents_net::env::serve_settings(vec![
            ("LIVEKIT_URL", URL.into()),
            ("LIVEKIT_TOKEN", WORKER_TOKEN.into()),
            ("LIVEKIT_API_KEY", String::new()),
            ("LIVEKIT_API_SECRET", String::new()),
        ])
    });
    assert!(lk_agents_net::token::uses_job_token("", ""));
}

/// A fake server: it issues `job-token-for-<claims>`.
fn server() -> Fetch {
    Box::new(|claims| {
        let expires_at = lk_agents_net::clock::unix_now() as i64 + 600;
        let token = JobToken { token: format!("job-token-for-{claims}"), expires_at };
        Box::pin(std::future::ready(Ok(token)))
    })
}

fn running_job(id: &str) -> JobKey {
    let job = JobKey(id.into());
    let clock = Arc::new(lk_agents_net::clock::RealClock::new());
    let tokens = Arc::new(JobTokens::new(server(), clock));
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    runtime.block_on(tokens.token(INFERENCE_CLAIMS)).unwrap();
    runtime.block_on(tokens.token(OBSERVABILITY_CLAIMS)).unwrap();
    job_token::open(job.clone(), tokens);
    job
}

// One test with two steps, because the registered settings apply to the whole binary.
#[test]
fn token_mode() {
    a_model_needs_no_key_and_a_request_outside_a_job_fails_clearly();
    a_jobs_requests_carry_its_tokens_and_a_given_key_still_signs();
}

// Builds models with no key in token mode, and makes requests with no job. Catches a model that
// fails to build without a key, since models are built in process setup before any job exists.
// Also catches a request outside a job that is sent without a token or with the worker token.
fn a_model_needs_no_key_and_a_request_outside_a_job_fails_clearly() {
    serve_token_mode();
    let credentials = Credentials::inference_credentials(None, None, None).unwrap();
    assert_eq!((credentials.api_key.as_str(), credentials.api_secret.as_str()), ("", ""));
    lk_agents_inference::interruption::InterruptionOptions::new(
        None,
        0.05,
        3.0,
        0.5,
        0.1,
        1.0,
        "https://gateway".into(),
        String::new(),
        String::new(),
    )
    .unwrap();
    let eot = lk_agents_inference::eot::select(Some("v1"), false, false, "https://x", "", "", true);
    assert_eq!(eot.unwrap().model, lk_agents_abi::tokenize::language::TurnDetectorModel::V1);

    let error = lk_agents_net::token::inference_bearer(None, "", "").unwrap_err();
    assert!(error.message.contains("no running job in token mode"), "{error}");
    let error =
        lk_agents_net::token::observability_bearer(&JobKey("AJ_none".into()), "", "").unwrap_err();
    assert!(error.message.contains("token mode"), "{error}");
}

// One running job with inference and telemetry tokens. Catches these bugs:
// - an inference request uses a credential other than the job's inference token, such as the
//   worker token;
// - telemetry does not use the job's telemetry token;
// - a model that has its own key and secret does not sign its own token.
fn a_jobs_requests_carry_its_tokens_and_a_given_key_still_signs() {
    serve_token_mode();
    let job = running_job("AJ_token_mode");
    let cx = Cx::NONE.with_job(Some(job.clone()));
    let auth = lk_agents_inference::llm::Auth::Gateway {
        api_key: String::new(),
        api_secret: String::new(),
    };
    let bearer = auth.bearer(&cx).unwrap();
    assert_eq!(bearer, format!("job-token-for-{INFERENCE_CLAIMS}"));
    assert_ne!(bearer, WORKER_TOKEN);

    let signed = lk_agents_net::token::inference_bearer(Some(&job), "key", "secret").unwrap();
    let claims = livekit_token_claims(&signed);
    assert!(claims.contains("\"perform\":true"), "{claims}");

    let telemetry = lk_agents_net::token::observability_bearer(&job, "", "").unwrap();
    assert_eq!(telemetry, format!("job-token-for-{OBSERVABILITY_CLAIMS}"));
    job_token::close(&job);
    assert!(auth.bearer(&cx).is_err(), "a closed job's token is gone");
}

/// Returns the payload of a signed token as JSON text.
fn livekit_token_claims(jwt: &str) -> String {
    use base64::Engine as _;
    let payload = jwt.split('.').nth(1).unwrap();
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload).unwrap();
    String::from_utf8(bytes).unwrap()
}
