//! The real LiveKit Inference gateway over its transport: the wire assumptions no
//! controlled peer can check (claim set, auth scheme, path, `include_usage`, error bodies).
//!
//! `#[ignore]`d (credentials and network). Run with:
//!
//! ```sh
//! set -a && . ~/livekit/.env && set +a
//! cargo test -p lk-agents-inference --test inference llm_live -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::{ChatContext, Role};
use lk_agents_inference::llm::LlmOptions;
use lk_agents_inference::llm::gateway::GatewayLlm;
use lk_agents_inference::llm::lifecycle::RequestPolicy;
use lk_agents_net::clock::RealClock;
use lk_agents_net::http::GatewayHttp;

fn llm(model: &str) -> Option<GatewayLlm> {
    let api_key = lk_agents_inference::credentials::resolve_api_key(
        None,
        std::env::var("LIVEKIT_INFERENCE_API_KEY").ok().as_deref(),
        std::env::var("LIVEKIT_API_KEY").ok().as_deref(),
    )
    .ok()?;
    let api_secret = lk_agents_inference::credentials::resolve_api_secret(
        None,
        std::env::var("LIVEKIT_INFERENCE_API_SECRET").ok().as_deref(),
        std::env::var("LIVEKIT_API_SECRET").ok().as_deref(),
    )
    .ok()?;
    let opts = LlmOptions {
        model: model.to_owned(),
        provider: None,
        base_url: lk_agents_inference::headers::default_inference_url(
            std::env::var("LIVEKIT_INFERENCE_URL").ok().as_deref(),
            std::env::var("LIVEKIT_URL").ok().as_deref(),
        ),
        api_key,
        api_secret,
        inference_class: None,
        extra_kwargs: [("max_completion_tokens".to_owned(), serde_json::json!(16))]
            .into_iter()
            .collect(),
    };
    let http = Arc::new(GatewayHttp::new().unwrap());
    Some(GatewayLlm::new(opts, http, tokio::runtime::Handle::current(), Arc::new(RealClock::new())))
}

fn policy() -> RequestPolicy {
    let limit = Some(Duration::from_secs(20));
    RequestPolicy {
        connect_timeout: limit,
        write_timeout: limit,
        read_timeout: limit,
        total_timeout: None,
        max_retry: 0,
        retry_interval: Duration::ZERO,
    }
}

#[allow(clippy::disallowed_methods, reason = "a live test stamps its message with wall time")]
fn chat_ctx(text: &str) -> ChatContext {
    ChatContext::new(vec![
        lk_agents_abi::ChatMessage::text("m1", Role::User, text, std::time::SystemTime::now())
            .into(),
    ])
}

// Catches: a claim set or auth scheme the gateway rejects, a wrong path, `include_usage`
// refused, or the framing losing a token.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LiveKit Inference credentials and the network"]
async fn a_streamed_completion_round_trips() {
    let Some(llm) = llm("openai/gpt-4o-mini") else { return eprintln!("skipped: no credentials") };
    let mut chunks = llm
        .chat(&chat_ctx("Say the single word: pong."), &[], Default::default(), policy(), None)
        .unwrap();
    let (mut text, mut usage) = (String::new(), None);
    while let Some(chunk) = chunks.next().await.expect("the gateway answered") {
        if let Some(content) = chunk.delta.as_ref().and_then(|d| d.content.as_ref()) {
            text.push_str(content);
        }
        usage = chunk.usage.or(usage);
    }
    assert!(!text.trim().is_empty(), "no text");
    let usage = usage.expect("include_usage produces a usage chunk");
    assert!(usage.input_tokens > 0 && usage.output_tokens > 0, "{usage:?}");
    eprintln!("live: {text:?} usage={usage:?}");
}

// Catches: a 4xx mapped to the wrong class, retried, or reported without the gateway's body.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs LiveKit Inference credentials and the network"]
async fn an_unknown_model_is_a_non_retryable_status_error() {
    let Some(llm) = llm("openai/definitely-not-a-model") else {
        return eprintln!("skipped: no credentials");
    };
    let mut chunks = llm.chat(&chat_ctx("hi"), &[], Default::default(), policy(), None).unwrap();
    let err = chunks.next().await.expect_err("an unknown model fails");
    eprintln!("live 4xx: {err:?}");
    assert!(err.status_code.is_some_and(|s| (400..500).contains(&s)), "{err:?}");
    assert!(!err.retryable && !err.message.is_empty());
}
