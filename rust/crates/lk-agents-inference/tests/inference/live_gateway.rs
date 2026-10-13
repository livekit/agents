//! Tier 4 (§4): the real gateway, over the real transport.
//!
//! `#[ignore]`d, because CI has no credentials and these are the only tests in the crate that can
//! fail for a reason that is not a bug. Run them with:
//!
//! ```sh
//! set -a && . ~/livekit/.env && set +a
//! cargo test -p lk-agents-inference --test inference live_gateway -- --ignored --nocapture
//! ```
//!
//! What they are for is narrow and worth stating: the 111 unit tests pin the *logic*, and nothing
//! in them can catch a wrong assumption about the wire — an auth scheme, a URL shape, a header
//! name the gateway lower-cases. Each test below is one such assumption.

use lk_agents_inference::abi::Llm;
use lk_agents_inference::llm::{LlmOptions, NativeLlm};
use lk_agents_inference::native::ReqwestTransport;
use lk_agents_inference::testkit::NoImages;
use lk_agents_abi::{ChatContext, Role};
use std::sync::Arc;

fn credentials() -> Option<(String, String, String)> {
    // The *inference* key pair, which is distinct from the room key pair. `resolve_credential`
    // reproduces the reference's fallback (§creds), and the point of using it here rather than
    // reading the vars directly is that the fallback itself gets exercised against a real key.
    let key = lk_agents_inference::creds::resolve_api_key(None).ok()?;
    let secret = lk_agents_inference::creds::resolve_api_secret(None).ok()?;
    let url = lk_agents_inference::default_inference_url_from_env();
    Some((key, secret, url))
}

fn options(model: &str) -> Option<LlmOptions> {
    let (api_key, api_secret, base_url) = credentials()?;
    Some(LlmOptions {
        model: model.to_owned(),
        provider: None,
        base_url,
        api_key,
        api_secret,
        inference_class: None,
        extra_kwargs: [("max_completion_tokens".to_owned(), serde_json::json!(16))]
            .into_iter()
            .collect(),
    })
}

fn chat_ctx(text: &str) -> ChatContext {
    ChatContext::new(vec![
        lk_agents_abi::ChatMessage::text("m1", Role::User, text, std::time::SystemTime::now())
            .into(),
    ])
}


/// Catches: every wire assumption the unit tests cannot — the JWT's claim set being one the
/// gateway rejects, `Authorization: Bearer` being the wrong scheme, `/chat/completions` being the
/// wrong path, `stream_options.include_usage` being refused, and the SSE framing losing a token.
/// A green unit suite proves none of these.
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn a_streamed_completion_round_trips() {
    let Some(opts) = options("openai/gpt-4o-mini") else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };
    let llm = NativeLlm::new(opts, Arc::new(ReqwestTransport::new().unwrap()), Arc::new(NoImages));

    let mut stream = llm
        .chat(&lk_agents_abi::ChatRequest {
            chat_ctx: chat_ctx("Say the single word: pong."),
            ..Default::default()
        })
        .await
        .expect("chat() accepted the request");
    let mut text = String::new();
    let mut usage = None;
    while let Some(chunk) = stream.next().await.expect("the gateway answered") {
        if let Some(delta) = &chunk.delta
            && let Some(content) = &delta.content
        {
            text.push_str(content);
        }
        if chunk.usage.is_some() {
            usage = chunk.usage;
        }
    }

    assert!(!text.trim().is_empty(), "the stream produced no text at all");
    let usage = usage.expect("stream_options.include_usage must produce a usage chunk");
    assert!(usage.prompt_tokens > 0, "prompt tokens were not reported: {usage:?}");
    assert!(usage.total_tokens > 0);
    eprintln!("live: {text:?} usage={usage:?}");
}

/// Catches: a 4xx being mapped to the wrong taxonomy class, and the gateway's own error body
/// being replaced by a bare status phrase. The coordinator's `404 page not found` (a doubled
/// `/v1`) is indistinguishable from a wrong host unless the body survives.
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn an_unknown_model_is_a_non_retryable_status_error() {
    let Some(opts) = options("openai/definitely-not-a-model") else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };
    let llm = NativeLlm::new(opts, Arc::new(ReqwestTransport::new().unwrap()), Arc::new(NoImages));
    let mut stream = llm
        .chat(&lk_agents_abi::ChatRequest { chat_ctx: chat_ctx("hi"), ..Default::default() })
        .await
        .expect("chat() accepted the request");
    let err = stream.next().await.expect_err("an unknown model must fail");
    eprintln!("live 4xx: status={:?} message={:?}", err.status_code(), err.message());
    let status = err.status_code().expect("a status code, not a bare connection error");
    assert!((400..500).contains(&status), "expected a 4xx, got {status}");
    assert!(!err.retryable(), "a 4xx other than 408/409/429 must not be retried");
    assert!(!err.message().is_empty());
}

/// Catches: everything the WebSocket half assumes and the unit tests fake — the `?model=` query
/// parameter, the `Authorization` header surviving the handshake (a WS upgrade drops headers
/// silently in several client libraries), `"type"` being the first key of `session.create`, and
/// the `input_transcript` → `output_audio` → `done` sequence.
///
/// The LLM's happy path proves the HTTP transport; this proves the other one, and they share no
/// code below [`lk_agents_inference::transport`].
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn a_tts_synthesis_round_trips_over_the_websocket() {
    use lk_agents_inference::abi::Tts;
    use lk_agents_inference::native::TungsteniteTransport;
    use lk_agents_inference::tts::{NativeTts, TtsOptions};

    let Some((api_key, api_secret, base_url)) = credentials() else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };

    let tts = NativeTts::new(
        TtsOptions {
            model: "cartesia/sonic-3".into(),
            base_url,
            api_key,
            api_secret,
            ..Default::default()
        },
        Arc::new(TungsteniteTransport),
        Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current())),
    );

    let mut source = tts.synthesize("Hello from Rust.").await.expect("the socket opened");
    let mut bytes = 0usize;
    let mut segments = std::collections::BTreeSet::new();
    let mut saw_final = false;
    while let Some(audio) = source.next().await.expect("the gateway answered") {
        bytes += audio.frame.data.len();
        segments.insert(audio.segment_id.clone());
        saw_final |= audio.is_final;
    }
    eprintln!("live tts: {bytes} bytes over segments {segments:?}");
    assert!(saw_final, "the `done` frame must produce a final chunk");
    assert!(bytes > 4_000, "expected roughly a second of 24 kHz PCM, got {bytes} bytes");
    assert_eq!(segments.len(), 1, "one synthesis is one segment: {segments:?}");
}

/// Catches: the STT handshake and `session.create` being wrong. Silence produces no transcript,
/// so what this asserts is narrower than the TTS test and deliberately so: the gateway must
/// **accept** the session and the audio frames rather than closing on a malformed opening frame.
/// A rejected `session.create` arrives as an unexpected close, which is exactly what would
/// otherwise be mistaken for "the user said nothing".
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn an_stt_session_opens_and_accepts_audio() {
    use lk_agents_inference::abi::{AudioFrame, Stt};
    use lk_agents_inference::native::TungsteniteTransport;
    use lk_agents_inference::stt::{NativeStt, SttOptions};
    use lk_agents_abi::SpeechEventType;

    let Some((api_key, api_secret, base_url)) = credentials() else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };

    let stt = NativeStt::new(
        SttOptions {
            model: Some("deepgram/nova-3".into()),
            language: None,
            encoding: lk_agents_inference::stt::options::DEFAULT_ENCODING.into(),
            sample_rate: 16_000,
            base_url,
            api_key,
            api_secret,
            extra_kwargs: Default::default(),
            fallback: Vec::new(),
            conn_options: lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
        },
        Arc::new(TungsteniteTransport),
        Arc::new(lk_agents_abi::TokioSpawner::new(tokio::runtime::Handle::current())),
    );

    let (sink, mut source) =
        stt.stream(None, None).await.expect("gateway STT is always duplex").expect("the socket opened");
    // Half a second of silence, in 20 ms frames, then close the input.
    for _ in 0..25 {
        sink.push(AudioFrame::new(vec![0u8; 16_000 / 50 * 2], 16_000, 1, 16_000 / 50))
            .expect("the stream is open");
    }
    sink.end_input();

    let mut events = Vec::new();
    let drained = tokio::time::timeout(std::time::Duration::from_secs(15), async {
        while let Some(event) = source.next().await? {
            let kind = event.kind;
            events.push(kind);
            if kind == SpeechEventType::EndOfSpeech {
                break;
            }
        }
        Ok::<_, lk_agents_abi::errors::ProviderError>(())
    })
    .await;

    eprintln!("live stt: {drained:?} events={events:?}");
    match drained {
        Ok(Ok(())) => {}
        Ok(Err(err)) => panic!("the gateway rejected the session: {err}"),
        Err(_) => panic!("the gateway never finalized the turn; got {events:?}"),
    }

    // Silence produces an *empty* final transcript, which is the exact case
    // `stt::machine`'s `an_empty_final_transcript_still_ends_the_turn` pins: the gateway really
    // does send one, and the whole turn has to close out on it.
    assert_eq!(
        events,
        [
            SpeechEventType::StartOfSpeech,
            SpeechEventType::RecognitionUsage,
            SpeechEventType::FinalTranscript,
            SpeechEventType::EndOfSpeech,
        ],
        "the live turn sequence changed"
    );
}
