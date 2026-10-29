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
//! The LLM's live checks run over the runtime's transport (`lk-agents-runtime/tests/llm_live.rs`).
//!
//! What they are for is narrow and worth stating: the 111 unit tests pin the *logic*, and nothing
//! in them can catch a wrong assumption about the wire — an auth scheme, a URL shape, a header
//! name the gateway lower-cases. Each test below is one such assumption.

use std::sync::Arc;

fn credentials() -> Option<(String, String, String)> {
    // The *inference* key pair, which is distinct from the room key pair. `resolve_credential`
    // reproduces the reference's fallback (§creds), and the point of using it here rather than
    // reading the vars directly is that the fallback itself gets exercised against a real key.
    let key = lk_agents_net::credentials::resolve_api_key(None, None).ok()?;
    let secret = lk_agents_net::credentials::resolve_api_secret(None, None).ok()?;
    let url = lk_agents_inference::default_inference_url_from_env();
    Some((key, secret, url))
}

/// Catches: the STT handshake and `session.create` being wrong. Silence produces no transcript,
/// so what this asserts is narrower than the TTS test and deliberately so: the gateway must
/// **accept** the session and the audio frames rather than closing on a malformed opening frame.
/// A rejected `session.create` arrives as an unexpected close, which is exactly what would
/// otherwise be mistaken for "the user said nothing".
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn an_stt_session_opens_and_accepts_audio() {
    use lk_agents_abi::SpeechEventType;
    use lk_agents_inference::abi::{AudioFrame, Stt};
    use lk_agents_inference::stt::{GatewayStt, SttOptions};
    use lk_agents_net::websocket::TungsteniteTransport;

    let Some((api_key, api_secret, base_url)) = credentials() else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };

    let stt: GatewayStt = GatewayStt::new(
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
        Arc::new(lk_agents_net::task::TokioSpawner::new(tokio::runtime::Handle::current())),
    );

    let (sink, mut source) = stt
        .stream(None, None)
        .await
        .expect("the socket opened")
        .expect("gateway STT is always duplex");
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
