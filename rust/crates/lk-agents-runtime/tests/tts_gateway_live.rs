//! Real gateway coverage for the runtime-selected native speech factory.

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

/// Catches: everything the WebSocket half assumes and the unit tests fake — the `?model=` query
/// parameter, the `Authorization` header surviving the handshake (a WS upgrade drops headers
/// silently in several client libraries), `"type"` being the first key of `session.create`, and
/// the `input_transcript` → `output_audio` → `done` sequence.
///
/// The LLM's happy path proves the HTTP transport; this proves the other one, and they share no
/// code below [`lk_agents_net::transport`].
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn a_tts_synthesis_round_trips_over_the_websocket() {
    use lk_agents_inference::abi::Tts;
    use lk_agents_inference::tts::TtsOptions;
    use lk_agents_net::websocket::TungsteniteTransport;
    use lk_agents_runtime::inference::tts::GatewayTts;

    let Some((api_key, api_secret, base_url)) = credentials() else {
        eprintln!("skipped: no inference credentials in the environment");
        return;
    };

    let tts = GatewayTts::new(
        TtsOptions {
            model: "cartesia/sonic-3".into(),
            base_url,
            api_key,
            api_secret,
            ..Default::default()
        },
        Arc::new(TungsteniteTransport),
        Arc::new(lk_agents_net::task::TokioSpawner::new(tokio::runtime::Handle::current())),
        Arc::new(lk_agents_abi::RealClock::new()),
        lk_agents_runtime::process::env().speech_text,
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
