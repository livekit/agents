use super::pump::shifted;
use super::*;
use crate::testkit::{FakeWs, text};
use async_trait::async_trait;
use lk_agents_abi::SpeechEventType;
use lk_agents_net::task::TokioSpawner;
use serde_json::json;

fn opts() -> SttOptions {
    SttOptions {
        model: Some("deepgram/nova-3".into()),
        language: None,
        encoding: options::DEFAULT_ENCODING.into(),
        sample_rate: 16_000,
        base_url: "https://gw/v1".into(),
        api_key: "k".into(),
        api_secret: "s".into(),
        extra_kwargs: Default::default(),
        fallback: Vec::new(),
        conn_options: lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
    }
}

fn stt(ws: Arc<FakeWs>) -> GatewayStt {
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    GatewayStt::new(opts(), ws, spawner)
}

/// A client whose streams do not retry, for tests about one connection's failure.
fn stt_once(ws: Arc<FakeWs>) -> GatewayStt {
    let mut opts = opts();
    opts.conn_options.max_retry = 0;
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    GatewayStt::new(opts, ws, spawner)
}

// Catches: a failed connection ending the stream instead of reconnecting under
// `conn_options`, or the exhausted error not naming the attempts.
#[tokio::test]
async fn a_failed_connection_is_retried_then_reported_with_its_attempts() {
    let ws = Arc::new(FakeWs::failing(TransportError::Connect("refused".into())));
    let mut opts = opts();
    opts.conn_options =
        ApiConnectOptions { max_retry: 2, retry_interval: 0.0, ..Default::default() };
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let stt: GatewayStt = GatewayStt::new(opts, ws.clone(), spawner);
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    let err = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(5),
        source.next(),
    )
    .await
    .expect("the retries settle")
    .expect_err("every attempt failed");
    assert_eq!(ws.connect_count(), 3, "the first attempt and two retries");
    assert!(err.message().contains("after 3 attempts"), "{}", err.message());
}

// Catches: an option change or a keyterm change reaching only streams opened after it.
#[tokio::test]
async fn option_and_keyterm_changes_reach_an_open_stream() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt(ws.clone());
    let (_sink, _source) = stt.stream(None, None).await.unwrap().unwrap();
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    let mut extra = serde_json::Map::new();
    extra.insert("endpointing".into(), json!(300));
    stt.update_options(None, Some("fr"), Some(&extra));
    stt.update_keyterms(&["LiveKit".into()]);
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    let updates: Vec<_> =
        ws.sent_json().into_iter().filter(|f| f["type"] == json!("session.update")).collect();
    assert_eq!(updates.len(), 2, "{updates:?}");
    assert_eq!(updates[0]["settings"]["extra"]["endpointing"], json!(300), "{updates:?}");
    assert_eq!(updates[1]["settings"]["extra"]["keyterm"], json!(["LiveKit"]), "{updates:?}");
}

// Catches: a user turn forwarded, a model without context carry-over sent one, or the
// context not cut to its tail.
#[tokio::test]
async fn an_assistant_turn_reaches_an_open_stream_as_agent_context() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let mut opts = opts();
    opts.model = Some("assemblyai/universal-3-6-pro".into());
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let stt: GatewayStt = GatewayStt::new(opts, ws.clone(), spawner);
    let (_sink, _source) = stt.stream(None, None).await.unwrap().unwrap();
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    Stt::push_conversation_item(&stt, "user", "hello");
    Stt::push_conversation_item(&stt, "assistant", &"x".repeat(2000));
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    let updates: Vec<_> =
        ws.sent_json().into_iter().filter(|f| f["type"] == json!("session.update")).collect();
    assert_eq!(updates.len(), 1, "{updates:?}");
    let context = updates[0]["settings"]["extra"]["agent_context"].as_str().unwrap();
    assert_eq!(context.len(), MAX_AGENT_CONTEXT_CHARS);

    let plain = stt_once(Arc::new(FakeWs::new(vec![]).keep_open()));
    Stt::push_conversation_item(&plain, "assistant", "hi");
    assert!(plain.shared.opts.lock().unwrap().extra_kwargs.get("agent_context").is_none());
}

// Catches: transcripts of a stream opened mid-input left at zero on the caller's timeline.
#[test]
fn the_start_time_offset_moves_every_timestamp() {
    let mut event = SpeechEvent::new(SpeechEventType::FinalTranscript);
    event.alternatives.push(lk_agents_abi::SpeechData {
        start_time: 1.0,
        end_time: 2.0,
        words: Some(vec![lk_agents_abi::TimedString {
            start_time: Some(1.0),
            end_time: Some(1.5),
            ..lk_agents_abi::TimedString::new("hi")
        }]),
        ..lk_agents_abi::SpeechData::new("en", "hi")
    });
    let event = shifted(event, 10.0);
    let alt = &event.alternatives[0];
    assert_eq!((alt.start_time, alt.end_time), (11.0, 12.0));
    let word = &alt.words.as_ref().unwrap()[0];
    assert_eq!(
        (word.start_time, word.end_time, word.start_time_offset),
        (Some(11.0), Some(11.5), Some(10.0))
    );
}

// Catches: an unbounded handshake, or one bounded on tokio's clock rather than the session's.
#[tokio::test]
async fn a_handshake_that_never_completes_is_a_retryable_timeout_on_the_session_clock() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::hanging());
    let stt = stt_once(ws.clone()).with_clock(clock.clone());
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();

    crate::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    let err = source.next().await.expect_err("the handshake never completed");
    assert_eq!(err.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    assert!(err.retryable());
    assert!(err.message().contains("connection timed out"), "{}", err.message());
    assert!(ws.sent().is_empty(), "nothing was sent on a socket that never opened");
}

// Catches: bounding the handshake but not the `session.create` acknowledgement.
#[tokio::test]
async fn a_gateway_that_connects_but_never_acknowledges_the_session_is_a_retryable_timeout() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt_once(ws.clone()).with_clock(clock.clone());
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();

    crate::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    let err = source.next().await.expect_err("the gateway never answered session.create");
    assert_eq!(err.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    assert!(err.retryable());
    assert_eq!(ws.sent_json()[0]["type"], json!("session.create"), "the socket did open");
}

/// Drain events until `stop`, or fail; bounded so a bug fails rather than hangs.
async fn drain_until(
    source: &mut dyn StreamSource<SpeechEvent>,
    stop: SpeechEventType,
) -> Vec<SpeechEvent> {
    let mut out = Vec::new();
    let drained = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        std::time::Duration::from_millis(500),
        async {
            while let Some(event) = source.next().await.expect("no provider error") {
                let kind = event.kind;
                out.push(event);
                if kind == stop {
                    return;
                }
            }
        },
    )
    .await;
    assert!(
        drained.is_ok(),
        "never reached {stop:?}; got {:?}",
        out.iter().map(|e| e.kind).collect::<Vec<_>>()
    );
    out
}

fn frame(samples: usize) -> AudioFrame {
    AudioFrame::new(vec![0u8; samples * 2], 16_000, 1, samples as u32)
}

// Catches: opening the socket without `session.create`, or sending it after the first audio.
#[tokio::test]
async fn the_session_is_created_before_any_audio_is_sent() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "session.created"}))]).keep_open());
    let stt = stt(ws.clone());
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    sink.push(frame(800)).unwrap();
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    ws.feed(text(json!({"type": "final_transcript", "transcript": "hello"})));
    let event = source.next().await.unwrap().unwrap();
    assert_eq!(event.kind, SpeechEventType::StartOfSpeech);
    let sent = ws.sent_json();
    assert_eq!(sent[0]["type"], json!("session.create"));
    assert_eq!(sent[1]["type"], json!("input_audio"));
    assert_eq!(
        source.close(stt.shared.clock.now().saturating_add(Duration::from_secs(5))).await,
        lk_agents_abi::Closed::Clean
    );
}

// Catches: the model in the query string but not the auth header, or the reverse.
#[tokio::test]
async fn the_handshake_carries_the_model_in_the_url_and_a_bearer_token() {
    let ws = Arc::new(FakeWs::new(vec![]));
    let stt = stt(ws.clone());
    let (_sink, _source) = stt.stream(None, None).await.unwrap().unwrap();
    tokio::task::yield_now().await;
    let (url, headers) = ws.last_connect().expect("connected");
    assert_eq!(url, "wss://gw/v1/stt?model=deepgram/nova-3");
    let auth = headers.iter().find(|(k, _)| k == "Authorization").unwrap();
    assert!(auth.1.starts_with("Bearer ey"), "{}", auth.1);
}

// Catches: the host's identification headers not reaching the socket, or the transcript
// request id not taken from the host's session id.
#[tokio::test]
async fn the_hosts_identification_headers_ride_on_the_handshake() {
    let ws = Arc::new(FakeWs::new(vec![]));
    let ident: Headers = [
        ("X-LiveKit-Room-ID", "RM_1"),
        ("X-LiveKit-Job-ID", "AJ_1"),
        ("X-LiveKit-Session-ID", "inference_abc"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let client = stt(ws.clone()).with_identification(ident);
    let (_sink, _source) = client.stream(None, None).await.unwrap().unwrap();
    tokio::task::yield_now().await;
    let (_, headers) = ws.last_connect().expect("connected");
    let get = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    assert_eq!(get("X-LiveKit-Room-ID"), Some("RM_1"));
    assert_eq!(get("X-LiveKit-Job-ID"), Some("AJ_1"));
    // A fresh session id per connection, as the reference mints one per call.
    let session = get("X-LiveKit-Session-ID").expect("a session id");
    assert!(session.starts_with("inference_") && session != "inference_abc", "{session}");
    assert!(get("Authorization").is_some());

    // No job: no `X-LiveKit-*` header.
    let ws = Arc::new(FakeWs::new(vec![]));
    let (_sink, _source) = stt(ws.clone()).stream(None, None).await.unwrap().unwrap();
    tokio::task::yield_now().await;
    let (_, headers) = ws.last_connect().expect("connected");
    assert!(!headers.iter().any(|(k, _)| k.starts_with("X-LiveKit-")), "{headers:?}");
}

// Catches: reporting a handshake rejection as a clean end of stream.
#[tokio::test]
async fn a_rejected_handshake_surfaces_as_an_error_not_an_empty_stream() {
    let ws = Arc::new(FakeWs::failing(TransportError::Status {
        status: 401,
        message: "Unauthorized".into(),
        headers: vec![],
        body: None,
    }));
    let stt = stt(ws);
    let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    let err = source.next().await.expect_err("401");
    assert_eq!(err.status_code(), Some(401));
    assert!(!err.retryable());
}

// Catches: forwarding audio unchunked instead of 50 ms frames (`sample_rate // 20`).
#[tokio::test]
async fn audio_is_reframed_to_fifty_millisecond_chunks() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "session.created"}))]).keep_open());
    let stt = stt(ws.clone());
    let (sink, _source) = stt.stream(None, None).await.unwrap().unwrap();
    // 10 ms frames: five of them make exactly one 50 ms gateway frame.
    for _ in 0..5 {
        sink.push(frame(160)).unwrap();
    }
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    let audio: Vec<_> =
        ws.sent_json().into_iter().filter(|v| v["type"] == json!("input_audio")).collect();
    assert_eq!(audio.len(), 1, "five 10ms frames make one 50ms frame");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(audio[0]["audio"].as_str().unwrap())
        .unwrap();
    assert_eq!(decoded.len(), 16_000 / 20 * 2);
}

// Catches: the send half ending the stream on `end_input`, losing the final transcript. The
// transcript is fed after the finalize, as in the real ordering.
#[tokio::test]
async fn closing_the_input_still_delivers_the_pending_final_transcript() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt(ws.clone());
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    sink.push(frame(800)).unwrap();
    sink.end_input();

    // Wait for the finalize to reach the socket, then answer it.
    for _ in 0..50 {
        if ws.sent_json().iter().any(|v| v["type"] == json!("session.finalize")) {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert!(
        ws.sent_json().iter().any(|v| v["type"] == json!("session.finalize")),
        "end_input must finalize the session: {:?}",
        ws.sent_json()
    );
    ws.feed(text(json!({"type": "final_transcript", "transcript": "bye"})));

    let events = drain_until(source.as_mut(), SpeechEventType::EndOfSpeech).await;
    let kinds: Vec<_> = events.iter().map(|e| e.kind).collect();
    assert!(kinds.contains(&SpeechEventType::FinalTranscript), "got {kinds:?}");
}

// Catches: a stream whose input ended never ending, since the gateway keeps the socket open
// after `session.finalize` (a `stt.stream()` read to its end hung forever against the real
// gateway): lack of an explicit finalization acknowledgment fails at the settlement deadline.
#[tokio::test]
async fn missing_completion_after_input_eof_fails_instead_of_clean_eof() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt_once(ws.clone()).with_clock(clock.clone());
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    sink.push(frame(800)).unwrap();
    sink.end_input();
    ws.feed(text(json!({"type":"session.created"})));
    ws.feed(text(json!({"type": "final_transcript", "transcript": "bye"})));
    let events = drain_until(source.as_mut(), SpeechEventType::EndOfSpeech).await;
    assert!(events.iter().any(|e| e.kind == SpeechEventType::FinalTranscript));

    crate::testkit::arm_then_advance(&clock, FINALIZATION_TIMEOUT / 2).await;
    let early = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(50),
        source.next(),
    )
    .await;
    assert!(early.is_err(), "ended before the inactivity timeout: {early:?}");
    crate::testkit::arm_then_advance(&clock, FINALIZATION_TIMEOUT).await;
    let end = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        source.next(),
    )
    .await
    .expect("the stream never ended after its input did");
    assert!(end.is_err(), "missing completion was reported as clean EOF: {end:?}");
}

// Catches: finalizing on the wrong VAD event (only END_OF_SPEECH), or on none.
#[tokio::test]
async fn only_the_vads_end_of_speech_finalizes_the_session() {
    use crate::abi::{Vad, VadCapabilities, VadSink, VadSource};
    use lk_agents_abi::{VadEvent, VadEventType};

    struct ScriptedVad(Vec<VadEventType>);
    impl Vad for ScriptedVad {
        fn label(&self) -> String {
            "test".into()
        }
        fn capabilities(&self) -> VadCapabilities {
            VadCapabilities::default()
        }
        fn stream(
            &self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(VadSink, VadSource), ProviderError>> {
            let source = ScriptedSource(self.0.clone().into_iter().collect());
            Box::pin(
                async move { Ok((Box::new(NullSink) as VadSink, Box::new(source) as VadSource)) },
            )
        }
    }
    struct NullSink;
    impl StreamSink<AudioFrame> for NullSink {
        fn push(&self, _: AudioFrame) -> Result<(), ProviderError> {
            Ok(())
        }
        fn flush(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        fn end_input(&self) {}
    }
    struct ScriptedSource(std::collections::VecDeque<VadEventType>);
    impl StreamSource<VadEvent> for ScriptedSource {
        fn next(
            &mut self,
        ) -> lk_agents_abi::BoxFuture<'_, Result<Option<VadEvent>, ProviderError>> {
            Box::pin(async move {
                Ok(self.0.pop_front().map(|kind| VadEvent::new(kind, 0, 0.0, 0.0, 0.0)))
            })
        }
        fn close(
            self: Box<Self>,
            _deadline: lk_agents_abi::Ts,
        ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
            Box::pin(async move { lk_agents_abi::Closed::Clean })
        }
    }

    // Three inference-done events and one end-of-speech: exactly one finalize.
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let vad = Arc::new(ScriptedVad(vec![
        VadEventType::InferenceDone,
        VadEventType::InferenceDone,
        VadEventType::StartOfSpeech,
        VadEventType::EndOfSpeech,
        VadEventType::InferenceDone,
    ]));
    let stt = stt(ws.clone()).with_vad(vad);
    let (sink, _source) = stt.stream(None, None).await.unwrap().unwrap();
    sink.push(frame(800)).unwrap();
    for _ in 0..60 {
        tokio::task::yield_now().await;
    }
    let finalizes =
        ws.sent_json().iter().filter(|v| v["type"] == json!("session.finalize")).count();
    assert_eq!(finalizes, 1, "sent {:?}", ws.sent_json());
}

// Catches: the billed audio duration computed wrong (0.05 s per 50 ms frame).
#[tokio::test]
async fn the_usage_record_reports_the_real_audio_duration() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt(ws.clone());
    let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
    // Four 50 ms frames = 0.2 s.
    for _ in 0..4 {
        sink.push(frame(800)).unwrap();
    }
    for _ in 0..40 {
        tokio::task::yield_now().await;
    }
    ws.feed(text(json!({"type": "final_transcript", "transcript": "hi"})));

    let events = drain_until(source.as_mut(), SpeechEventType::EndOfSpeech).await;
    let usage = events
        .iter()
        .find(|e| e.kind == SpeechEventType::RecognitionUsage)
        .and_then(|e| e.recognition_usage)
        .expect("a usage record");
    assert!((usage.audio_duration - 0.2).abs() < 1e-9, "got {}", usage.audio_duration);
}

// Catches: reporting the default `offline_recognize`, which routes the session to a batch
// `recognize()` the gateway does not have.
#[tokio::test]
async fn the_gateway_stt_reports_no_batch_recognition() {
    let stt = stt(Arc::new(FakeWs::new(vec![])));
    assert!(!Stt::capabilities(&stt).offline_recognize);
}

// Catches: a mid-stream `session.update` starved by continuous audio.
#[tokio::test]
async fn a_session_update_is_not_starved_by_the_audio_stream() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let stt = stt(ws.clone());
    let (sink, _source) = stt.stream_native(None, 0.0).unwrap();
    for _ in 0..50 {
        sink.push(frame(800)).unwrap();
    }
    sink.control.update_options(Some("deepgram/nova-2"), None, None);
    for _ in 0..200 {
        tokio::task::yield_now().await;
    }
    let sent = ws.sent_json();
    let update_at = sent.iter().position(|v| v["type"] == json!("session.update"));
    let first_audio = sent.iter().position(|v| v["type"] == json!("input_audio"));
    assert!(update_at.is_some(), "the update was never sent");
    assert!(first_audio.is_some());
    assert!(update_at < first_audio, "queued audio went out ahead of the update: {sent:?}");
}

// Catches: applying a keyterm change mid-utterance instead of at END_OF_SPEECH.
#[tokio::test]
async fn a_keyterm_change_mid_utterance_waits_for_the_end_of_speech() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "start_of_speech"}))]).keep_open());
    let stt = stt(ws.clone());
    let (sink, mut source) = stt.stream_native(None, 0.0).unwrap();

    // Let the start_of_speech land, then change the keyterms.
    assert_eq!(source.next().await.unwrap().unwrap().kind, SpeechEventType::StartOfSpeech);
    let extra = stt.update_session_keyterms(&["livekit".to_string()]).unwrap();
    sink.control.apply_keyterms(extra);
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    assert!(
        !ws.sent_json().iter().any(|v| v["type"] == json!("session.update")),
        "the update must not be sent while the user is speaking"
    );

    // Now end the turn; the parked update is released there.
    ws.feed(text(json!({"type": "final_transcript", "transcript": "done"})));
    drain_until(&mut source, SpeechEventType::EndOfSpeech).await;
    for _ in 0..20 {
        tokio::task::yield_now().await;
    }
    let update = ws
        .sent_json()
        .into_iter()
        .find(|v| v["type"] == json!("session.update"))
        .expect("released at end of speech");
    assert_eq!(update["settings"]["extra"]["keyterm"], json!(["livekit"]));
}

// Catches: re-sending an unchanged keyterm list, or sending one to a provider without keyterms.
#[test]
fn session_keyterms_are_only_applied_when_they_change() {
    let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let _guard = rt.enter();
    let stt = stt(Arc::new(FakeWs::new(vec![])));
    let terms = vec!["livekit".to_string()];
    assert!(stt.update_session_keyterms(&terms).is_some());
    assert!(stt.update_session_keyterms(&terms).is_none(), "unchanged");

    stt.update_options(Some("cartesia/ink-whisper"), None, None);
    assert!(
        stt.update_session_keyterms(&["other".to_string()]).is_none(),
        "cartesia has no keyterm prompting"
    );
}

#[tokio::test]
async fn stt_pump_close_saturation_joins_or_retains_incomplete_cleanup() {
    use lk_agents_net::transport::{WsReceiver, WsTransport};
    struct ObservedWs {
        inner: FakeWs,
        closing: tokio::sync::watch::Sender<bool>,
        released: tokio::sync::watch::Receiver<bool>,
    }
    struct ObservedSender {
        inner: Arc<dyn WsSender>,
        closing: tokio::sync::watch::Sender<bool>,
        released: tokio::sync::watch::Receiver<bool>,
    }
    #[async_trait]
    impl WsSender for ObservedSender {
        async fn send_text(&self, text: String) -> Result<(), TransportError> {
            self.inner.send_text(text).await
        }
        async fn send_binary(&self, data: Vec<u8>) -> Result<(), TransportError> {
            self.inner.send_binary(data).await
        }
        fn is_closed(&self) -> bool {
            self.inner.is_closed()
        }
        async fn close(&self) {
            self.closing.send_replace(true);
            let _ = self.released.clone().wait_for(|released| *released).await;
            self.inner.close().await;
        }
    }
    #[async_trait]
    impl WsTransport for ObservedWs {
        async fn connect(
            &self,
            url: &str,
            headers: Vec<(String, String)>,
        ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
            let (inner, receiver) = self.inner.connect(url, headers).await?;
            Ok((
                Arc::new(ObservedSender {
                    inner,
                    closing: self.closing.clone(),
                    released: self.released.clone(),
                }),
                receiver,
            ))
        }
    }
    for releases in [true, false] {
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let (closing, mut observed) = tokio::sync::watch::channel(false);
        let (release, released) = tokio::sync::watch::channel(releases);
        let ws = Arc::new(ObservedWs { inner: FakeWs::new(vec![]).keep_open(), closing, released });
        let provider: GatewayStt = GatewayStt::new(
            opts(),
            ws.clone(),
            Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())),
        )
        .with_clock(clock.clone());
        let (_sink, source) = provider.stream(None, None).await.unwrap().unwrap();
        ws.inner.feed(text(json!({"type":"session.created"})));
        for _ in 0..lk_agents_abi::AUDIO_QUEUE_FRAMES {
            ws.inner.feed(text(json!({"type":"final_transcript","transcript":"hi"})));
        }
        for _ in 0..128 {
            tokio::task::yield_now().await;
        }
        let deadline = clock.now().saturating_add(Duration::from_secs(1));
        let close = source.close(deadline);
        tokio::pin!(close);
        if releases {
            assert_eq!(close.await, lk_agents_abi::Closed::Clean);
            assert!(*observed.borrow(), "clean close preceded socket settlement");
        } else {
            match futures_util::future::select(
                Box::pin(observed.wait_for(|closing| *closing)),
                close.as_mut(),
            )
            .await
            {
                futures_util::future::Either::Left(_) => {}
                futures_util::future::Either::Right((result, _)) => {
                    panic!("close completed before socket cleanup: {result:?}")
                }
            }
            clock.advance(Duration::from_secs(1));
            assert_eq!(close.await, lk_agents_abi::Closed::TimedOut);
            release.send_replace(true);
            for _ in 0..32 {
                tokio::task::yield_now().await;
            }
            assert_eq!(
                release.receiver_count(),
                1,
                "the pump retained cleanup after timeout and eventually joined"
            );
        }
    }
}

#[tokio::test]
async fn a_call_policy_replaces_native_stt_defaults_without_mutating_the_client() {
    let ws = Arc::new(FakeWs::failing(TransportError::Connect("refused".into())));
    let client: Arc<dyn Stt> = Arc::new(stt(ws.clone()));
    let conn = ApiConnectOptions { max_retry: 0, ..Default::default() };
    let call = client.with_conn_options(&conn).unwrap_or_else(|| client.clone());
    let (_, mut source) = call.stream(None, None).await.unwrap().unwrap();
    let failure = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(250),
        source.next(),
    )
    .await
    .expect("the call must finish after its one allowed attempt");
    assert!(failure.is_err());
    assert_eq!(ws.connect_count(), 1);
    source.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
}

#[tokio::test]
async fn input_after_a_terminal_timeout_preserves_its_phase() {
    let ws = Arc::new(FakeWs::hanging());
    let mut options = opts();
    options.conn_options =
        ApiConnectOptions { max_retry: 0, total_timeout: Some(0.0), ..Default::default() };
    let client: GatewayStt = GatewayStt::new(options, ws, Arc::new(TokioSpawner::current()));
    let (sink, mut source) = client.stream(None, None).await.unwrap().unwrap();
    let error = source.next().await.expect_err("the operation expired before launch");
    assert_eq!(error.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    let error = sink.flush().expect_err("input cannot revive a failed operation");
    let ProviderError::Local(error) = error else { panic!("a native failure") };
    assert_eq!(error.phase, Some(lk_agents_abi::errors::TimeoutPhase::Total));
    source.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
}

/// Yield until `ready`, or fail; bounded so a bug fails rather than hangs.
async fn until(what: &str, ready: impl Fn() -> bool) {
    let reached =
        lk_agents_abi::timeout(&lk_agents_abi::RealClock::new(), Duration::from_secs(2), async {
            while !ready() {
                tokio::task::yield_now().await;
            }
        })
        .await;
    assert!(reached.is_ok(), "never {what}");
}

// Catches: a live recognition that dies when the gateway drops it mid-session after it already
// delivered transcripts and audio is in flight. A reconnect sends only new audio, so it repeats
// nothing; only a dropped stream whose input already ended has nothing left to recognize.
#[tokio::test]
async fn a_session_dropped_before_the_input_ends_reconnects_after_delivered_transcripts() {
    let dropped = [Ok(WsMessage::Closed), text(json!({"type": "session.closed"}))];
    for drop in dropped {
        let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "session.created"}))]).keep_open());
        let stt = stt(ws.clone());
        let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
        sink.push(frame(800)).unwrap();
        ws.feed(text(json!({"type": "final_transcript", "transcript": "first"})));
        drain_until(source.as_mut(), SpeechEventType::FinalTranscript).await;
        sink.push(frame(800)).unwrap();
        let audio = |ws: &FakeWs| {
            ws.sent_json().iter().filter(|v| v["type"] == json!("input_audio")).count()
        };
        until("sent the second frame", || audio(&ws) == 2).await;
        ws.feed(drop.clone());
        until("reconnected", || ws.connect_count() == 2).await;

        ws.feed(text(json!({"type": "session.created"})));
        sink.push(frame(800)).unwrap();
        sink.end_input();
        until("finalized", || {
            ws.sent_json().iter().any(|v| v["type"] == json!("session.finalize"))
        })
        .await;
        ws.feed(text(json!({"type": "final_transcript", "transcript": "second"})));
        ws.feed(text(json!({"type": "session.finalized"})));
        let events = drain_until(source.as_mut(), SpeechEventType::FinalTranscript).await;
        assert_eq!(events.last().unwrap().alternatives[0].text, "second");
        let end = lk_agents_abi::timeout(
            &lk_agents_abi::RealClock::new(),
            Duration::from_secs(2),
            async {
                loop {
                    match source.next().await {
                        Ok(Some(_)) => continue,
                        other => return other,
                    }
                }
            },
        )
        .await
        .expect("the stream ends");
        assert!(matches!(end, Ok(None)), "{drop:?}: {end:?}");
        source.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
    }
}

// Catches: a recognition that failed after its input ended reconnecting: the new session has no
// audio to send, finalizes empty and reports clean success with the transcript lost.
#[tokio::test]
async fn a_session_failing_after_its_input_ended_is_not_retried() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "session.created"}))]).keep_open());
    let (sink, mut source) = stt(ws.clone()).stream(None, None).await.unwrap().unwrap();
    sink.push(frame(800)).unwrap();
    sink.end_input();
    until("finalized", || ws.sent_json().iter().any(|v| v["type"] == json!("session.finalize")))
        .await;
    ws.feed(text(json!({"type": "error", "code": 2006})));
    let error = source.next().await.expect_err("the session failed");
    assert!(!error.retryable());
    assert_eq!(ws.connect_count(), 1);
}
