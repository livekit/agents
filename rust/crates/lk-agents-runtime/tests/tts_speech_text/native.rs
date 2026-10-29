use super::NativeTts;
use base64::Engine as _;
use lk_agents_abi::{ApiConnectOptions, Clock, StreamSink as _, StreamSource as _, Tts};
use lk_agents_inference::headers::Headers;
use lk_agents_inference::testkit::{FakeWs, text};
use lk_agents_inference::tts::{AUDIO_QUEUE_CAPACITY, TEXT_QUEUE_CAPACITY, TtsOptions};
use lk_agents_net::task::TokioSpawner;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

fn opts() -> TtsOptions {
    TtsOptions {
        model: "cartesia/sonic-3".into(),
        base_url: "https://gw/v1".into(),
        api_key: "k".into(),
        api_secret: "s".into(),
        ..Default::default()
    }
}

fn tts(ws: Arc<FakeWs>) -> NativeTts {
    NativeTts::new(
        opts(),
        ws,
        Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())),
        super::speech(),
    )
}

fn map(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
    pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect()
}

#[tokio::test]
async fn malformed_audio_is_a_failure_even_when_followed_by_done() {
    for frame in [
        json!({"type": "output_audio"}),
        json!({"type": "output_audio", "audio": "%%%"}),
        json!({"type": "output_audio", "audio": pcm(3)}),
    ] {
        let ws = Arc::new(FakeWs::new(vec![text(frame), text(json!({"type": "done"}))]));
        let mut source = tts(ws).synthesize("hi").await.unwrap();
        assert!(source.next().await.is_err(), "malformed audio cannot become successful output");
    }
}

#[tokio::test]
async fn done_before_the_final_flush_cannot_finish_a_synthesis() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let (sink, mut source) = tts(ws.clone()).stream().await.unwrap();
    sink.push("Hello there, this is the first sentence.".to_owned()).unwrap();
    sink.flush().unwrap();
    until_sent(&ws, 2).await;
    ws.feed(text(json!({"type": "done"})));
    assert!(source.next().await.is_err(), "the application still owns open input");
}

// Catches: an unbounded handshake, or one bounded on tokio's clock rather than the session's.
#[tokio::test]
async fn a_handshake_that_never_completes_is_a_retryable_timeout_on_the_session_clock() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::hanging());
    let tts = tts(ws.clone()).with_clock(clock.clone());
    let mut source = tts.synthesize("hi").await.unwrap();

    lk_agents_inference::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    let err = source.next().await.expect_err("the handshake never completed");
    assert_eq!(err.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    assert!(err.retryable());
    assert!(err.message().contains("connection timed out"), "{}", err.message());
    assert!(ws.sent().is_empty(), "nothing was sent on a socket that never opened");
}

// Catches: bounding the handshake but not the first reply after text; later pauses stay legal.
#[tokio::test]
async fn a_gateway_that_takes_the_text_and_never_answers_is_a_retryable_timeout() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone()).with_clock(clock.clone());
    let mut source = tts.synthesize("hi").await.unwrap();

    lk_agents_inference::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
    let err = source.next().await.expect_err("the gateway never answered");
    assert_eq!(err.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    assert!(err.retryable());
    let sent = ws.sent_json();
    assert_eq!(sent[0]["type"], json!("session.create"), "the socket did open");
    assert_eq!(sent[1]["type"], json!("input_transcript"), "and the text went out");
}

fn pcm(n: usize) -> String {
    base64::engine::general_purpose::STANDARD.encode(vec![1u8; n])
}

// Catches: `synthesize()` growing a second, one-shot implementation.
#[tokio::test]
async fn synthesize_is_the_duplex_path_with_one_push_and_an_end() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "session.created", "session_id": "sess_1"})),
        text(json!({"type": "output_audio", "audio": pcm(4)})),
        text(json!({"type": "done"})),
    ]));
    let tts = tts(ws.clone());
    let mut source = tts.synthesize("Hello there.").await.unwrap();
    let first = source.next().await.unwrap().unwrap();
    assert_eq!(&*first.frame.data, &[1u8; 4]);
    assert_eq!(first.segment_id, "sess_1");
    assert!(!first.is_final);
    let last = source.next().await.unwrap().unwrap();
    assert!(last.is_final);
    assert!(source.next().await.unwrap().is_none());

    let sent = ws.sent_json();
    assert_eq!(sent[0]["type"], json!("session.create"));
    assert_eq!(sent[1]["type"], json!("input_transcript"));
    assert_eq!(sent[1]["transcript"], json!("Hello there. "));
    assert_eq!(sent[2]["type"], json!("session.flush"));
}

// Catches: a socket opened, configured and closed per synthesis, which put the gateway's
// handshake and `session.create` before every reply's first audio. The reference's
// `ConnectionPool` hands a finished synthesis's socket to the next one.
#[tokio::test]
async fn a_second_synthesis_reuses_the_socket_the_first_finished_on() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone());
    for _ in 0..2 {
        ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
        ws.feed(text(json!({"type": "done"})));
        let mut source = tts.synthesize("Hello there.").await.unwrap();
        while source.next().await.unwrap().is_some() {}
        for _ in 0..16 {
            tokio::task::yield_now().await;
        }
    }
    assert_eq!(ws.connect_count(), 1, "one socket");
    let creates = ws.sent_json().iter().filter(|f| f["type"] == json!("session.create")).count();
    assert_eq!(creates, 1, "configured once: {:?}", ws.sent_json());
}

// Catches: frames without `USERDATA_TTS_STARTED_TIME`, so the session measured TTFB from the
// first token it pushed, sentence buffering included, and had no time to first sentence.
// `_mark_started()` stamps the first sentence sent.
#[tokio::test]
async fn frames_carry_the_instant_the_first_sentence_went_out() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone()).with_clock(clock.clone());
    let (sink, mut source) = tts.stream().await.unwrap();
    sink.push("Hello there, this is".to_owned()).unwrap();
    clock.advance(Duration::from_millis(300));
    let sent_at = clock.wall().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    sink.push(" the first sentence. And".to_owned()).unwrap();
    until_sent(&ws, 2).await;
    ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
    let frame = source.next().await.unwrap().unwrap();
    let started = frame.started_time.expect("the frame carries the stamp");
    assert!((started - sent_at).abs() < 1e-3, "{started} != {sent_at}");
}

/// Yield until `ws` has recorded `n` frames (or give up), for send-side ordering tests.
async fn until_sent(ws: &FakeWs, n: usize) -> Vec<serde_json::Value> {
    for _ in 0..256 {
        if ws.sent().len() >= n {
            break;
        }
        tokio::task::yield_now().await;
    }
    ws.sent_json()
}

// Catches: holding sentences until `end_input` (or a flush), which makes time-to-first-audio
// carry the whole LLM generation. `_sentence_stream_task` sends each sentence as it arrives.
#[tokio::test]
async fn the_first_sentence_goes_out_before_end_input() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone());
    let (sink, _source) = tts.stream().await.unwrap();
    for token in ["Hello there, this is", " the first sentence.", " And here comes", " the second"]
    {
        sink.push(token.to_owned()).unwrap();
    }

    let sent = until_sent(&ws, 2).await;
    assert_eq!(sent.len(), 2, "session.create, then the sentence, with the input open: {sent:?}");
    assert_eq!(sent[0]["type"], json!("session.create"));
    assert_eq!(sent[1]["type"], json!("input_transcript"));
    assert_eq!(sent[1]["transcript"], json!("Hello there, this is the first sentence. "));

    // A flush sends what the tokenizer holds; the wire's own flush waits for the end.
    sink.flush().unwrap();
    let sent = until_sent(&ws, 3).await;
    assert_eq!(sent.len(), 3, "{sent:?}");
    assert_eq!(sent[2]["transcript"], json!("And here comes the second "));

    sink.end_input();
    let sent = until_sent(&ws, 4).await;
    assert_eq!(sent.len(), 4, "{sent:?}");
    assert_eq!(sent[3], json!({"type": "session.flush"}), "one flush, at the end of input");
}

// Catches: the trailing space added twice, or short sentences sent alone instead of batched
// up to the tokenizer's floor.
#[tokio::test]
async fn sentences_are_batched_and_carry_exactly_one_trailing_space() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone());
    let (sink, _source) = tts.stream().await.unwrap();
    for sentence in ["One. ", "Two! ", "Three? "] {
        sink.push(sentence.to_owned()).unwrap();
    }
    sink.end_input();

    let sent = until_sent(&ws, 3).await;
    let transcripts: Vec<&str> = sent
        .iter()
        .filter(|v| v["type"] == json!("input_transcript"))
        .map(|v| v["transcript"].as_str().unwrap())
        .collect();
    assert_eq!(transcripts, ["One. Two! Three? "]);
}

// Catches: an expressive synthesis sending standard markup instead of the provider's, or its
// bracket cues left in the alignment the transcript is synchronised against.
#[tokio::test]
async fn an_expressive_synthesis_converts_markup_and_drops_cues_from_the_alignment() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "session.created", "session_id": "sess_1"})),
        text(json!({"type": "output_alignment", "words": [
            {"word": "[laughs]", "start": 0.0, "end": 0.2},
            {"word": "hi", "start": 0.2, "end": 0.4},
        ]})),
        text(json!({"type": "output_audio", "audio": pcm(4)})),
        text(json!({"type": "done"})),
    ]));
    let mut opts = opts();
    opts.model = "inworld/inworld-tts-2".into();
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let tts = NativeTts::new(opts, ws.clone(), spawner, super::speech());
    tts.set_expressive(true);
    let mut source = tts
        .synthesize("say <expr type=\"prosody\" label=\"whisper\">this</expr> now")
        .await
        .unwrap();
    let mut words = Vec::new();
    while let Some(audio) = source.next().await.unwrap() {
        words.extend(audio.timed_transcript.into_iter().map(|t| t.text));
    }
    assert!(words.iter().all(|w| !w.contains('[')), "{words:?}");
    let transcript = ws.sent_json()[1]["transcript"].as_str().unwrap().to_owned();
    assert_eq!(transcript, "say [whisper]this now ", "converted to the provider's markup");
}

// Catches: the host's identification headers (the job's `X-LiveKit-*` identity) not reaching
// the socket, or the request id not derived from the host's session id.
#[tokio::test]
async fn a_stream_opened_with_identification_connects_with_those_headers() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let client = tts(ws.clone());
    let ident: Headers = [
        ("User-Agent", "LiveKit Agents/1 (python 3)"),
        ("X-LiveKit-Room-ID", "RM_1"),
        ("X-LiveKit-Job-ID", "AJ_1"),
        ("X-LiveKit-Session-ID", "inference_abc"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v.to_owned()))
    .collect();
    let (_sink, _source) = client.stream_identified(Some(ident)).unwrap();
    until_sent(&ws, 1).await;
    let (_, headers) = ws.last_connect().expect("connected");
    let get = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    assert_eq!(get("X-LiveKit-Room-ID"), Some("RM_1"));
    assert_eq!(get("X-LiveKit-Job-ID"), Some("AJ_1"));
    assert_eq!(get("X-LiveKit-Session-ID"), Some("inference_abc"));
    assert_eq!(get("User-Agent"), Some("LiveKit Agents/1 (python 3)"), "the host's, not ours");
    assert!(get("Authorization").is_some_and(|v| v.starts_with("Bearer ")));

    // Without identification (no job): no `X-LiveKit-*` header at all.
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let (_sink, _source) = tts(ws.clone()).stream().await.unwrap();
    until_sent(&ws, 1).await;
    let (_, headers) = ws.last_connect().expect("connected");
    let livekit: Vec<_> = headers.iter().filter(|(k, _)| k.starts_with("X-LiveKit-")).collect();
    assert!(livekit.is_empty(), "{livekit:?}");
}

// Catches: never sending `session.flush` for an empty synthesis.
#[tokio::test]
async fn an_empty_synthesis_still_flushes_and_terminates() {
    let ws = Arc::new(FakeWs::new(vec![text(json!({"type": "done"}))]));
    let tts = tts(ws.clone());
    let mut source = tts.synthesize("").await.unwrap();
    let last = source.next().await.unwrap().unwrap();
    assert!(last.is_final);
    let sent = ws.sent_json();
    assert!(sent.iter().any(|v| v["type"] == json!("session.flush")));
    assert!(!sent.iter().any(|v| v["type"] == json!("input_transcript")), "nothing to synthesise");
}

// Catches: dropping alignment that arrived before its audio.
#[tokio::test]
async fn alignment_rides_out_on_the_next_audio_frame() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "output_alignment", "session_id": "s",
                        "words": [{"word": "hi", "start": 0.0, "end": 0.2}]})),
        text(json!({"type": "output_audio", "audio": pcm(2)})),
        text(json!({"type": "done"})),
    ]));
    let tts = tts(ws);
    let mut source = tts.synthesize("hi").await.unwrap();
    let frame = source.next().await.unwrap().unwrap();
    assert_eq!(frame.timed_transcript.len(), 1);
    assert_eq!(frame.timed_transcript[0].text, "hi ", "cartesia re-adds the space");
    let last = source.next().await.unwrap().unwrap();
    assert!(last.timed_transcript.is_empty(), "already delivered");
}

// Catches: Cartesia's word timestamps losing the separator it strips (the synchronized transcript
// read "helloworld."), their times, or request ids not derived from the connection's session.
#[tokio::test]
async fn cartesia_words_keep_their_separators_through_a_stream() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "session.created", "session_id": "session-1"})),
        text(json!({"type": "output_audio", "audio": pcm(4800)})),
        text(json!({"type": "output_alignment", "words": [
            {"word": "hello", "start": 0.0, "end": 0.2},
            {"word": "world.", "start": 0.2, "end": 0.5},
        ]})),
        text(json!({"type": "done"})),
    ]));
    let opts = TtsOptions { extra_kwargs: map(&[("add_timestamps", json!(true))]), ..opts() };
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    let client = NativeTts::new(opts, ws, spawner, super::speech());
    let ident: Headers = [("X-LiveKit-Session-ID".to_owned(), "inference_connection".to_owned())]
        .into_iter()
        .collect();
    let (sink, mut source) = client.stream_identified(Some(ident)).unwrap();
    sink.push("hello world.".to_owned()).unwrap();
    sink.end_input();
    let mut words = Vec::new();
    while let Some(audio) = source.next().await.unwrap() {
        assert!(audio.request_id.starts_with("inference_connection_tts_"), "{}", audio.request_id);
        words
            .extend(audio.timed_transcript.into_iter().map(|t| (t.text, t.start_time, t.end_time)));
    }
    assert_eq!(
        words,
        [("hello ".to_owned(), Some(0.0), Some(0.2)), ("world. ".to_owned(), Some(0.2), Some(0.5))]
    );
}

// Catches: reporting a gateway `error` frame as a clean end of stream.
#[tokio::test]
async fn a_gateway_error_frame_fails_the_stream() {
    let ws =
        Arc::new(FakeWs::new(vec![text(json!({"type": "error", "message": "voice not found"}))]));
    let tts = tts(ws);
    let mut source = tts.synthesize("hi").await.unwrap();
    let err = source.next().await.expect_err("the gateway said error");
    assert!(err.message().contains("voice not found"), "{}", err.message());
}

// Catches: `update_options(extra_kwargs=...)` replacing rather than merging.
#[tokio::test]
async fn tts_update_options_merges_the_extra_kwargs() {
    let tts = tts(Arc::new(FakeWs::new(vec![])));
    tts.update_options(None, None, None, Some(&map(&[("add_timestamps", json!(true))]).into()));
    assert!(tts.aligned_transcript());
    tts.update_options(None, Some("bob"), None, Some(&map(&[("speed", json!(1.2))]).into()));
    assert!(tts.aligned_transcript(), "the earlier flag survived the merge");
}

#[tokio::test]
async fn partial_audio_then_provider_failure_preserves_audio_and_fails() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "output_audio", "session_id": "segment", "audio": pcm(4)})),
        text(json!({"type": "error", "message": "synthesis failed"})),
    ]));
    let mut source = tts(ws).synthesize("hi").await.unwrap();
    assert_eq!(&*source.next().await.unwrap().unwrap().frame.data, &[1; 4]);
    let error = source.next().await.expect_err("partial audio is not successful completion");
    assert!(error.message().contains("synthesis failed"));
    assert!(source.next().await.unwrap().is_none());
}

#[tokio::test]
async fn close_settles_a_pump_while_its_audio_queue_is_full() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let client = tts(ws.clone()).with_clock(clock.clone());
    let (_sink, source) = client.stream().await.unwrap();
    for _ in 0..AUDIO_QUEUE_CAPACITY + 2 {
        ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
    }
    until_sent(&ws, 1).await;
    for _ in 0..32 {
        tokio::task::yield_now().await;
    }
    assert_eq!(
        Box::new(source).close(clock.now().saturating_add(Duration::from_secs(1))).await,
        lk_agents_abi::Closed::Clean
    );
}

#[tokio::test]
async fn identified_views_keep_their_expressive_configuration() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let mut options = opts();
    options.model = "inworld/inworld-tts-2".into();
    let client = NativeTts::new(
        options,
        ws.clone(),
        Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())),
        super::speech(),
    );
    let expressive = client.identified(Headers::default(), None);
    let plain = client.identified(Headers::default(), None);
    expressive.set_expressive(true);
    plain.set_expressive(false);
    for (view, drops_cues) in [(expressive, true), (plain, false)] {
        ws.feed(text(
            json!({"type": "output_alignment", "words": [{"word": "[laughs]"}, {"word": "hello"}]}),
        ));
        ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
        ws.feed(text(json!({"type": "done"})));
        let mut source = view.synthesize("hello").await.unwrap();
        let frame = source.next().await.unwrap().unwrap();
        assert_eq!(
            frame.timed_transcript.iter().all(|token| !token.text.contains('[')),
            drops_cues
        );
        while source.next().await.unwrap().is_some() {}
    }
}

#[tokio::test]
async fn a_call_policy_replaces_native_tts_defaults_before_connecting() {
    let ws = Arc::new(FakeWs::hanging());
    let client: Arc<dyn Tts> = Arc::new(tts(ws.clone()));
    let conn = ApiConnectOptions { connect_timeout: Some(0.0), max_retry: 0, ..Default::default() };
    let call = client.with_conn_options(&conn).unwrap_or_else(|| client.clone());
    let mut source = call.synthesize("hello").await.unwrap();
    let failure = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_millis(250),
        source.next(),
    )
    .await
    .expect("the explicit zero connect budget must take effect")
    .unwrap_err();
    assert_eq!(failure.api_kind(), Some(lk_agents_abi::errors::ApiErrorKind::Timeout));
    source.close(lk_agents_abi::Ts::from_nanos(u64::MAX)).await;
}

#[tokio::test]
async fn input_capacity_failure_at_eof_is_not_retryable() {
    let ws = Arc::new(FakeWs::new(Vec::new()).keep_open());
    let client = tts(ws);
    let (sink, mut source) = client.stream().await.unwrap();
    for _ in 0..TEXT_QUEUE_CAPACITY - 1 {
        sink.push("hello".to_owned()).unwrap();
        sink.flush().unwrap();
    }
    sink.push("tail".to_owned()).unwrap();
    sink.end_input();
    let error = lk_agents_abi::timeout(
        &lk_agents_abi::RealClock::new(),
        Duration::from_secs(1),
        source.next(),
    )
    .await
    .expect("EOF capacity refusal must settle the synthesis")
    .expect_err("the final buffered text could not be enqueued");
    assert!(error.message().contains("input queue overflowed at EOF"));
    assert!(
        !error.retryable(),
        "deterministic local capacity failure must not reopen the provider"
    );
}
