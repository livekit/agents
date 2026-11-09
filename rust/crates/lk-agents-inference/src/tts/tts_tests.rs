use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use base64::Engine;
use lk_agents_abi::{ApiConnectOptions, Clock, StreamInput as _, StreamOutput as _, Tts};
use lk_agents_net::clock::SimClock;
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use parking_lot::Mutex;
use serde_json::json;
use tokio::sync::mpsc;

use super::{AUDIO_QUEUE_CAPACITY, GatewayTts, TEXT_QUEUE_CAPACITY, TtsOptions};
use crate::headers::Headers;
use crate::testkit::{FakeWs, text};

/// What opening a gateway connection costs before its first frame. Measured on calls: a reply
/// that reconnected took 0.57 s to its first audio, one on a pooled socket 0.36 s.
const CONNECT: Duration = Duration::from_millis(200);
/// What the client may add to the voice's own time to first audio.
const BUDGET: Duration = Duration::from_millis(50);

// Catches: a socket opened, configured and closed per synthesis (54c213e74), which put the
// gateway's connection setup before every reply's first audio; and a prewarm that opens nothing,
// which puts it before the first.
#[test]
fn each_reply_pays_only_the_voice() {
    for seed in 0..4 {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()
            .expect("a paused runtime");
        runtime.block_on(async move {
            let gateway = Gateway(Arc::new(Shared {
                rng: Mutex::new(fastrand::Rng::with_seed(seed)),
                connects: AtomicU32::new(0),
                first_audio: Mutex::new(Duration::ZERO),
            }));
            let shared = gateway.0.clone();
            let opts = TtsOptions {
                model: "cartesia/sonic-3".into(),
                base_url: "https://gateway.fake/v1".into(),
                api_key: "k".into(),
                api_secret: "s".into(),
                ..Default::default()
            };
            let voice = GatewayTts::new(
                opts,
                Arc::new(gateway),
                Arc::new(TokioSpawner::current()),
                Arc::new(SimClock::new())
            );
            // The session prewarms its voice as it starts; the greeting comes a moment later.
            voice.prewarm();
            tokio::time::sleep(Duration::from_millis(500)).await;
            for reply in 0..5 {
                let asked = tokio::time::Instant::now();
                let mut audio = voice.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Sure, let me check that for you.").await.unwrap();
                audio.next().await.unwrap().expect("the reply's first audio");
                let took = asked.elapsed();
                let owed = *shared.first_audio.lock();
                assert!(
                    took <= owed + BUDGET,
                    "seed {seed}, reply {reply}: first audio after {took:?}, the voice owed {owed:?} \
                     ({} connections opened)",
                    shared.connects.load(Ordering::SeqCst)
                );
                while audio.next().await.unwrap().is_some() {}
                // The person answers before the next reply.
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
    }
}

// Catches: a synthesis that does not report the connection it acquired (1.8.10's
// `lk.agents.connection.acquire_time`), a reused socket reported as a fresh one, or the report
// repeated on later frames of the same synthesis.
#[test]
fn each_synthesis_reports_the_connection_it_acquired() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .expect("a paused runtime");
    runtime.block_on(async move {
        let gateway = Gateway(Arc::new(Shared {
            rng: Mutex::new(fastrand::Rng::with_seed(0)),
            connects: AtomicU32::new(0),
            first_audio: Mutex::new(Duration::ZERO),
        }));
        let opts = TtsOptions {
            model: "cartesia/sonic-3".into(),
            base_url: "https://gateway.fake/v1".into(),
            api_key: "k".into(),
            api_secret: "s".into(),
            ..Default::default()
        };
        let voice = GatewayTts::new(
            opts,
            Arc::new(gateway),
            Arc::new(TokioSpawner::current()),
            Arc::new(SimClock::new()),
        );
        let mut reported = Vec::new();
        for _ in 0..2 {
            let mut audio = voice
                .synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Sure, let me check that for you.")
                .await
                .unwrap();
            let first = audio.next().await.unwrap().expect("the reply's first audio");
            reported.push(first.connection.expect("the first frame reports the connection"));
            while let Some(frame) = audio.next().await.unwrap() {
                assert_eq!(frame.connection, None, "reported once per synthesis");
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
        assert!(!reported[0].reused && reported[0].acquire_time >= CONNECT.as_secs_f64());
        assert_eq!((reported[1].reused, reported[1].acquire_time), (true, 0.0));
    });
}

#[tokio::test(start_paused = true)]
async fn gateway_error_codes_and_retry_policy_reach_the_public_audio_source_without_provider_text()
{
    for (code, retryable, status) in
        [("BAD_REQUEST", false, 400), ("RATE_LIMITED", false, 429), ("RATE_LIMITED", true, 429)]
    {
        let gateway = Arc::new(FrameGateway {
            connects: AtomicU32::new(0),
            frames: vec![
                serde_json::json!({
                    "type": "error", "code": code,
                    "data": "private provider text containing credentials",
                    "retryable": retryable,
                })
                .to_string(),
            ],
        });
        let voice = GatewayTts::new(
            TtsOptions {
                model: "cartesia/sonic-3".into(),
                base_url: "https://gateway.fake/v1".into(),
                api_key: "k".into(),
                api_secret: "s".into(),
                ..Default::default()
            },
            gateway.clone(),
            Arc::new(TokioSpawner::current()),
            Arc::new(SimClock::new()),
        );
        let mut audio =
            voice.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Hello.").await.unwrap();
        let error = audio.next().await.expect_err("a gateway refusal must reach the caller");
        assert_eq!(error.status_code(), Some(status));
        assert_eq!(error.retryable(), retryable);
        assert_eq!(error.message(), format!("LiveKit Inference TTS returned {code}"));
        assert!(!error.message().contains("private provider text"));
        assert!(!error.message().contains("{\"type\""));
        assert!(audio.next().await.unwrap().is_none());
        assert_eq!(gateway.connects.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn gateway_fallback_notification_keeps_the_replys_audio_playable() {
    let pcm = base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4800]);
    let gateway = Arc::new(FrameGateway {
        connects: AtomicU32::new(0),
        frames: vec![
            serde_json::json!({
                "type": "fallback_activated", "session_id": "sess",
                "fallback_type": "system_default", "provider": "inworld",
                "model": "inworld-tts-2", "voice": "ashley", "cause": "provider_error",
            })
            .to_string(),
            serde_json::json!({"type": "output_audio", "session_id": "sess", "audio": pcm})
                .to_string(),
            serde_json::json!({"type": "done", "session_id": "sess"}).to_string(),
        ],
    });
    let voice = GatewayTts::new(
        TtsOptions {
            model: "cartesia/sonic-3".into(),
            base_url: "https://gateway.fake/v1".into(),
            api_key: "k".into(),
            api_secret: "s".into(),
            ..Default::default()
        },
        gateway.clone(),
        Arc::new(TokioSpawner::current()),
        Arc::new(SimClock::new()),
    );
    let mut audio = voice.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Hello.").await.unwrap();
    let frame = audio.next().await.unwrap().expect("the fallback voice remains audible");
    let mut samples = frame.frame.samples_per_channel;
    while let Some(frame) = audio.next().await.unwrap() {
        samples += frame.frame.samples_per_channel;
    }
    assert_eq!(samples, 2400);
    assert_eq!(gateway.connects.load(Ordering::SeqCst), 1);
}

struct FrameGateway {
    connects: AtomicU32,
    frames: Vec<String>,
}

#[async_trait::async_trait]
impl WsTransport for FrameGateway {
    async fn connect(
        &self,
        _url: &str,
        _headers: Headers,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        self.connects.fetch_add(1, Ordering::SeqCst);
        let (to_server, mut from_client) = mpsc::unbounded_channel::<String>();
        let (to_client, from_server) = mpsc::unbounded_channel();
        let frames = self.frames.clone();
        tokio::spawn(async move {
            let mut frames = Some(frames);
            while let Some(frame) = from_client.recv().await {
                let frame: serde_json::Value = serde_json::from_str(&frame).unwrap();
                match frame["type"].as_str() {
                    Some("session.create") => {
                        assert_eq!(frame["sample_rate"], 24_000);
                        let _ = to_client.send(WsMessage::Text(
                            r#"{"type":"session.created","session_id":"sess"}"#.into(),
                        ));
                    }
                    Some("session.flush") => {
                        for frame in frames.take().unwrap_or_default() {
                            let _ = to_client.send(WsMessage::Text(frame));
                        }
                    }
                    _ => {}
                }
            }
        });
        Ok((Arc::new(Sender(Mutex::new(Some(to_server)))), Box::new(Receiver(from_server))))
    }
}

struct Shared {
    rng: Mutex<fastrand::Rng>,
    connects: AtomicU32,
    /// The time to first audio drawn for the latest synthesis.
    first_audio: Mutex<Duration>,
}

/// The gateway: each connection a server that answers a synthesis's first text with audio after
/// the drawn time to first audio, and its flush with `done` once that audio went out.
struct Gateway(Arc<Shared>);

async fn serve(
    shared: Arc<Shared>,
    mut rx: mpsc::UnboundedReceiver<String>,
    tx: mpsc::UnboundedSender<WsMessage>,
) {
    let mut audio: Option<tokio::sync::oneshot::Receiver<()>> = None;
    let mut answering = false;
    while let Some(frame) = rx.recv().await {
        let frame: serde_json::Value = serde_json::from_str(&frame).unwrap_or_default();
        match frame["type"].as_str() {
            Some("session.create") => {
                let created = serde_json::json!({"type": "session.created", "session_id": "sess"});
                let _ = tx.send(WsMessage::Text(created.to_string()));
            }
            Some("input_transcript") if !answering => {
                answering = true;
                let delay = Duration::from_millis(shared.rng.lock().u64(190..=250));
                *shared.first_audio.lock() = delay;
                let (sent, sent_rx) = tokio::sync::oneshot::channel();
                audio = Some(sent_rx);
                let tx = tx.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let pcm = base64::engine::general_purpose::STANDARD.encode(vec![0u8; 4800]);
                    let frame = serde_json::json!({"type": "output_audio", "audio": pcm});
                    let _ = tx.send(WsMessage::Text(frame.to_string()));
                    let _ = sent.send(());
                });
            }
            Some("session.flush") => {
                if let Some(sent) = audio.take() {
                    let _ = sent.await;
                }
                answering = false;
                let _ = tx.send(WsMessage::Text(serde_json::json!({"type": "done"}).to_string()));
            }
            _ => {}
        }
    }
    let _ = tx.send(WsMessage::Closed);
}

#[async_trait::async_trait]
impl WsTransport for Gateway {
    async fn connect(
        &self,
        _url: &str,
        _headers: Headers,
    ) -> Result<(Arc<dyn WsSender>, Box<dyn WsReceiver>), TransportError> {
        tokio::time::sleep(CONNECT).await;
        self.0.connects.fetch_add(1, Ordering::SeqCst);
        let (to_server, from_client) = mpsc::unbounded_channel();
        let (to_client, from_server) = mpsc::unbounded_channel();
        tokio::spawn(serve(self.0.clone(), from_client, to_client));
        Ok((Arc::new(Sender(Mutex::new(Some(to_server)))), Box::new(Receiver(from_server))))
    }
}

struct Sender(Mutex<Option<mpsc::UnboundedSender<String>>>);

#[async_trait::async_trait]
impl WsSender for Sender {
    async fn send_text(&self, text: String) -> Result<(), TransportError> {
        let tx = self.0.lock();
        tx.as_ref().and_then(|tx| tx.send(text).ok()).ok_or(TransportError::Closed)
    }

    async fn send_binary(&self, _data: Vec<u8>) -> Result<(), TransportError> {
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.0.lock().is_none()
    }

    async fn close(&self) {
        self.0.lock().take();
    }
}

struct Receiver(mpsc::UnboundedReceiver<WsMessage>);

#[async_trait::async_trait]
impl WsReceiver for Receiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        Ok(self.0.recv().await.unwrap_or(WsMessage::Closed))
    }
}

fn opts() -> TtsOptions {
    TtsOptions {
        model: "cartesia/sonic-3".into(),
        base_url: "https://gw/v1".into(),
        api_key: "k".into(),
        api_secret: "s".into(),
        ..Default::default()
    }
}

fn tts(ws: Arc<FakeWs>) -> GatewayTts {
    tts_on(ws, Arc::new(lk_agents_net::clock::RealClock::new()))
}

fn tts_on(ws: Arc<FakeWs>, clock: Arc<dyn lk_agents_abi::Clock>) -> GatewayTts {
    let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
    GatewayTts::new(opts(), ws, spawner, clock)
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
        let mut source =
            tts(ws).synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();
        assert!(source.next().await.is_err(), "malformed audio cannot become successful output");
    }
}

#[tokio::test]
async fn done_before_the_final_flush_cannot_finish_a_synthesis() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let (sink, mut source) =
        tts(ws.clone()).stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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
    let tts = tts_on(ws.clone(), clock.clone());
    let mut source = tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();

    crate::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
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
    let tts = tts_on(ws.clone(), clock.clone());
    let mut source = tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();

    crate::testkit::arm_then_advance(&clock, Duration::from_secs(10)).await;
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
    let mut source =
        tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Hello there.").await.unwrap();
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
// handshake and `session.create` before every reply's first audio.
#[tokio::test]
async fn a_second_synthesis_reuses_the_socket_the_first_finished_on() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone());
    for _ in 0..2 {
        ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
        ws.feed(text(json!({"type": "done"})));
        let mut source =
            tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Hello there.").await.unwrap();
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
#[tokio::test]
async fn frames_carry_the_instant_the_first_sentence_went_out() {
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts_on(ws.clone(), clock.clone());
    let (sink, mut source) =
        tts.stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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
// carry the whole LLM generation.
#[tokio::test]
async fn the_first_sentence_goes_out_before_end_input() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let tts = tts(ws.clone());
    let (sink, _source) =
        tts.stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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
    let (sink, _source) =
        tts.stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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
    let tts = GatewayTts::new(
        opts,
        ws.clone(),
        spawner,
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );
    let (sink, mut source) =
        Tts::stream(&tts, &lk_agents_abi::telemetry::Cx::NONE, true).await.unwrap().split();
    sink.push("say <expr type=\"prosody\" label=\"whisper\">this</expr> now".into()).unwrap();
    sink.end_input();
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
    let (_sink, _source) = client.stream_identified(Some(ident), false).unwrap();
    until_sent(&ws, 1).await;
    let (_, headers) = ws.last_connect().expect("connected");
    let get = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str());
    assert_eq!(get("X-LiveKit-Room-ID"), Some("RM_1"));
    assert_eq!(get("X-LiveKit-Job-ID"), Some("AJ_1"));
    assert_eq!(get("X-LiveKit-Session-ID"), None);
    assert_eq!(get("User-Agent"), Some("LiveKit Agents/1 (python 3)"), "the host's, not ours");
    assert!(get("Authorization").is_some_and(|v| v.starts_with("Bearer ")));

    // Without identification (no job): no `X-LiveKit-*` header at all.
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let (_sink, _source) =
        tts(ws.clone()).stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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
    let mut source = tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "").await.unwrap();
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
    let mut source = tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();
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
    let client =
        GatewayTts::new(opts, ws, spawner, Arc::new(lk_agents_net::clock::RealClock::new()));
    let ident: Headers = [("X-LiveKit-Session-ID".to_owned(), "inference_connection".to_owned())]
        .into_iter()
        .collect();
    let (sink, mut source) = client.stream_identified(Some(ident), false).unwrap();
    sink.push("hello world.".to_owned()).unwrap();
    sink.end_input();
    let mut words = Vec::new();
    while let Some(audio) = source.next().await.unwrap() {
        assert!(
            audio.request_id.starts_with("inference_") && audio.request_id.contains("_tts_"),
            "{}",
            audio.request_id
        );
        words
            .extend(audio.timed_transcript.into_iter().map(|t| (t.text, t.start_time, t.end_time)));
    }
    assert_eq!(
        words,
        [("hello ".to_owned(), Some(0.0), Some(0.2)), ("world. ".to_owned(), Some(0.2), Some(0.5))]
    );
}

// Catches: reporting a gateway `error` frame as a clean end of stream, or putting the frame's raw
// provider text in place of its code.
#[tokio::test]
async fn a_gateway_error_frame_fails_the_stream() {
    let ws = Arc::new(FakeWs::new(vec![text(
        json!({"type": "error", "code": "NOT_FOUND", "data": "voice not found"}),
    )]));
    let tts = tts(ws);
    let mut source = tts.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();
    let err = source.next().await.expect_err("the gateway said error");
    assert!(err.message().contains("NOT_FOUND"), "{}", err.message());
    assert!(!err.message().contains("voice not found"), "{}", err.message());
}

// Catches: `update_options(extra_kwargs=...)` replacing rather than merging.
#[tokio::test]
async fn tts_update_options_merges_the_extra_kwargs() {
    let tts = tts(Arc::new(FakeWs::new(vec![])));
    tts.update_options(None, None, None, Some(&map(&[("add_timestamps", json!(true))])));
    assert!(tts.aligned_transcript());
    tts.update_options(None, Some("bob"), None, Some(&map(&[("speed", json!(1.2))])));
    assert!(tts.aligned_transcript(), "the earlier flag survived the merge");
}

#[tokio::test]
async fn partial_audio_then_provider_failure_preserves_audio_and_fails() {
    let ws = Arc::new(FakeWs::new(vec![
        text(json!({"type": "output_audio", "session_id": "segment", "audio": pcm(4)})),
        text(json!({"type": "error", "code": "SERVER_ERROR", "data": "synthesis failed"})),
    ]));
    let mut source = tts(ws).synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hi").await.unwrap();
    assert_eq!(&*source.next().await.unwrap().unwrap().frame.data, &[1; 4]);
    let error = source.next().await.expect_err("partial audio is not successful completion");
    assert!(error.message().contains("SERVER_ERROR"), "{}", error.message());
    assert!(source.next().await.unwrap().is_none());
}

#[tokio::test]
async fn close_settles_a_pump_while_its_audio_queue_is_full() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let client = tts_on(ws.clone(), clock.clone());
    let (_sink, source) =
        client.stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
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

// Catches: expressive leaking from one synthesis to another on the same client (a shared flag).
#[tokio::test]
async fn each_synthesis_carries_its_own_expressive_option() {
    let ws = Arc::new(FakeWs::new(vec![]).keep_open());
    let mut options = opts();
    options.model = "inworld/inworld-tts-2".into();
    let client = GatewayTts::new(
        options,
        ws.clone(),
        Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())),
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );
    for expressive in [true, false] {
        ws.feed(text(
            json!({"type": "output_alignment", "words": [{"word": "[laughs]"}, {"word": "hello"}]}),
        ));
        ws.feed(text(json!({"type": "output_audio", "audio": pcm(4)})));
        ws.feed(text(json!({"type": "done"})));
        let (sink, mut source) =
            Tts::stream(&client, &lk_agents_abi::telemetry::Cx::NONE, expressive)
                .await
                .unwrap()
                .split();
        sink.push("hello".into()).unwrap();
        sink.end_input();
        let frame = source.next().await.unwrap().unwrap();
        assert_eq!(
            frame.timed_transcript.iter().all(|token| !token.text.contains('[')),
            expressive
        );
        while source.next().await.unwrap().is_some() {}
    }
}

#[tokio::test]
async fn a_call_policy_replaces_gateway_tts_defaults_before_connecting() {
    let ws = Arc::new(FakeWs::hanging());
    let client: Arc<dyn Tts> = Arc::new(tts(ws.clone()));
    let conn = ApiConnectOptions { connect_timeout: Some(0.0), max_retry: 0, ..Default::default() };
    let call = client.with_conn_options(&conn).unwrap_or_else(|| client.clone());
    let mut source = call.synthesize(&lk_agents_abi::telemetry::Cx::NONE, "hello").await.unwrap();
    let failure = lk_agents_abi::timeout(
        &lk_agents_net::clock::RealClock::new(),
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
    let (sink, mut source) =
        client.stream(&lk_agents_abi::telemetry::Cx::NONE, false).await.unwrap().split();
    for _ in 0..TEXT_QUEUE_CAPACITY - 1 {
        sink.push("hello".to_owned()).unwrap();
        sink.flush().unwrap();
    }
    sink.push("tail".to_owned()).unwrap();
    sink.end_input();
    let error = lk_agents_abi::timeout(
        &lk_agents_net::clock::RealClock::new(),
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

fn credentials() -> Option<(String, String, String)> {
    // The inference key pair, distinct from the room's; resolved so the fallback is exercised too.
    let key = crate::credentials::resolve_api_key(
        None,
        std::env::var("LIVEKIT_INFERENCE_API_KEY").ok().as_deref(),
        std::env::var("LIVEKIT_API_KEY").ok().as_deref(),
    )
    .ok()?;
    let secret = crate::credentials::resolve_api_secret(
        None,
        std::env::var("LIVEKIT_INFERENCE_API_SECRET").ok().as_deref(),
        std::env::var("LIVEKIT_API_SECRET").ok().as_deref(),
    )
    .ok()?;
    let url = crate::headers::default_inference_url(
        std::env::var("LIVEKIT_INFERENCE_URL").ok().as_deref(),
        std::env::var("LIVEKIT_URL").ok().as_deref(),
    );
    Some((key, secret, url))
}

// Catches: what the WebSocket half assumes and the unit tests fake: the `?model=` query, the
// `Authorization` header surviving the upgrade (several clients drop it), `"type"` first in
// `session.create`, and the `input_transcript`, `output_audio`, `done` sequence.
#[tokio::test]
#[ignore = "needs LIVEKIT_INFERENCE_API_KEY/_SECRET and the network"]
async fn a_tts_synthesis_round_trips_over_the_websocket() {
    use lk_agents_net::websocket::TungsteniteTransport;

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
        Arc::new(lk_agents_net::clock::RealClock::new()),
    );

    let mut source = tts
        .synthesize(&lk_agents_abi::telemetry::Cx::NONE, "Hello from Rust.")
        .await
        .expect("the socket opened");
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
