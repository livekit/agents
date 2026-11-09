use std::{
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use base64::Engine;
use lk_agents_abi::Tts;
use lk_agents_inference::tts::{GatewayTts, TtsOptions};
use lk_agents_net::clock::SimClock;
use lk_agents_net::task::TokioSpawner;
use lk_agents_net::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
use parking_lot::Mutex;
use tokio::sync::mpsc;

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
        _headers: lk_agents_inference::headers::Headers,
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
        _headers: lk_agents_inference::headers::Headers,
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
