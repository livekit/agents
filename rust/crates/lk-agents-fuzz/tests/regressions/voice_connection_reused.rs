//! A call's replies pay the gateway's connection setup once: each reply's first audio comes within
//! the voice's own time to first audio and a budget. A latency contract on virtual time: the
//! production LiveKit Inference TTS client over a gateway whose connection setup (handshake, TLS,
//! upgrade) costs what a real one does and whose time to first audio is drawn from the corpus
//! (`tts/cartesia/sonic-3/first_audio`).
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicU32, Ordering},
    },
    time::Duration,
};

use base64::Engine;
use lk_agents_abi::{SimClock, TokioSpawner, Tts};
use lk_agents_fuzz::{corpus, drive::simulate};
use lk_agents_inference::tts::{NativeTts, TtsOptions};
use lk_agents_net::transport::{TransportError, WsMessage, WsReceiver, WsSender, WsTransport};
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
        simulate(async move {
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
            let voice = NativeTts::<lk_agents_voice::tts::processing::SpeechText>::new(
                opts,
                Arc::new(gateway),
                Arc::new(TokioSpawner::current()),
            )
            .with_clock(Arc::new(SimClock::new()));
            // The session prewarms its voice as it starts; the greeting comes a moment later.
            voice.prewarm();
            tokio::time::sleep(Duration::from_millis(500)).await;
            for reply in 0..5 {
                let asked = tokio::time::Instant::now();
                let mut audio = voice.synthesize("Sure, let me check that for you.").await.unwrap();
                audio.next().await.unwrap().expect("the reply's first audio");
                let took = asked.elapsed();
                let owed = *shared.first_audio.lock().unwrap();
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
                let path = ["tts", "cartesia/sonic-3", "first_audio"];
                let delay =
                    Duration::from_millis(corpus::draw_ms(&mut shared.rng.lock().unwrap(), &path));
                *shared.first_audio.lock().unwrap() = delay;
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
        _headers: Vec<(String, String)>,
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
        let tx = self.0.lock().unwrap();
        tx.as_ref().and_then(|tx| tx.send(text).ok()).ok_or(TransportError::Closed)
    }

    async fn send_binary(&self, _data: Vec<u8>) -> Result<(), TransportError> {
        Ok(())
    }

    fn is_closed(&self) -> bool {
        self.0.lock().unwrap().is_none()
    }

    async fn close(&self) {
        self.0.lock().unwrap().take();
    }
}

struct Receiver(mpsc::UnboundedReceiver<WsMessage>);

#[async_trait::async_trait]
impl WsReceiver for Receiver {
    async fn recv(&mut self) -> Result<WsMessage, TransportError> {
        Ok(self.0.recv().await.unwrap_or(WsMessage::Closed))
    }
}
