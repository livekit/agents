//! `inference/tts.py` -- the gateway TTS, WebSocket + JSON (§1.4). `synthesize()` is the duplex
//! path with one push and the input closed, as in the reference, so [`NativeTts::stream`] always
//! returns `Some`. The reference's connection pool is not ported: a socket opens per synthesis.
//! Text arrives normalised, converted and split by the host (`inference/tts.py`).

#[cfg(feature = "python")]
mod python;
pub mod wire;

use crate::abi::{
    StreamSink, StreamSource, SynthesizedAudio, Tts, TtsCapabilities, TtsSink,
    TtsSource,
};
use crate::headers::{HEADER_SESSION_ID, Headers, gateway_headers};
use crate::ids::{RequestKind, create_inference_request_id};
use crate::transport::{
    WsMessage, WsSender, WsTransport, connect_error, next_terminal, open_ws, stream_closed,
    to_ws_url, within,
};
use crate::vad::machine::mono;
use async_trait::async_trait;
use base64::Engine as _;
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{ApiConnectOptions, Clock, TimedString};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;
use wire::FallbackModel;

const WHAT: &str = "LiveKit Inference TTS";

// ---------------------------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TtsOptions {
    pub model: String,
    pub voice: Option<String>,
    pub language: Option<String>,
    pub encoding: String,
    pub sample_rate: u32,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    pub extra_kwargs: serde_json::Map<String, serde_json::Value>,
    pub fallback: Vec<FallbackModel>,
    /// Not an `Option`: `__init__` substitutes the default; its `timeout` also bounds the
    /// handshake and the gateway's first message.
    pub conn_options: ApiConnectOptions,
}

impl Default for TtsOptions {
    fn default() -> Self {
        TtsOptions {
            model: String::new(),
            voice: None,
            language: None,
            encoding: wire::DEFAULT_ENCODING.to_owned(),
            sample_rate: wire::DEFAULT_SAMPLE_RATE,
            base_url: String::new(),
            api_key: String::new(),
            api_secret: String::new(),
            extra_kwargs: Default::default(),
            fallback: Vec::new(),
            conn_options: lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
        }
    }
}

pub struct NativeTts {
    opts: Mutex<TtsOptions>,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    /// Bound into, not rebuilt, so `update_options` state survives a session binding its own.
    clock: Arc<lk_agents_abi::BoundClock>,
}

/// `inference.TTS`'s constructor keywords, in the host's order.
#[lk_agents_abi::host(client(class = crate::tts::InferenceTts, build = crate::tts::python::build, positional = model))]
pub struct TtsClientConfig {
    pub model: String,
    #[host(kwarg(default = None))]
    pub voice: Option<String>,
    #[host(kwarg(default = None))]
    pub language: Option<String>,
    #[host(kwarg(default = "pcm_s16le".to_owned()))]
    pub encoding: String,
    #[host(kwarg(default = 24_000))]
    pub sample_rate: u32,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    #[host(kwarg(default = None))]
    pub extra_kwargs: Option<crate::extra::ExtraKwargs>,
    #[host(kwarg(default = None))]
    pub fallback: Option<crate::extra::Fallbacks<FallbackModel>>,
    #[host(kwarg(default = 10.0))]
    pub timeout: f64,
}

impl NativeTts {
    pub fn new(opts: TtsOptions, ws: Arc<dyn WsTransport>, spawner: Arc<dyn Spawner>) -> Self {
        NativeTts {
            opts: Mutex::new(opts),
            ws,
            spawner,
            clock: lk_agents_abi::BoundClock::new(),
        }
    }

    pub fn with_clock(self, clock: Arc<dyn Clock>) -> Self {
        self.clock.bind(clock);
        self
    }

    pub fn bind_clock(&self, clock: Arc<dyn Clock>) {
        self.clock.bind(clock);
    }

    /// Headers are per stream, not a client field: each synthesis gets its own session id.
    /// `None`: no job, the native headers.
    pub fn stream_identified(
        &self,
        identification: Option<Headers>,
    ) -> Result<(TtsSinkHandle, TtsSourceHandle), ProviderError> {
        self.open(identification.as_ref())
    }
}

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::NativeTts`.
#[lk_agents_abi::host(object(
    name = "InferenceTts",
    cell = "arc",
    rust_name = InferenceTts,
    stub = "def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
    stub = "def bind_clock(self, clock: SessionClock) -> None: ...",
    stub = "def prewarm(self, *, headers: list[tuple[str, str]] | None = None) -> None: ..."
))]
impl NativeTts {
    /// `_resolve_expressive_options` decided this turn is expressive. A no-op here: the host
    /// converts the markup and drops bracket cues from alignment itself (`inference/tts.py`).
    #[host(method)]
    pub fn set_expressive(&self, expressive: bool) {
        let _ = expressive;
    }

    pub fn model(&self) -> String {
        self.opts.lock().unwrap().model.clone()
    }

    #[host(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.opts.lock().unwrap().sample_rate
    }

    /// `TTSCapabilities.aligned_transcript`, recomputed from the model and extra kwargs.
    #[host(getter)]
    pub fn aligned_transcript(&self) -> bool {
        let opts = self.opts.lock().unwrap();
        wire::has_aligned_transcript(&opts.model, &opts.extra_kwargs)
    }

    /// `update_options`. Unlike the LLM's, `extra` merges (`.update()`).
    #[host(method(kw_only))]
    pub fn update_options(
        &self,
        model: Option<&str>,
        voice: Option<&str>,
        language: Option<&str>,
        extra: Option<&crate::extra::ExtraKwargs>,
    ) {
        let mut opts = self.opts.lock().unwrap();
        if let Some(model) = model {
            opts.model = model.to_owned();
        }
        if let Some(voice) = voice {
            opts.voice = Some(voice.to_owned());
        }
        if let Some(language) = language {
            opts.language =
                Some(lk_agents_abi::tokenize::language::LanguageCode::new(language).as_str().to_owned());
        }
        if let Some(extra) = extra {
            for (k, v) in extra.iter() {
                opts.extra_kwargs.insert(k.clone(), v.clone());
            }
        }
    }

    /// Open a duplex synthesis. The gateway has no one-shot endpoint, so `synthesize` is this plus
    /// one push and `end_input`. `headers` is `get_inference_headers()`, read by the host in the
    /// synthesis's job context; each sentence pushed goes out as it arrives.
    #[host(method(name = "stream", stream = lk_agents_abi::InferenceTtsStream, kw_only))]
    pub async fn stream_for_host(&self, headers: Option<Vec<(String, String)>>) -> Result<(TtsSink, TtsSource), ProviderError> {
        let (sink, source) = self.stream_identified(headers.map(|pairs| pairs.into_iter().collect()))?;
        Ok((Box::new(sink), Box::new(source)))
    }

    fn open(
        &self,
        identification: Option<&Headers>,
    ) -> Result<(TtsSinkHandle, TtsSourceHandle), ProviderError> {
        let opts = self.opts.lock().unwrap().clone();
        let provider = opts.model.split('/').next().unwrap_or("").to_owned();

        let (text_tx, text_rx) = mpsc::unbounded_channel::<TextCmd>();
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Result<SynthesizedAudio, ApiError>>();

        let headers =
            gateway_headers(identification, None, None, &opts.api_key, &opts.api_secret)?;
        let connection_session_id = headers.get(HEADER_SESSION_ID).map(str::to_owned);

        let pump = Pump {
            ws: self.ws.clone(),
            url: format!("{}/tts?model={}", to_ws_url(&opts.base_url), opts.model),
            headers: headers.into_vec(),
            timeout: Duration::from_secs_f64(opts.conn_options.timeout.max(0.0)),
            clock: self.clock.clone() as Arc<dyn Clock>,
            create_frame: wire::session_create(&opts),
            sample_rate: opts.sample_rate,
            opts,
            provider,
            // `create_inference_request_id(connection.session_id, "tts")`: no session id yields a
            // bare shortuuid.
            request_id: create_inference_request_id(
                connection_session_id.as_deref(),
                RequestKind::Tts,
                "",
            ),
            text_rx,
            audio_tx,
        };
        self.spawner.spawn_unboxed(pump.run());

        Ok((TtsSinkHandle { text: text_tx }, TtsSourceHandle { audio: audio_rx, done: false }))
    }
}

#[async_trait]
impl Tts for NativeTts {
    fn label(&self) -> String {
        format!("livekit.{}", self.model())
    }

    /// `streaming` is always `True`; `aligned_transcript` reads the live extra kwargs.
    fn capabilities(&self) -> TtsCapabilities {
        TtsCapabilities { streaming: true, aligned_transcript: self.aligned_transcript() }
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }

    fn num_channels(&self) -> u16 {
        1
    }

    /// `synthesize(text)` = `_synthesize_with_stream(text)`: the gateway has no one-shot endpoint.
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        let (sink, source) = self.open(None)?;
        // The host's tokenizer yields no sentence for empty text, so nothing is sent for it.
        if !text.is_empty() {
            sink.push(text.to_owned())?;
        }
        sink.end_input();
        Ok(Box::new(source))
    }

    async fn stream(&self) -> Result<(TtsSink, TtsSource), ProviderError> {
        self.open(None)
            .map(|(sink, source)| (Box::new(sink) as TtsSink, Box::new(source) as TtsSource))
    }
}

// ---------------------------------------------------------------------------------------------
// The two halves
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum TextCmd {
    Push(String),
    Flush,
    End,
}

pub struct TtsSinkHandle {
    text: mpsc::UnboundedSender<TextCmd>,
}

impl StreamSink<String> for TtsSinkHandle {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        self.text.send(TextCmd::Push(item)).map_err(|_| stream_closed(WHAT))
    }

    /// The `_FlushSentinel`. A no-op on the wire: pushes are already whole sentences.
    fn flush(&self) -> Result<(), ProviderError> {
        self.text.send(TextCmd::Flush).map_err(|_| stream_closed(WHAT))
    }

    fn end_input(&self) {
        let _ = self.text.send(TextCmd::End);
    }
}

pub struct TtsSourceHandle {
    audio: mpsc::UnboundedReceiver<Result<SynthesizedAudio, ApiError>>,
    done: bool,
}

#[async_trait]
impl StreamSource<SynthesizedAudio> for TtsSourceHandle {
    async fn next(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        next_terminal(&mut self.audio, &mut self.done).await
    }

    /// Nothing here awaits the network, so the deadline is not consulted.
    async fn close(mut self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        self.done = true;
        self.audio.close();
        lk_agents_abi::Closed::Clean
    }
}

// ---------------------------------------------------------------------------------------------
// The pump
// ---------------------------------------------------------------------------------------------

struct Pump {
    ws: Arc<dyn WsTransport>,
    url: String,
    headers: Vec<(String, String)>,
    /// `conn_options.timeout`: bounds the handshake and the first message once text is sent, not
    /// later reads (between segments the gateway waits on us).
    timeout: Duration,
    clock: Arc<dyn Clock>,
    create_frame: serde_json::Value,
    opts: TtsOptions,
    provider: String,
    request_id: String,
    sample_rate: u32,
    text_rx: mpsc::UnboundedReceiver<TextCmd>,
    audio_tx: mpsc::UnboundedSender<Result<SynthesizedAudio, ApiError>>,
}

impl Pump {
    async fn run(self) {
        let Pump {
            ws,
            url,
            headers,
            timeout,
            clock,
            create_frame,
            opts,
            provider,
            request_id,
            sample_rate,
            text_rx,
            audio_tx,
        } = self;

        let opened = open_ws(ws.as_ref(), &url, headers, &*clock, timeout, WHAT, &create_frame);
        let (sender, mut receiver) = match opened.await {
            Ok(pair) => pair,
            Err(err) => {
                let _ = audio_tx.send(Err(err));
                return;
            }
        };

        // `input_sent_event`: nothing is read back until the session is configured and text (or
        // the flush) is on the wire.
        let (input_sent_tx, input_sent_rx) = tokio::sync::watch::channel(false);

        let send = Self::send_loop(sender.clone(), text_rx, &opts, input_sent_tx);
        let recv = Self::recv_loop(
            receiver.as_mut(),
            &audio_tx,
            RecvCtx {
                provider: &provider,
                request_id,
                sample_rate,
                input_sent: input_sent_rx,
                clock: clock.as_ref(),
                first_message_timeout: timeout,
            },
        );

        let outcome = tokio::select! {
            r = send => r,
            r = recv => r,
        };
        if let Err(err) = outcome {
            let _ = audio_tx.send(Err(err));
        }
        sender.close().await;
    }

    /// `_sentence_stream_task` over host-split sentences: each push is one `input_transcript`,
    /// sent as it arrives; end of input is the one `session.flush`.
    async fn send_loop(
        sender: Arc<dyn WsSender>,
        mut text_rx: mpsc::UnboundedReceiver<TextCmd>,
        opts: &TtsOptions,
        input_sent: tokio::sync::watch::Sender<bool>,
    ) -> Result<(), ApiError> {
        loop {
            match text_rx.recv().await {
                Some(TextCmd::Push(sentence)) => {
                    let frame = wire::input_transcript(&sentence, opts);
                    sender
                        .send_text(frame.to_string())
                        .await
                        .map_err(|e| ApiError::connection(Some(e.to_string())))?;
                    let _ = input_sent.send(true);
                }
                // The `_FlushSentinel` only flushes the host's tokenizer; nothing goes on the wire.
                Some(TextCmd::Flush) => {}
                Some(TextCmd::End) | None => break,
            }
        }

        // Sent even for empty input, so the gateway still sends `done`.
        sender
            .send_text(wire::session_flush().to_string())
            .await
            .map_err(|e| ApiError::connection(Some(e.to_string())))?;
        // "needed in case empty input is sent" -- the flush itself releases the receive half.
        let _ = input_sent.send(true);
        std::future::pending::<()>().await;
        Ok(())
    }

    async fn recv_loop(
        receiver: &mut dyn crate::transport::WsReceiver,
        audio_tx: &mpsc::UnboundedSender<Result<SynthesizedAudio, ApiError>>,
        ctx: RecvCtx<'_>,
    ) -> Result<(), ApiError> {
        let RecvCtx {
            provider,
            request_id,
            sample_rate,
            mut input_sent,
            clock,
            first_message_timeout,
        } = ctx;
        let _ = input_sent.wait_for(|sent| *sent).await;
        // Only the first read after text is on the wire is bounded; see `Pump::timeout`.
        let mut awaiting_first = true;
        let mut segment_id: Option<String> = None;
        // Alignment arrives on its own message and rides out on the next audio frame.
        let mut pending_transcript: Vec<TimedString> = Vec::new();

        loop {
            let recv = receiver.recv();
            let received = if awaiting_first {
                awaiting_first = false;
                within(clock, first_message_timeout, recv).await
            } else {
                recv.await
            };
            let msg = match received {
                Ok(msg) => msg,
                Err(err) => return Err(connect_error(WHAT, err)),
            };
            match msg {
                WsMessage::Closed => {
                    return Err(ApiError::status(
                        "Gateway connection closed unexpectedly",
                        -1,
                        Some(request_id.clone()),
                        None,
                    ));
                }
                WsMessage::Other => continue,
                WsMessage::Text(raw) => {
                    let Some(incoming) = wire::decode(&raw, provider) else { continue };
                    // The *first* message carrying a session id names the segment.
                    if segment_id.is_none()
                        && let Some(id) = incoming.session_id()
                    {
                        segment_id = Some(id.to_owned());
                    }
                    match incoming {
                        wire::Incoming::OutputAudio { audio, .. } => {
                            let Ok(data) = base64::engine::general_purpose::STANDARD.decode(&audio)
                            else {
                                continue;
                            };
                            let out = SynthesizedAudio {
                                timed_transcript: std::mem::take(&mut pending_transcript),
                                segment_id: segment_id.clone().unwrap_or_default(),
                                ..SynthesizedAudio::new(
                                    mono(data, sample_rate),
                                    request_id.clone(),
                                )
                            };
                            if audio_tx.send(Ok(out)).is_err() {
                                return Ok(());
                            }
                        }
                        wire::Incoming::OutputAlignment { tokens, .. } => {
                            pending_transcript.extend(tokens);
                        }
                        wire::Incoming::Done { .. } => {
                            // `output_emitter.end_input()` flushes a final, possibly empty frame;
                            // the driver keys on `is_final`.
                            let out = SynthesizedAudio {
                                is_final: true,
                                timed_transcript: std::mem::take(&mut pending_transcript),
                                segment_id: segment_id.clone().unwrap_or_default(),
                                ..SynthesizedAudio::new(
                                    mono(Vec::new(), sample_rate),
                                    request_id.clone(),
                                )
                            };
                            let _ = audio_tx.send(Ok(out));
                            return Ok(());
                        }
                        wire::Incoming::Error { raw } => {
                            // `APIError(f"...: {msg.data}")` -- the whole frame; the detail is in
                            // sibling keys.
                            return Err(ApiError::new(format!(
                                "LiveKit Inference TTS returned error: {raw}"
                            )));
                        }
                        _ => continue,
                    }
                }
            }
        }
    }
}

struct RecvCtx<'a> {
    provider: &'a str,
    request_id: String,
    sample_rate: u32,
    input_sent: tokio::sync::watch::Receiver<bool>,
    clock: &'a dyn Clock,
    first_message_timeout: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{FakeWs, text};
    use lk_agents_abi::TokioSpawner;
    use serde_json::json;

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
        NativeTts::new(opts(), ws, Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())))
    }

    fn map(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
        pairs.iter().map(|(k, v)| ((*k).to_owned(), v.clone())).collect()
    }

    // Catches: an unbounded handshake, or one bounded on tokio's clock rather than the session's.
    #[tokio::test]
    async fn a_handshake_that_never_completes_is_a_retryable_timeout_on_the_session_clock() {
        let clock = Arc::new(lk_agents_abi::VirtualClock::new());
        let ws = Arc::new(FakeWs::hanging());
        let tts = tts(ws.clone()).with_clock(clock.clone());
        let mut source = tts.synthesize("hi").await.unwrap();

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
        let tts = tts(ws.clone()).with_clock(clock.clone());
        let mut source = tts.synthesize("hi").await.unwrap();

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
        sink.push("Hello there.".to_owned()).unwrap();

        let sent = until_sent(&ws, 2).await;
        assert_eq!(sent.len(), 2, "session.create, then the sentence, with the input open: {sent:?}");
        assert_eq!(sent[0]["type"], json!("session.create"));
        assert_eq!(sent[1]["type"], json!("input_transcript"));
        assert_eq!(sent[1]["transcript"], json!("Hello there. "));

        // A flush is the host tokenizer's business: nothing more goes on the wire for it.
        sink.flush().unwrap();
        sink.push("How are you?".to_owned()).unwrap();
        let sent = until_sent(&ws, 3).await;
        assert_eq!(sent.len(), 3, "{sent:?}");
        assert_eq!(sent[2]["transcript"], json!("How are you? "));

        sink.end_input();
        let sent = until_sent(&ws, 4).await;
        assert_eq!(sent.len(), 4, "{sent:?}");
        assert_eq!(sent[3], json!({"type": "session.flush"}), "one flush, at the end of input");
    }

    // Catches: the trailing space added twice (host and wire) or not at all. Each sentence is its
    // own `input_transcript` carrying `converted + " "` -- one space, never two.
    #[tokio::test]
    async fn each_sentence_carries_exactly_one_trailing_space() {
        let ws = Arc::new(FakeWs::new(vec![]).keep_open());
        let tts = tts(ws.clone());
        let (sink, _source) = tts.stream().await.unwrap();
        for sentence in ["One.", "Two!", "Three?"] {
            sink.push(sentence.to_owned()).unwrap();
        }
        sink.end_input();

        let sent = until_sent(&ws, 5).await;
        let transcripts: Vec<&str> = sent
            .iter()
            .filter(|v| v["type"] == json!("input_transcript"))
            .map(|v| v["transcript"].as_str().unwrap())
            .collect();
        assert_eq!(transcripts, ["One. ", "Two! ", "Three? "]);
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
        assert!(
            !sent.iter().any(|v| v["type"] == json!("input_transcript")),
            "nothing to synthesise"
        );
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

    // Catches: reporting a gateway `error` frame as a clean end of stream.
    #[tokio::test]
    async fn a_gateway_error_frame_fails_the_stream() {
        let ws = Arc::new(FakeWs::new(vec![text(
            json!({"type": "error", "message": "voice not found"}),
        )]));
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
}
