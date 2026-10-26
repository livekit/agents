//! `inference/tts.py` -- the gateway TTS, WebSocket + JSON (§1.4). `synthesize()` is the duplex
//! path with one push and the input closed, as in the reference, so [`NativeTts::stream`] always
//! returns `Some`. A synthesis takes a configured socket from the client's pool and gives it back
//! once the gateway said `done`, as the reference's `ConnectionPool` does.
//! Text is normalised, split into sentences and converted to the provider's markup here, as
//! `inference/tts.py`'s `SynthesizeStream` does.

#[cfg(feature = "python")]
mod python;
pub mod wire;

use crate::abi::{
    StreamSink, StreamSource, SynthesizedAudio, Tts, TtsCapabilities, TtsSink,
    TtsSource,
};
use crate::headers::{BoundIdentity, HEADER_SESSION_ID, Headers, gateway_headers};
use crate::ids::{RequestKind, create_inference_request_id};
use crate::transport::{
    WsMessage, WsReceiver, WsSender, WsTransport, connect_error, next_terminal, open_ws,
    stream_closed, to_ws_url, within,
};
use lk_agents_abi::utils::connection_pool::ConnectionPool;
use std::collections::HashMap;
use crate::vad::machine::mono;
use async_trait::async_trait;
use base64::Engine as _;
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tts::markup::{Markup, SentenceTokenizerConfig, drop_bracket_cues, sentence_tokenizer_of};
use lk_agents_abi::{ApiConnectOptions, Clock, SentenceSplitter, TimedString};
use lk_agents_media::blingfire::{SentenceBuffer, SentenceBufferOptions};
use std::sync::atomic::{AtomicBool, Ordering};
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
    /// `_expressive`: the framework sets it before a synthesis opens.
    expressive: AtomicBool,
    identity: BoundIdentity,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    /// Bound into, not rebuilt, so `update_options` state survives a session binding its own.
    clock: Arc<lk_agents_abi::BoundClock>,
    connections: Arc<Mutex<Connections>>,
}

/// One configured gateway socket, and the session id it was opened under.
struct Connection {
    sender: Arc<dyn WsSender>,
    receiver: Box<dyn WsReceiver>,
    session_id: Option<String>,
}

/// `utils.ConnectionPool(max_session_duration=300, mark_refreshed_on_get=True)`.
struct Connections {
    pool: ConnectionPool,
    idle: HashMap<u64, Connection>,
    next_id: u64,
}

impl Default for Connections {
    fn default() -> Self {
        Connections { pool: ConnectionPool::new(Some(300.0), true), idle: HashMap::new(), next_id: 0 }
    }
}

impl Connections {
    /// An idle socket young enough to reuse, and the expired ones to close.
    fn take(&mut self, now: f64) -> (Option<(u64, Connection)>, Vec<Connection>) {
        let reused = self.pool.get(now).and_then(|id| self.idle.remove(&id).map(|c| (id, c)));
        let mut expired = Vec::new();
        while let Some(id) = self.pool.next_closing() {
            expired.extend(self.idle.remove(&id));
        }
        (reused, expired)
    }

    fn opened(&mut self, now: f64) -> u64 {
        self.next_id += 1;
        self.pool.connected(self.next_id, now);
        self.next_id
    }

    fn give_back(&mut self, id: u64, connection: Connection) {
        self.idle.insert(id, connection);
        self.pool.put(id);
    }

    fn forget(&mut self, id: u64) {
        self.pool.remove(id);
        while self.pool.next_closing().is_some() {}
    }
}

fn pool_secs(clock: &dyn Clock) -> f64 {
    clock.now().as_nanos() as f64 / 1e9
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
    #[host(kwarg(default = 3))]
    pub max_retry: u32,
}

impl NativeTts {
    pub fn new(opts: TtsOptions, ws: Arc<dyn WsTransport>, spawner: Arc<dyn Spawner>) -> Self {
        NativeTts {
            opts: Mutex::new(opts),
            expressive: AtomicBool::new(false),
            identity: BoundIdentity::default(),
            ws,
            spawner,
            clock: lk_agents_abi::BoundClock::new(),
            connections: Arc::default(),
        }
    }

    /// `prewarm()`: a configured socket waits in the pool for the first synthesis.
    pub fn prewarm_identified(&self, identification: Option<Headers>) {
        let opts = self.opts.lock().unwrap().clone();
        let identification = identification.or_else(|| self.identity.for_connection());
        let Ok(headers) =
            gateway_headers(identification.as_ref(), None, None, &opts.api_key, &opts.api_secret)
        else {
            return;
        };
        let session_id = headers.get(HEADER_SESSION_ID).map(str::to_owned);
        let (ws, clock, connections) =
            (self.ws.clone(), self.clock.clone() as Arc<dyn Clock>, self.connections.clone());
        let url = format!("{}/tts?model={}", to_ws_url(&opts.base_url), opts.model);
        let timeout = Duration::from_secs_f64(opts.conn_options.timeout.max(0.0));
        let create_frame = wire::session_create(&opts);
        self.spawner.spawn_unboxed(async move {
            let opened =
                open_ws(ws.as_ref(), &url, headers.into_vec(), &*clock, timeout, WHAT, &create_frame);
            if let Ok((sender, receiver)) = opened.await {
                let mut connections = connections.lock().unwrap();
                let id = connections.opened(pool_secs(&*clock));
                connections.give_back(id, Connection { sender, receiver, session_id });
            }
        });
    }

    pub fn with_clock(self, clock: Arc<dyn Clock>) -> Self {
        self.clock.bind(clock);
        self
    }

    pub fn bind_clock(&self, clock: Arc<dyn Clock>) {
        self.clock.bind(clock);
    }

    /// The client as the provider a session holds directly.
    pub fn as_tts(self: &Arc<Self>) -> Arc<dyn Tts> {
        Arc::clone(self) as Arc<dyn Tts>
    }

    /// Headers are per stream, not a client field: each synthesis gets its own session id.
    /// `None`: the identity a session bound, else no job, the native headers.
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
    as_dyn(Tts = as_tts),
    stub = "def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
    stub = "def bind_clock(self, clock: SessionClock) -> None: ...",
    stub = "def prewarm(self, *, headers: list[tuple[str, str]] | None = None) -> None: ..."
))]
impl NativeTts {
    /// `_set_expressive`: whether syntheses opened from here batch sentences for markup and drop
    /// bracket cues from the alignment.
    #[host(method)]
    pub fn set_expressive(&self, expressive: bool) {
        self.expressive.store(expressive, Ordering::SeqCst);
    }

    /// `get_inference_headers()` of the job the client now serves; every later synthesis carries
    /// them, with a fresh session id.
    #[host(method)]
    pub fn bind_identification(&self, headers: Vec<(String, String)>) {
        self.identity.bind(headers.into_iter().collect());
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
        let expressive = self.expressive.load(Ordering::SeqCst);
        let identification = match identification {
            Some(headers) => Some(headers.clone()),
            None => self.identity.for_connection(),
        };
        let identification = identification.as_ref();

        let (text_tx, text_rx) = mpsc::unbounded_channel::<TextCmd>();
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Result<SynthesizedAudio, ApiError>>();

        let headers =
            gateway_headers(identification, None, None, &opts.api_key, &opts.api_secret)?;
        let session_id = headers.get(HEADER_SESSION_ID).map(str::to_owned);

        let pump = Pump {
            connections: self.connections.clone(),
            session_id,
            ws: self.ws.clone(),
            url: format!("{}/tts?model={}", to_ws_url(&opts.base_url), opts.model),
            headers: headers.into_vec(),
            timeout: Duration::from_secs_f64(opts.conn_options.timeout.max(0.0)),
            clock: self.clock.clone() as Arc<dyn Clock>,
            create_frame: wire::session_create(&opts),
            sample_rate: opts.sample_rate,
            opts,
            provider,
            expressive,
            text_rx,
            audio_tx,
        };
        let (max_token_len, min_token_len, xml_aware) = sentence_tokenizer_of(&pump.provider, expressive);
        let sentences = SentenceBuffer::with_options(SentenceBufferOptions {
            retain_format: false,
            ..SentenceBufferOptions::from_config(SentenceTokenizerConfig {
                max_token_len,
                min_token_len,
                xml_aware,
            })
        });
        let prep = TextPrep {
            markup: Markup::from_provider_key(wire::markup_provider_key(&pump.opts.model)),
            sentences: Box::new(sentences),
        };
        self.spawner.spawn_unboxed(pump.run());

        Ok((
            TtsSinkHandle { text: text_tx, prep: Mutex::new(prep) },
            TtsSourceHandle { audio: audio_rx, done: false },
        ))
    }
}

#[async_trait]
impl Tts for NativeTts {
    fn label(&self) -> String {
        "livekit.agents.inference.tts.TTS".into()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId { label: Tts::label(self), model: self.model(), provider: "livekit".into() }
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

    fn markup_provider(&self) -> String {
        wire::markup_provider_key(&self.model()).to_owned()
    }

    /// The gateway stream lowers the markup itself.
    fn lowers_markup(&self) -> bool {
        true
    }

    fn set_expressive(&self, on: bool) {
        NativeTts::set_expressive(self, on);
    }

    fn prewarm(&self) {
        self.prewarm_identified(None);
    }

    /// `synthesize(text)` = `_synthesize_with_stream(text)`: the gateway has no one-shot endpoint.
    async fn synthesize(&self, text: &str) -> Result<TtsSource, ProviderError> {
        let (sink, source) = self.open(None)?;
        sink.push(text.to_owned())?;
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
    prep: Mutex<TextPrep>,
}

/// `SynthesizeStream`'s input side: the markup normalised, the text split into the provider's
/// sentences, each converted to its markup.
struct TextPrep {
    markup: Markup,
    sentences: Box<dyn SentenceSplitter>,
}

impl TtsSinkHandle {
    fn send(&self, sentences: Vec<String>) -> Result<(), ProviderError> {
        let markup = self.prep.lock().unwrap().markup;
        for sentence in sentences {
            let converted = markup.convert(&markup.normalize(&sentence));
            self.text.send(TextCmd::Push(converted)).map_err(|_| stream_closed(WHAT))?;
        }
        Ok(())
    }
}

impl StreamSink<String> for TtsSinkHandle {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        let sentences = {
            let mut prep = self.prep.lock().unwrap();
            let normalized = prep.markup.normalize(&item);
            prep.sentences.push(&normalized)
        };
        self.send(sentences)
    }

    /// The `_FlushSentinel`: whatever the tokenizer holds goes out as a sentence.
    fn flush(&self) -> Result<(), ProviderError> {
        let sentences = self.prep.lock().unwrap().sentences.flush();
        self.send(sentences)?;
        self.text.send(TextCmd::Flush).map_err(|_| stream_closed(WHAT))
    }

    fn end_input(&self) {
        let sentences = self.prep.lock().unwrap().sentences.flush();
        let _ = self.send(sentences);
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
    connections: Arc<Mutex<Connections>>,
    /// Of a socket this synthesis opens; a reused one keeps its own.
    session_id: Option<String>,
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
    sample_rate: u32,
    /// Bracket cues are dropped from the alignment: the provider aligned the converted text.
    expressive: bool,
    text_rx: mpsc::UnboundedReceiver<TextCmd>,
    audio_tx: mpsc::UnboundedSender<Result<SynthesizedAudio, ApiError>>,
}

impl Pump {
    async fn run(self) {
        let Pump {
            connections,
            session_id,
            ws,
            url,
            headers,
            timeout,
            clock,
            create_frame,
            opts,
            provider,
            sample_rate,
            expressive,
            text_rx,
            audio_tx,
        } = self;

        let (reused, expired) = connections.lock().unwrap().take(pool_secs(&*clock));
        for connection in expired {
            connection.sender.close().await;
        }
        let (id, mut connection) = match reused {
            Some(reused) => reused,
            None => {
                let opened =
                    open_ws(ws.as_ref(), &url, headers, &*clock, timeout, WHAT, &create_frame);
                match opened.await {
                    Ok((sender, receiver)) => {
                        let id = connections.lock().unwrap().opened(pool_secs(&*clock));
                        (id, Connection { sender, receiver, session_id })
                    }
                    Err(err) => {
                        let _ = audio_tx.send(Err(err));
                        return;
                    }
                }
            }
        };
        // `create_inference_request_id(connection.session_id, "tts")`: no session id yields a
        // bare shortuuid.
        let request_id =
            create_inference_request_id(connection.session_id.as_deref(), RequestKind::Tts, "");
        let sender = connection.sender.clone();

        // `input_sent_event`: nothing is read back until the session is configured and text (or
        // the flush) is on the wire.
        let (input_sent_tx, input_sent_rx) = tokio::sync::watch::channel(false);

        let send = Self::send_loop(sender.clone(), text_rx, &opts, input_sent_tx);
        let recv = Self::recv_loop(
            connection.receiver.as_mut(),
            &audio_tx,
            RecvCtx {
                provider: &provider,
                request_id,
                sample_rate,
                expressive,
                input_sent: input_sent_rx,
                clock: clock.as_ref(),
                first_message_timeout: timeout,
            },
        );

        let outcome = tokio::select! {
            r = send => r.map(|()| false),
            r = recv => r,
        };
        match outcome {
            // The gateway said `done`: the socket is configured and free for the next synthesis.
            Ok(true) => connections.lock().unwrap().give_back(id, connection),
            outcome => {
                if let Err(err) = outcome {
                    let _ = audio_tx.send(Err(err));
                }
                connections.lock().unwrap().forget(id);
                sender.close().await;
            }
        }
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
    ) -> Result<bool, ApiError> {
        let RecvCtx {
            provider,
            request_id,
            sample_rate,
            expressive,
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
        // An unclosed bracket cue, held across alignment messages.
        let mut held: Vec<TimedString> = Vec::new();

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
                                return Ok(false);
                            }
                        }
                        wire::Incoming::OutputAlignment { tokens, .. } => {
                            pending_transcript.extend(if expressive {
                                drop_bracket_cues(tokens, &mut held, false)
                            } else {
                                tokens
                            });
                        }
                        wire::Incoming::Done { .. } => {
                            if !held.is_empty() {
                                pending_transcript.extend(drop_bracket_cues(Vec::new(), &mut held, true));
                            }
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
                            return Ok(true);
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
    expressive: bool,
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
        for token in ["Hello there, this is", " the first sentence.", " And here comes", " the second"] {
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
        let tts = NativeTts::new(opts, ws.clone(), spawner);
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
