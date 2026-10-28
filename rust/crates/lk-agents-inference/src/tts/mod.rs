//! `inference/tts.py` -- the gateway TTS, WebSocket + JSON (§1.4). `synthesize()` is the duplex
//! path with one push and the input closed, as in the reference, so [`NativeTts::stream`] always
//! returns `Some`. A synthesis takes a configured socket from the client's pool and gives it back
//! once the gateway said `done`, as the reference's `ConnectionPool` does.
//! Text is normalised, split into sentences and converted to the provider's markup here, as
//! `inference/tts.py`'s `SynthesizeStream` does.

pub mod machine;
mod ports;
mod pump;
pub use ports::TtsHost;
pub mod wire;

use crate::abi::{
    StreamSink, StreamSource, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource,
};
use crate::headers::{BoundIdentity, HEADER_SESSION_ID, Headers, gateway_headers};
use crate::ids::{RequestKind, create_inference_request_id};
use async_trait::async_trait;
use lk_agents_net::transport::{
    WsReceiver, WsSender, WsTransport, open_ws, stream_closed, to_ws_url,
};

use futures_util::future::{Either, select};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::tts::stream_adapter::SentenceTextProcessor;
use lk_agents_abi::tts::text::{TranscriptProcessor, TtsTextProcessorFactory};
use lk_agents_abi::utils::connection_pool::ConnectionPool;
use lk_agents_abi::{ApiConnectOptions, Clock, SentenceSplitter, Ts};
use lk_agents_abi::{Spawner, SpawnerExt};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::collections::HashMap;
use std::pin::pin;
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::atomic::{AtomicBool, Ordering};
#[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, watch};
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

/// One gateway TTS client as one session holds it: the shared client, and the identity of the
/// job that session serves (two sessions sharing an instance bill each its own job).
#[derive(Clone)]
pub struct NativeTts<P: TtsTextProcessorFactory> {
    speech: std::marker::PhantomData<P>,
    shared: Arc<TtsShared>,
    identity: Arc<BoundIdentity>,
    conn_options: Option<ApiConnectOptions>,
    /// The host provider this view serves: its `error` event is each failure the core reports.
    reporter: Option<Arc<dyn lk_agents_abi::Tts>>,
}

impl<P: TtsTextProcessorFactory> std::ops::Deref for NativeTts<P> {
    type Target = TtsShared;

    fn deref(&self) -> &TtsShared {
        &self.shared
    }
}

/// What every session's view of one client shares: its options and its pooled connections.
pub struct TtsShared {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    opts: Arc<Mutex<TtsOptions>>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    expressive: AtomicBool,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    /// Bound into, not rebuilt, so `update_options` state survives a session binding its own.
    clock: Arc<lk_agents_abi::BoundClock>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    connections: Arc<Mutex<Connections>>,
}

/// One configured gateway socket, and the session id it was opened under.
struct Connection {
    sender: Arc<dyn WsSender>,
    receiver: Box<dyn WsReceiver>,
    session_id: Option<String>,
    configuration: String,
}

/// `utils.ConnectionPool(max_session_duration=300, mark_refreshed_on_get=True)`.
struct Connections {
    pool: ConnectionPool,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    idle: HashMap<u64, Connection>,
    next_id: u64,
}

impl Default for Connections {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn default() -> Self {
        Connections {
            pool: ConnectionPool::new(Some(300.0), true),
            idle: HashMap::new(),
            next_id: 0,
        }
    }
}

impl Connections {
    /// An idle socket young enough to reuse, and the expired ones to close.
    fn take(&mut self, now: f64) -> (Option<(u64, Connection)>, Vec<Connection>) {
        let reused = self.pool.get(now).and_then(|id| self.idle.remove(&id).map(|c| (id, c)));
        let mut expired = Vec::new();
        while let Some(id) = self.pool.pop_closing() {
            expired.extend(self.idle.remove(&id));
        }
        (reused, expired)
    }

    fn opened(&mut self, now: f64) -> u64 {
        self.next_id += 1;
        self.pool.connected(self.next_id, now, self.pool.generation(), None);
        self.next_id
    }

    fn give_back(&mut self, id: u64, connection: Connection) {
        self.idle.insert(id, connection);
        self.pool.put(id);
    }

    fn forget(&mut self, id: u64) {
        self.pool.remove(id);
        while self.pool.pop_closing().is_some() {}
    }
}

fn pool_secs(clock: &dyn Clock) -> f64 {
    clock.now().as_nanos() as f64 / 1e9
}

impl<P: TtsTextProcessorFactory> NativeTts<P> {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { conn_options: conn_options.or(self.conn_options), ..self.clone() }
    }

    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    pub fn new(opts: TtsOptions, ws: Arc<dyn WsTransport>, spawner: Arc<dyn Spawner>) -> Self {
        let shared = TtsShared {
            opts: Arc::new(Mutex::new(opts)),
            expressive: AtomicBool::new(false),
            ws,
            spawner,
            clock: lk_agents_abi::BoundClock::new(),
            connections: Arc::default(),
        };
        NativeTts {
            speech: std::marker::PhantomData,
            shared: Arc::new(shared),
            identity: Arc::default(),
            conn_options: None,
            reporter: None,
        }
    }

    /// The same client as a session serving another job holds it: its syntheses carry that job's
    /// identification.
    /// `reporter` is the host provider whose `error` events this view's failures are.
    pub fn identified(
        &self,
        headers: Headers,
        reporter: Option<Arc<dyn lk_agents_abi::Tts>>,
    ) -> Self {
        let mut view = NativeTts::new(
            self.opts.lock().unwrap().clone(),
            self.ws.clone(),
            self.spawner.clone(),
        );
        let shared = Arc::get_mut(&mut view.shared).expect("a newly constructed view is unshared");
        shared.opts = self.opts.clone();
        shared.connections = self.connections.clone();
        shared.clock = self.clock.clone();
        view.set_expressive(self.expressive.load(Ordering::SeqCst));
        view.conn_options = self.conn_options;
        view.reporter = reporter;
        view.identity.bind(headers);
        view
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
        let create_frame = wire::session_create(&opts);
        let headers = headers.into_vec();
        let configuration = connection_configuration(&create_frame, &headers);
        self.spawner.spawn_unboxed(async move {
            let opened = open_ws(
                ws.as_ref(),
                &url,
                headers,
                &*clock,
                &opts.conn_options,
                WHAT,
                &create_frame,
            );
            if let Ok((sender, receiver)) = opened.await {
                let mut connections = connections.lock().unwrap();
                let id = connections.opened(pool_secs(&*clock));
                connections
                    .give_back(id, Connection { sender, receiver, session_id, configuration });
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
impl<P: TtsTextProcessorFactory> NativeTts<P> {
    /// `_set_expressive`: whether syntheses opened from here batch sentences for markup and drop
    /// bracket cues from the alignment.
    pub fn set_expressive(&self, expressive: bool) {
        self.expressive.store(expressive, Ordering::SeqCst);
    }

    pub fn model(&self) -> String {
        self.opts.lock().unwrap().model.clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.opts.lock().unwrap().sample_rate
    }

    /// `TTSCapabilities.aligned_transcript`, recomputed from the model and extra kwargs.
    pub fn aligned_transcript(&self) -> bool {
        let opts = self.opts.lock().unwrap();
        wire::has_aligned_transcript(&opts.model, &opts.extra_kwargs)
    }

    /// `update_options`. Unlike the LLM's, `extra` merges (`.update()`).
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
            opts.language = Some(
                lk_agents_abi::tokenize::language::LanguageCode::new(language).as_str().to_owned(),
            );
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
    pub async fn stream_for_host(
        &self,
        headers: Option<Vec<(String, String)>>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<(TtsSink, TtsSource), ProviderError> {
        let (sink, source) = self
            .configured(conn_options)
            .stream_identified(headers.map(|pairs| pairs.into_iter().collect()))?;
        Ok((Box::new(sink), Box::new(source)))
    }

    #[allow(clippy::disallowed_methods, reason = "S0: migrate live execution/state to runtime")]
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    fn open(
        &self,
        identification: Option<&Headers>,
    ) -> Result<(TtsSinkHandle, TtsSourceHandle), ProviderError> {
        let mut opts = self.opts.lock().unwrap().clone();
        if let Some(conn) = self.conn_options {
            opts.conn_options = conn;
        }
        let provider = opts.model.split('/').next().unwrap_or("").to_owned();
        let expressive = self.expressive.load(Ordering::SeqCst);
        let identification = match identification {
            Some(headers) => Some(headers.clone()),
            None => self.identity.for_connection(),
        };
        let identification = identification.as_ref();

        opts.conn_options.validate().map_err(|error| ApiError::new(error.to_string()))?;
        let (text_tx, text_rx) = mpsc::channel::<TextCmd>(TEXT_QUEUE_CAPACITY);
        let (audio_tx, audio_rx) =
            mpsc::channel::<Result<SynthesizedAudio, ApiError>>(AUDIO_QUEUE_CAPACITY);
        let (stop, stopped) = watch::channel(false);
        let (done, settled) = oneshot::channel();
        let (failure, failed) = watch::channel(None);
        let clock = self.clock.clone() as Arc<dyn Clock>;
        let accepted_at = clock.now();

        let headers = gateway_headers(identification, None, None, &opts.api_key, &opts.api_secret)?;
        let session_id = headers.get(HEADER_SESSION_ID).map(str::to_owned);

        let pump = Pump {
            connections: self.connections.clone(),
            session_id,
            ws: self.ws.clone(),
            url: format!("{}/tts?model={}", to_ws_url(&opts.base_url), opts.model),
            headers: headers.into_vec(),
            clock: clock.clone(),
            accepted_at,
            stopped,
            done: Some(done),
            failure,
            create_frame: wire::session_create(&opts),
            sample_rate: opts.sample_rate,
            opts,
            provider,
            transcript: P::transcript(expressive),
            text_rx,
            audio_tx,
        };
        let prep = TextPrep {
            pending_bytes: 0,
            markup: P::markup(wire::markup_provider_key(&pump.opts.model)),
            sentences: P::sentence_splitter(Some(&pump.provider), expressive),
        };
        self.spawner.spawn_unboxed(pump.run());

        Ok((
            TtsSinkHandle { text: text_tx, prep: Mutex::new(prep) },
            TtsSourceHandle {
                audio: audio_rx,
                done: false,
                stop,
                settled: Some(settled),
                failed,
                clock,
            },
        ))
    }
}

#[async_trait]
impl<P: TtsTextProcessorFactory> Tts for NativeTts<P> {
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    /// The provider's `error` event, on the host provider this view serves.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(reporter) = &self.reporter {
            reporter.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        "livekit.agents.inference.tts.TTS".into()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId {
            label: Tts::label(self),
            model: self.model(),
            provider: "livekit".into(),
        }
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
    End,
    Failed,
}

pub struct TtsSinkHandle {
    text: mpsc::Sender<TextCmd>,
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    prep: Mutex<TextPrep>,
}

/// `SynthesizeStream`'s input side: the markup normalised, the text split into the provider's
/// sentences, each converted to its markup.
struct TextPrep {
    pending_bytes: usize,
    markup: Option<Box<dyn SentenceTextProcessor>>,
    sentences: Box<dyn SentenceSplitter>,
}

impl TtsSinkHandle {
    fn send(&self, sentences: Vec<String>) -> Result<(), ProviderError> {
        let prep = self.prep.lock().unwrap();
        for sentence in sentences {
            let converted = match &prep.markup {
                Some(markup) => markup.convert(&markup.normalize(&sentence)),
                None => sentence,
            };
            let permit = self.text.try_reserve().map_err(|_| stream_closed(WHAT))?;
            if self.text.capacity() == 0 {
                return Err(ApiError::new("TTS pending text queue is full").into());
            }
            permit.send(TextCmd::Push(converted));
        }
        Ok(())
    }
}

impl StreamSink<String> for TtsSinkHandle {
    fn push(&self, item: String) -> Result<(), ProviderError> {
        if item.len() > MAX_PENDING_TEXT_BYTES {
            return Err(ApiError::new("TTS text exceeds the pending text limit").into());
        }
        let sentences = {
            let mut prep = self.prep.lock().unwrap();
            if prep.pending_bytes.saturating_add(item.len()) > MAX_PENDING_TEXT_BYTES {
                return Err(ApiError::new("TTS pending text exceeds its limit").into());
            }
            prep.pending_bytes += item.len();
            let normalized = match &prep.markup {
                Some(markup) => markup.normalize(&item),
                None => item,
            };
            let sentences = prep.sentences.push(&normalized);
            prep.pending_bytes =
                prep.pending_bytes.saturating_sub(sentences.iter().map(String::len).sum());
            sentences
        };
        self.send(sentences)
    }

    /// The `_FlushSentinel`: whatever the tokenizer holds goes out as a sentence.
    fn flush(&self) -> Result<(), ProviderError> {
        let sentences = {
            let mut prep = self.prep.lock().unwrap();
            prep.pending_bytes = 0;
            prep.sentences.flush()
        };
        self.send(sentences)
    }

    fn end_input(&self) {
        let sentences = {
            let mut prep = self.prep.lock().unwrap();
            prep.pending_bytes = 0;
            prep.sentences.flush()
        };
        let command = match self.send(sentences) {
            Ok(()) => TextCmd::End,
            Err(_) => TextCmd::Failed,
        };
        let _ = self.text.try_send(command);
    }
}

pub struct TtsSourceHandle {
    audio: mpsc::Receiver<Result<SynthesizedAudio, ApiError>>,
    done: bool,
    stop: watch::Sender<bool>,
    settled: Option<oneshot::Receiver<Result<(), ApiError>>>,
    failed: watch::Receiver<Option<ApiError>>,
    clock: Arc<dyn Clock>,
}

impl StreamSource<SynthesizedAudio> for TtsSourceHandle {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            let next = if let Ok(item) = self.audio.try_recv() {
                Some(item)
            } else {
                let received = {
                    match select(pin!(self.audio.recv()), pin!(self.failed.changed())).await {
                        Either::Left((next, _)) => Ok(next),
                        Either::Right(_) => Err(()),
                    }
                };
                match received {
                    Ok(next) => next,
                    Err(()) => self.failed.borrow().clone().map(Err),
                }
            };
            match next {
                Some(Ok(audio)) => Ok(Some(audio)),
                Some(Err(error)) => {
                    self.done = true;
                    Err(error.into())
                }
                None => {
                    self.done = true;
                    match self.settled.take() {
                        Some(settled) => match settled.await {
                            Ok(Ok(())) => Ok(None),
                            Ok(Err(error)) => Err(error.into()),
                            Err(_) => {
                                Err(ApiError::new("TTS pump ended without a terminal outcome")
                                    .into())
                            }
                        },
                        None => Ok(None),
                    }
                }
            }
        })
    }

    fn cancel_safe(&self) -> bool {
        true
    }

    fn close(
        mut self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move {
            let _ = self.stop.send(true);
            self.done = true;
            self.audio.close();
            let settled = self.settled.take();
            lk_agents_abi::Closed::bounded(self.clock.as_ref(), deadline, async move {
                if let Some(settled) = settled {
                    let _ = settled.await;
                }
            })
            .await
        })
    }
}

impl Drop for TtsSourceHandle {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
    }
}

pub const TEXT_QUEUE_CAPACITY: usize = 64;
pub const AUDIO_QUEUE_CAPACITY: usize = 8;
const MAX_PENDING_TEXT_BYTES: usize = 64 * 1024;

struct Pump {
    #[allow(clippy::disallowed_types, reason = "S0: migrate live execution/state to runtime")]
    connections: Arc<Mutex<Connections>>,
    session_id: Option<String>,
    ws: Arc<dyn WsTransport>,
    url: String,
    headers: Vec<(String, String)>,
    clock: Arc<dyn Clock>,
    accepted_at: Ts,
    stopped: watch::Receiver<bool>,
    done: Option<oneshot::Sender<Result<(), ApiError>>>,
    failure: watch::Sender<Option<ApiError>>,
    create_frame: serde_json::Value,
    opts: TtsOptions,
    provider: String,
    sample_rate: u32,
    transcript: Box<dyn TranscriptProcessor>,
    text_rx: mpsc::Receiver<TextCmd>,
    audio_tx: mpsc::Sender<Result<SynthesizedAudio, ApiError>>,
}

fn connection_configuration(frame: &serde_json::Value, headers: &[(String, String)]) -> String {
    let identity: Vec<_> =
        headers.iter().filter(|(name, _)| !name.eq_ignore_ascii_case("authorization")).collect();
    format!("{frame}|{}", serde_json::to_string(&identity).expect("string headers serialize"))
}
