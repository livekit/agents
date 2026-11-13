use crate::abi::{
    StreamInput, StreamOutput, SynthesizedAudio, Tts, TtsCapabilities, TtsSink, TtsSource,
};
use crate::headers::{Headers, gateway_headers};
use crate::ids::{RequestKind, create_inference_request_id};
use crate::telemetry;
use futures_util::future::{Either, select};
use lk_agents_abi::Spawner;
use lk_agents_abi::errors::Invalid;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::inference::fallback::{TtsFallbackEntry, TtsFallbacks};
use lk_agents_abi::inference::models::InferenceProvider;
use lk_agents_abi::inference::tts_client::InferenceTts;
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::provider::{NativeLabel, ProviderEvents};
use lk_agents_abi::utils::connection_pool::ConnectionPoolState;
use lk_agents_abi::utils::stream::Entry;
use lk_agents_abi::{ApiConnectOptions, BoxFuture, Clock, SpeechChunk, Ts};
use lk_agents_net::connection_pool::ConnectionPool;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::{WsReceiver, WsSender, WsTransport, join_url, open_ws, to_ws_url};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::pin::pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use wire::FallbackModel;

// A synthesis takes a configured socket from the client's pool and gives it back only once the
// gateway said `done`.

pub mod emitter;
pub(crate) mod fed;
pub mod options;
pub mod pacing;
mod provider_options;
mod pump;
pub(crate) mod request;
pub mod sentence_adapter;
pub use request::{stream, synthesize};
pub(crate) mod stage;
pub mod stream;
pub mod wire;

const WHAT: &str = "LiveKit Inference TTS";

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

/// One gateway TTS client. Every job it is handed to shares it; each request carries the identity
/// of the job its telemetry context runs for, so two jobs sharing an instance bill each its own.
#[derive(Clone)]
pub struct GatewayTts {
    shared: Arc<TtsClient>,
    conn_options: Option<ApiConnectOptions>,
    /// The name logs and metrics give it: its host class's when its host built it.
    label: String,
    /// The host object it was built as, which hears each failure the core reports.
    events: Option<Arc<dyn ProviderEvents>>,
    /// How its text is split and paced, which every synthesis over it reads.
    options: Arc<options::OptionsCell>,
}

impl std::ops::Deref for GatewayTts {
    type Target = TtsClient;

    fn deref(&self) -> &TtsClient {
        &self.shared
    }
}

/// What every session's view of one client shares: its options and its pooled connections.
pub struct TtsClient {
    opts: Arc<Mutex<TtsOptions>>,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    /// Prewarms and stream pumps: each ends on its own (a pump once its handles are gone).
    tasks: Arc<TaskSet>,
    clock: Arc<dyn Clock>,
    connections: Arc<Mutex<Connections>>,
}

/// One configured gateway socket.
struct Connection {
    sender: Arc<dyn WsSender>,
    receiver: Box<dyn WsReceiver>,
    configuration: String,
}

/// `utils.ConnectionPool(max_session_duration=300, mark_refreshed_on_get=True)`.
struct Connections {
    pool: ConnectionPool,
    idle: HashMap<u64, Connection>,
    next_id: u64,
    /// The client was closed: no socket is pooled any more.
    closed: bool,
}

impl Default for Connections {
    fn default() -> Self {
        Connections {
            pool: ConnectionPool::new(Some(300.0), true),
            idle: HashMap::new(),
            next_id: 0,
            closed: false,
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

    /// Back to the pool, or handed back to be closed once the client is.
    fn give_back(&mut self, id: u64, connection: Connection) -> Option<Connection> {
        if self.closed {
            self.forget(id);
            return Some(connection);
        }
        self.idle.insert(id, connection);
        self.pool.put(id);
        None
    }

    /// Pools nothing more; the idle sockets, to close.
    fn close(&mut self) -> Vec<Connection> {
        self.closed = true;
        let ids: Vec<u64> = self.idle.keys().copied().collect();
        ids.into_iter()
            .filter_map(|id| {
                self.pool.remove(id);
                self.idle.remove(&id)
            })
            .collect()
    }

    fn forget(&mut self, id: u64) {
        self.pool.remove(id);
        while self.pool.pop_closing().is_some() {}
    }
}

impl GatewayTts {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { conn_options: conn_options.or(self.conn_options), ..self.clone() }
    }

    pub fn new(
        opts: TtsOptions,
        ws: Arc<dyn WsTransport>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let shared = TtsClient {
            opts: Arc::new(Mutex::new(opts)),
            ws,
            spawner,
            tasks: Arc::default(),
            clock,
            connections: Arc::default(),
        };
        let options = lk_agents_abi::tts::types::TtsOptions {
            pacing: TtsCapabilities { streaming: true, aligned_transcript: false }.default_pacing(),
            ..Default::default()
        };
        GatewayTts {
            shared: Arc::new(shared),
            conn_options: None,
            label: NativeLabel::InferenceTts.as_str().into(),
            events: None,
            options: Arc::new(options::OptionsCell::native(options)),
        }
    }

    /// `prewarm()`: a socket configured for the job `cx` runs for waits in the pool for its first
    /// synthesis.
    pub fn prewarm_for(&self, cx: &telemetry::Cx) {
        let opts = self.opts.lock().clone();
        let Ok(token) = lk_agents_net::token::inference_bearer(&opts.api_key, &opts.api_secret)
        else {
            return;
        };
        let headers = gateway_headers(cx, None, None, &token);
        let (ws, clock, connections) =
            (self.ws.clone(), self.clock.clone(), self.connections.clone());
        let query = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("model", &opts.model)
            .finish();
        let url = format!("{}?{query}", join_url(&to_ws_url(&opts.base_url), "tts"));
        let Ok(create_frame) = wire::session_create(&opts) else { return };
        let configuration = connection_configuration(&create_frame, &headers);
        self.tasks.spawn(self.spawner.as_ref(), "tts prewarm", async move {
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
                let refused = {
                    let mut connections = connections.lock();
                    let id = connections.opened(clock.now().as_secs_f64());
                    connections.give_back(id, Connection { sender, receiver, configuration })
                };
                if let Some(connection) = refused {
                    connection.sender.close().await;
                }
            }
        });
    }

    /// One synthesis for the job `cx` runs for.
    pub fn stream_for(
        &self,
        cx: &telemetry::Cx,
    ) -> Result<(SynthesisInput, SynthesisOutput), ProviderError> {
        self.open(cx)
    }
}

/// `inference/tts.py`'s WebSocket, over `lk-agents-inference::tts::GatewayTts`.
impl GatewayTts {
    pub fn model(&self) -> String {
        self.opts.lock().model.clone()
    }

    pub fn sample_rate(&self) -> u32 {
        self.opts.lock().sample_rate
    }

    /// `TTSCapabilities.aligned_transcript`, recomputed from the model and extra kwargs.
    pub fn aligned_transcript(&self) -> bool {
        let opts = self.opts.lock();
        wire::has_aligned_transcript(&opts.model, &opts.extra_kwargs)
    }

    /// `update_options`. Unlike the LLM's, `extra` merges (`.update()`).
    pub fn update_options(
        &self,
        model: Option<&str>,
        voice: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        let mut opts = self.opts.lock();
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

    fn open(&self, cx: &telemetry::Cx) -> Result<(SynthesisInput, SynthesisOutput), ProviderError> {
        let mut opts = self.opts.lock().clone();
        if let Some(conn) = self.conn_options {
            opts.conn_options = conn;
        }
        let provider = InferenceProvider::of_model(&opts.model);

        opts.conn_options.validate().map_err(|error| ApiError::new(error.to_string()))?;
        let (text_tx, text_rx) = mpsc::channel::<String>(TEXT_QUEUE_CAPACITY);
        let (audio_tx, audio_rx) =
            mpsc::channel::<Result<SynthesizedAudio, ApiError>>(AUDIO_QUEUE_CAPACITY);
        let stop = CancellationToken::new();
        let (done, settled) = oneshot::channel();
        let (failure, failed) = lk_agents_net::signal::signal(None);
        let gate = lk_agents_net::source::InputGate::default();
        let clock = self.clock.clone();
        let accepted_at = clock.now();

        let token = lk_agents_net::token::inference_bearer(&opts.api_key, &opts.api_secret)?;
        let headers = gateway_headers(cx, None, None, &token);
        let session_id = Some(crate::ids::shortuuid(crate::ids::SESSION_ID_PREFIX));

        let query = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("model", &opts.model)
            .finish();
        let pump = Pump {
            connections: self.connections.clone(),
            tasks: self.tasks.clone(),
            spawner: self.spawner.clone(),
            session_id,
            ws: self.ws.clone(),
            url: format!("{}?{query}", join_url(&to_ws_url(&opts.base_url), "tts")),
            headers,
            clock: clock.clone(),
            accepted_at,
            stopped: stop.clone(),
            done: Some(done),
            failure,
            gate: gate.clone(),
            create_frame: wire::session_create(&opts)?,
            sample_rate: opts.sample_rate,
            opts,
            provider,
            transcript: std::collections::BTreeMap::new(),
            text_rx,
            audio_tx,
        };
        self.tasks.spawn(self.spawner.as_ref(), "tts stream", pump.run());

        Ok((
            SynthesisInput { text: Mutex::new(Some(text_tx)), gate: gate.clone() },
            SynthesisOutput {
                audio: audio_rx,
                done: false,
                stop,
                settled: Some(settled),
                failed,
                gate,
                clock,
            },
        ))
    }
}

impl Tts for GatewayTts {
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Tts>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn conn_options(&self) -> Option<ApiConnectOptions> {
        self.conn_options
    }

    /// The provider's `error` event, on the host object it was built as.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(events) = &self.events {
            events.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        self.label.clone()
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
        wire::markup_provider(&self.model()).map_or("", |p| p.as_str()).to_owned()
    }

    fn options(&self) -> Arc<dyn lk_agents_abi::tts::types::SharedTtsOptions> {
        self.options.clone()
    }

    fn prewarm(&self, cx: &telemetry::Cx) {
        self.prewarm_for(cx);
    }

    /// `synthesize(text)` = `_synthesize_with_stream(text)`: the gateway has no one-shot endpoint.
    fn synthesize<'a>(
        &'a self,
        cx: &telemetry::Cx,
        chunk: &'a SpeechChunk,
    ) -> lk_agents_abi::BoxFuture<'a, Result<TtsSource, ProviderError>> {
        let opened = self.open(cx);
        Box::pin(async move {
            let (sink, source) = opened?;
            if !chunk.text.trim().is_empty() {
                sink.queue(chunk.text.clone())?;
            }
            sink.end_input();
            Ok(Box::new(source) as TtsSource)
        })
    }

    fn stream(
        &self,
        cx: &telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<lk_agents_abi::DuplexStream<TtsSink, TtsSource>, ProviderError>,
    > {
        let opened = self.open(cx);
        Box::pin(async move {
            opened.map(|(sink, source)| {
                lk_agents_abi::DuplexStream::new(
                    Box::new(sink) as TtsSink,
                    Box::new(source) as TtsSource,
                )
            })
        })
    }
}

/// `SynthesizeStream`'s input side: each chunk its stage prepared, sent to the gateway as is. The
/// input ends when the sender goes: the pump reads what was queued, then the end.
pub struct SynthesisInput {
    text: Mutex<Option<mpsc::Sender<String>>>,
    /// What a refused call answers: the synthesis's failure, else its end.
    gate: lk_agents_net::source::InputGate,
}

impl SynthesisInput {
    fn sender(&self) -> Result<mpsc::Sender<String>, ProviderError> {
        self.gate.admit()?;
        let sender = self.text.lock().clone();
        sender.ok_or_else(|| self.gate.refused())
    }

    /// Queues `text` without waiting: refused when the queue is full.
    fn queue(&self, text: String) -> Result<(), ProviderError> {
        self.sender()?.try_send(text).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                ApiError::new("TTS pending text queue is full").into()
            }
            mpsc::error::TrySendError::Closed(_) => self.gate.refused(),
        })
    }
}

impl StreamInput<SpeechChunk> for SynthesisInput {
    fn push(&self, chunk: SpeechChunk) -> Result<(), ProviderError> {
        self.queue(chunk.text)
    }

    /// Each chunk is a whole request already: a segment's end sends nothing more.
    fn flush(&self) -> Result<(), ProviderError> {
        Ok(())
    }

    fn end_input(&self) {
        self.gate.end();
        drop(self.text.lock().take());
    }

    /// Waits for room in the queue: a provider behind holds the feed, it never refuses text.
    fn send(&self, entry: Entry<SpeechChunk>) -> BoxFuture<'static, Result<(), ProviderError>> {
        let sender = match entry {
            Entry::Item(_) => self.sender(),
            Entry::Flush => return Box::pin(std::future::ready(Ok(()))),
            Entry::EndInput => {
                self.end_input();
                return Box::pin(std::future::ready(Ok(())));
            }
        };
        let gate = self.gate.clone();
        Box::pin(async move {
            let Entry::Item(chunk) = entry else { unreachable!("only an item is sent") };
            sender?.send(chunk.text).await.map_err(|_| gate.refused())
        })
    }
}

pub struct SynthesisOutput {
    audio: mpsc::Receiver<Result<SynthesizedAudio, ApiError>>,
    done: bool,
    stop: CancellationToken,
    settled: Option<oneshot::Receiver<Result<(), ApiError>>>,
    failed: lk_agents_net::signal::Cell<Option<ApiError>>,
    /// Its input's: ended with the synthesis, the failure kept by the pump as it happens.
    gate: lk_agents_net::source::InputGate,
    clock: Arc<dyn Clock>,
}

impl SynthesisOutput {
    /// The next audio or failure; what is already queued goes first, then whichever of audio or
    /// a published failure comes first.
    async fn next_item(&mut self) -> Option<Result<SynthesizedAudio, ApiError>> {
        if let Ok(item) = self.audio.try_recv() {
            return Some(item);
        }
        let audio = match select(pin!(self.audio.recv()), pin!(self.failed.changed())).await {
            Either::Left((next, _)) => Some(next),
            Either::Right(_) => None,
        };
        audio.unwrap_or_else(|| self.failed.get().map(Err))
    }

    /// The pump's terminal outcome once the audio ended.
    async fn settle(&mut self) -> Result<Option<SynthesizedAudio>, ProviderError> {
        let Some(settled) = self.settled.take() else { return Ok(None) };
        match settled.await {
            Ok(Ok(())) => Ok(None),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => Err(ApiError::new("TTS pump ended without a terminal outcome").into()),
        }
    }
}

impl StreamOutput<SynthesizedAudio> for SynthesisOutput {
    fn next(
        &mut self,
    ) -> lk_agents_abi::BoxFuture<'_, Result<Option<SynthesizedAudio>, ProviderError>> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            let next = match self.next_item().await {
                Some(Ok(audio)) => return Ok(Some(audio)),
                Some(Err(error)) => Err(error.into()),
                None => self.settle().await,
            };
            self.done = true;
            match &next {
                Err(error) => self.gate.fail(error.clone()),
                Ok(_) => self.gate.end(),
            }
            next
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
            self.gate.end();
            self.stop.cancel();
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

impl lk_agents_abi::utils::stream::ProviderStream<SynthesizedAudio> for SynthesisOutput {}

impl Drop for SynthesisOutput {
    fn drop(&mut self) {
        self.gate.end();
        self.stop.cancel();
    }
}

pub const TEXT_QUEUE_CAPACITY: usize = 64;
pub const AUDIO_QUEUE_CAPACITY: usize = 8;

struct Pump {
    connections: Arc<Mutex<Connections>>,
    /// Where a checked-out connection dropped unreleased is closed.
    tasks: Arc<TaskSet>,
    spawner: Arc<dyn Spawner>,
    session_id: Option<String>,
    ws: Arc<dyn WsTransport>,
    url: String,
    headers: Headers,
    clock: Arc<dyn Clock>,
    accepted_at: Ts,
    stopped: CancellationToken,
    done: Option<oneshot::Sender<Result<(), ApiError>>>,
    failure: lk_agents_net::signal::Producer<Option<ApiError>>,
    /// The input's answer: the failure, recorded as it is published.
    gate: lk_agents_net::source::InputGate,
    create_frame: serde_json::Value,
    opts: TtsOptions,
    provider: Option<InferenceProvider>,
    sample_rate: u32,
    /// Each segment's aligned words, waiting for the frame that carries them.
    transcript: std::collections::BTreeMap<String, Vec<lk_agents_abi::TranscriptChunk>>,
    text_rx: mpsc::Receiver<String>,
    audio_tx: mpsc::Sender<Result<SynthesizedAudio, ApiError>>,
}

fn connection_configuration(frame: &serde_json::Value, headers: &Headers) -> String {
    let identity: Vec<_> =
        headers.iter().filter(|(name, _)| !name.eq_ignore_ascii_case("authorization")).collect();
    format!("{frame}|{}", serde_json::to_string(&identity).expect("string headers serialize"))
}

impl InferenceTts for GatewayTts {
    type Env = ProcessEnv;
    type Options = options::OptionsCell;

    #[allow(clippy::too_many_arguments, reason = "Preserve the public host operation signature")]
    fn new(
        env: &ProcessEnv,
        label: String,
        events: Arc<dyn ProviderEvents>,
        sentence_tokenizer: lk_agents_abi::tts::types::OptionsTokenizer,
        model: String,
        voice: Option<String>,
        language: Option<String>,
        encoding: Option<String>,
        sample_rate: Option<u32>,
        base_url: Option<String>,
        api_key: Option<String>,
        api_secret: Option<String>,
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        fallback: Option<TtsFallbacks>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<Self, Invalid> {
        use lk_agents_abi::inference::InferenceCredentials as _;
        let credentials =
            crate::credentials::Credentials::inference_credentials(base_url, api_key, api_secret)?;
        let (model, parsed_voice) = wire::parse_model_string(&model);
        let entries = fallback.map_or_else(Vec::new, TtsFallbacks::entries);
        let opts = TtsOptions {
            model: model.to_owned(),
            voice: voice.or(parsed_voice.map(str::to_owned)),
            language: language.map(|l| LanguageCode::new(&l).as_str().to_owned()),
            encoding: encoding.unwrap_or_else(|| wire::DEFAULT_ENCODING.to_owned()),
            sample_rate: sample_rate.unwrap_or(wire::DEFAULT_SAMPLE_RATE),
            base_url: credentials.base_url,
            api_key: credentials.api_key,
            api_secret: credentials.api_secret,
            conn_options: conn_options.unwrap_or(lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS),
            extra_kwargs: extra_kwargs.unwrap_or_default(),
            fallback: entries
                .into_iter()
                .map(|entry| match entry {
                    TtsFallbackEntry::Model(model) => model,
                    TtsFallbackEntry::Name(name) => wire::normalize_fallback_str(&name),
                })
                .collect(),
        };
        let tts = GatewayTts::new(
            opts,
            Arc::new(lk_agents_net::websocket::TungsteniteTransport),
            (env.spawner)().map_err(|e| Invalid(e.to_string()))?,
            env.clock.clone(),
        );
        let capabilities = Tts::capabilities(&tts);
        let options = options::OptionsCell::host(sentence_tokenizer, capabilities);
        Ok(GatewayTts { label, events: Some(events), options: Arc::new(options), ..tts })
    }

    fn model(&self) -> String {
        GatewayTts::model(self)
    }
    fn sample_rate(&self) -> u32 {
        GatewayTts::sample_rate(self)
    }
    fn capabilities(&self) -> TtsCapabilities {
        Tts::capabilities(self)
    }
    fn num_channels(&self) -> u16 {
        Tts::num_channels(self)
    }
    fn conn_options(&self) -> ApiConnectOptions {
        self.opts.lock().conn_options
    }
    fn options(&self) -> Arc<options::OptionsCell> {
        self.options.clone()
    }
    fn prewarm(&self, cx: telemetry::Cx) {
        self.prewarm_for(&cx);
    }
    async fn aclose(&self) {
        let idle = self.connections.lock().close();
        for connection in idle {
            connection.sender.close().await;
        }
    }
    fn update_options(
        &self,
        voice: Option<String>,
        model: Option<String>,
        language: Option<String>,
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.update_options(
            model.as_deref(),
            voice.as_deref(),
            language.as_deref(),
            extra_kwargs.as_ref(),
        );
    }
}

#[cfg(test)]
#[path = "tts_tests.rs"]
mod tests;
