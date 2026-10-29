//! Gateway STT adapter. `lifecycle` owns synchronous completion decisions, `machine` builds
//! speech observations, and `wire` validates gateway messages. This module executes socket and
//! media work and owns stream cancellation, bounded queues and pump settlement.

pub mod lifecycle;
pub mod machine;
pub mod options;
pub mod retrying;
pub mod stream_adapter;
pub mod wire;

use crate::abi::{AudioBuffer, AudioFrame, StreamSink, StreamSource, Stt, SttCapabilities};
use crate::headers::{BoundIdentity, HEADER_SESSION_ID, Headers, gateway_headers};
use async_trait::async_trait;
use lk_agents_net::attempts::{AttemptStep, Attempts, RequestPolicy};
use lk_agents_net::transport::{
    TransportError, WsMessage, WsSender, WsTransport, connect_error, open_ws, stream_closed,
    to_ws_url, within,
};

use base64::Engine as _;
use lifecycle::{SttEffect, SttEvent, SttProtocol};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{ApiConnectOptions, Clock, RealClock, SpeechEvent};
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_media::AudioByteStream;
use lk_agents_media::{FrameResampler, OnChange, Quality};
use options::FallbackModel;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use lk_agents_net::env::{Env, Standalone};
use tokio::sync::mpsc;

const WHAT: &str = "LiveKit Inference STT";

/// Maximum settlement time after audio EOF. Silence at expiry is a failure, never clean EOF.
pub(super) const FINALIZATION_TIMEOUT: Duration = Duration::from_secs(30);

/// `_MAX_AGENT_CONTEXT_CHARS`: AssemblyAI caps `agent_context`, so the tail is kept.
const MAX_AGENT_CONTEXT_CHARS: usize = 1750;

/// `STTOptions`, plus the credentials.
#[derive(Debug, Clone)]
pub struct SttOptions {
    pub model: Option<String>,
    pub language: Option<String>,
    pub encoding: String,
    pub sample_rate: u32,
    pub base_url: String,
    pub api_key: String,
    pub api_secret: String,
    pub extra_kwargs: serde_json::Map<String, serde_json::Value>,
    pub fallback: Vec<FallbackModel>,
    /// Not an `Option`: `__init__` substitutes `DEFAULT_API_CONNECT_OPTIONS`. `connect_timeout`
    /// bounds the handshake, `read_timeout` the session acknowledgement.
    pub conn_options: ApiConnectOptions,
}

impl Default for SttOptions {
    fn default() -> Self {
        SttOptions {
            model: None,
            language: None,
            encoding: options::DEFAULT_ENCODING.to_owned(),
            sample_rate: options::DEFAULT_SAMPLE_RATE,
            base_url: String::new(),
            api_key: String::new(),
            api_secret: String::new(),
            extra_kwargs: Default::default(),
            fallback: Vec::new(),
            conn_options: lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS,
        }
    }
}

/// One gateway STT client as one session holds it: the shared client, and the identity of the
/// job that session serves (two sessions sharing an instance bill each its own job). `E` is the
/// hosting process a host-constructed client takes its executor and clock from.
pub struct NativeStt<E: Env = Standalone> {
    shared: Arc<SttShared>,
    identity: Arc<BoundIdentity>,
    conn_options: Option<ApiConnectOptions>,
    /// The host provider this view serves: its `error` event is each failure the core reports.
    reporter: Option<Arc<dyn lk_agents_abi::Stt>>,
    env: PhantomData<fn() -> E>,
}

impl<E: Env> Clone for NativeStt<E> {
    fn clone(&self) -> Self {
        NativeStt {
            shared: Arc::clone(&self.shared),
            identity: Arc::clone(&self.identity),
            conn_options: self.conn_options,
            reporter: self.reporter.clone(),
            env: PhantomData,
        }
    }
}

impl<E: Env> std::ops::Deref for NativeStt<E> {
    type Target = SttShared;

    fn deref(&self) -> &SttShared {
        &self.shared
    }
}

/// What every session's view of one client shares: its options, connections and open streams.
pub struct SttShared {
    opts: Mutex<SttOptions>,
    capabilities: Mutex<SttCapabilities>,
    /// Framework-managed keyterms, merged into `extra` at connect and by `session.update`.
    session_keyterms: Mutex<Vec<String>>,
    keyterms_unsupported_warned: std::sync::atomic::AtomicBool,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    clock: Arc<dyn Clock>,
    /// The Speechmatics path: a client-side VAD whose end-of-speech sends `session.finalize`.
    vad: Mutex<Option<Arc<dyn crate::abi::Vad>>>,
    /// Every open stream, which `update_options` and keyterm changes reach mid-stream.
    live: Mutex<Vec<Weak<StreamControl>>>,
}

impl<E: Env> NativeStt<E> {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { conn_options: conn_options.or(self.conn_options), ..self.clone() }
    }

    pub fn new(opts: SttOptions, ws: Arc<dyn WsTransport>, spawner: Arc<dyn Spawner>) -> Self {
        let capabilities =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        let shared = SttShared {
            opts: Mutex::new(opts),
            capabilities: Mutex::new(capabilities),
            session_keyterms: Mutex::new(Vec::new()),
            keyterms_unsupported_warned: Default::default(),
            ws,
            spawner,
            clock: Arc::new(RealClock::new()),
            vad: Mutex::new(None),
            live: Mutex::new(Vec::new()),
        };
        NativeStt {
            shared: Arc::new(shared),
            identity: Arc::default(),
            conn_options: None,
            reporter: None,
            env: PhantomData,
        }
    }

    /// The same client as a session serving another job holds it: its connections carry that
    /// job's identification.
    /// `reporter` is the host provider whose `error` events this view's failures are.
    pub fn identified(
        &self,
        headers: Headers,
        reporter: Option<Arc<dyn lk_agents_abi::Stt>>,
    ) -> Self {
        let view = NativeStt {
            shared: Arc::clone(&self.shared),
            identity: Arc::default(),
            conn_options: self.conn_options,
            reporter,
            env: PhantomData,
        };
        view.identity.bind(headers);
        view
    }

    /// `get_inference_headers()`: the job's `X-LiveKit-*` identity; each connection gets a fresh
    /// session id.
    pub fn with_identification(self, headers: Headers) -> Self {
        self.identity.bind(headers);
        self
    }

    /// Before the client is shared.
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        Arc::get_mut(&mut self.shared).expect("a client is clocked before it is shared").clock =
            clock;
        self
    }

    pub fn with_vad(self, vad: Arc<dyn crate::abi::Vad>) -> Self {
        self.set_vad(Some(vad));
        self
    }

    /// The model's VAD as the host resolved it (`_resolve_vad_for_model`); later streams use it.
    pub fn set_vad(&self, vad: Option<Arc<dyn crate::abi::Vad>>) {
        *self.vad.lock().unwrap() = vad;
    }

    /// The open streams; a dropped one is pruned on the way.
    fn live_streams(&self) -> Vec<Arc<StreamControl>> {
        let mut live = self.live.lock().unwrap();
        live.retain(|w| w.strong_count() > 0);
        live.iter().filter_map(Weak::upgrade).collect()
    }

    pub fn model(&self) -> String {
        // `"unknown"`, not "": a blank model name in metrics looks like a dropped field.
        self.opts.lock().unwrap().model.clone().unwrap_or_else(|| "unknown".to_owned())
    }

    /// `update_options` on the STT; the live streams get it as a mid-stream `session.update`.
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        let mut opts = self.opts.lock().unwrap();
        if let Some(model) = model {
            let (name, parsed_language) = options::parse_model_string(model);
            opts.model = Some(name.to_owned());
            if let Some(parsed) = parsed_language
                && language.is_none()
            {
                opts.language = Some(
                    lk_agents_abi::tokenize::language::LanguageCode::new(parsed)
                        .as_str()
                        .to_owned(),
                );
            }
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
        // Recompute every capability together; per-branch updates are how `__init__` and
        // `update_options` drifted apart.
        *self.capabilities.lock().unwrap() =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        // A user `extra` re-merges the session keyterms, so the update does not drop them.
        let extra = extra.map(|extra| {
            let mut merged = extra.clone();
            let terms = self.session_keyterms.lock().unwrap().clone();
            let model = opts.model.as_deref().map(|m| options::parse_model_string(m).0);
            if let Some(keyterms) =
                options::keyterms_extra_for_model(model, Some(&opts.extra_kwargs), &terms)
            {
                merged.extend(keyterms);
            }
            merged
        });
        let model = model.map(|m| options::parse_model_string(m).0.to_owned());
        let language = language.and(opts.language.clone());
        drop(opts);
        for stream in self.live_streams() {
            stream.update_options(model.as_deref(), language.as_deref(), extra.as_ref());
        }
    }

    /// `_update_session_keyterms`: a warning once for a model without keyterm prompting, else the
    /// keyterms reach every open stream (now if the user is silent, at end of speech if not).
    pub fn update_keyterms(&self, keyterms: &[String]) {
        if !self.capabilities.lock().unwrap().keyterms {
            if !self.keyterms_unsupported_warned.swap(true, std::sync::atomic::Ordering::SeqCst) {
                log::warn!(
                    target: "livekit.agents",
                    stt = Stt::label(self).as_str();
                    "keyterms are not supported by this STT, ignoring keyterms update"
                );
            }
            return;
        }
        if let Some(extra) = self.update_session_keyterms(keyterms) {
            for stream in self.live_streams() {
                stream.apply_keyterms(extra.clone());
            }
        }
    }

    /// `_push_conversation_item`: an assistant turn's text, its tail, biases recognition as
    /// `agent_context` on a model that carries it.
    pub fn push_conversation_item(&self, role: &str, text: &str) {
        if role != "assistant" || text.is_empty() || !self.capabilities.lock().unwrap().chat_context
        {
            return;
        }
        let tail: String = {
            let chars: Vec<char> = text.chars().collect();
            chars[chars.len().saturating_sub(MAX_AGENT_CONTEXT_CHARS)..].iter().collect()
        };
        let mut extra = serde_json::Map::new();
        extra.insert("agent_context".into(), tail.into());
        self.update_options(None, None, Some(&extra));
    }

    /// The `extra` entry a keyterm change applies, or `None` when unsupported or unchanged.
    pub fn update_session_keyterms(
        &self,
        keyterms: &[String],
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        {
            let existing = self.session_keyterms.lock().unwrap();
            if *existing == keyterms {
                return None;
            }
        }
        let opts = self.opts.lock().unwrap();
        let extra = options::keyterms_extra_for_model(
            opts.model.as_deref(),
            Some(&opts.extra_kwargs),
            keyterms,
        )?;
        drop(opts);
        *self.session_keyterms.lock().unwrap() = keyterms.to_vec();
        Some(extra)
    }

    /// `_sanitize_options`: a per-stream copy, so a later `update_options` on the STT cannot
    /// mutate a stream that is already running.
    fn sanitize(&self, language: Option<&str>) -> SttOptions {
        let mut opts = self.opts.lock().unwrap().clone();
        if let Some(conn) = self.conn_options {
            opts.conn_options = conn;
        }
        if let Some(language) = language {
            opts.language = Some(
                lk_agents_abi::tokenize::language::LanguageCode::new(language).as_str().to_owned(),
            );
        }
        opts
    }

    /// The unboxed pair, for callers that need the live stream's `update_options`.
    pub fn stream_native(
        &self,
        language: Option<&str>,
        start_time_offset: f64,
    ) -> Result<(SttSinkHandle, SttSourceHandle), ProviderError> {
        let opts = self.sanitize(language);
        opts.conn_options
            .validate()
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        let keyterms = self.session_keyterms.lock().unwrap().clone();

        let (audio_tx, audio_rx) = mpsc::channel::<SinkCmd>(lk_agents_abi::AUDIO_QUEUE_FRAMES);
        let (event_tx, event_rx) =
            mpsc::channel::<Result<SpeechEvent, ApiError>>(lk_agents_abi::AUDIO_QUEUE_FRAMES);
        let (update_tx, update_rx) =
            mpsc::channel::<serde_json::Value>(lk_agents_abi::AUDIO_QUEUE_FRAMES);
        let (stop, stop_rx) = tokio::sync::watch::channel(false);
        let (ended, ended_rx) = tokio::sync::watch::channel(false);
        let (failure, failure_rx) = tokio::sync::watch::channel(None);
        let (done, done_rx) = tokio::sync::oneshot::channel();
        let control = Arc::new(StreamControl {
            stop: stop.clone(),
            ended,
            failure,

            updates: update_tx.clone(),
            deferred: Arc::new(DeferredExtra::default()),
        });
        self.live.lock().unwrap().push(Arc::downgrade(&control));

        let pump = Pump {
            ws: self.ws.clone(),
            identification: self.identity.for_connection(),
            clock: self.clock.clone(),
            create_frame: wire::session_create(&opts, &keyterms),
            vad: self.vad.lock().unwrap().clone(),
            offset: start_time_offset,
            opts,
            audio_rx,
            update_rx,
            deferred: control.deferred.clone(),
            update_tx,
            event_tx,
            failure: control.failure.clone(),
            stop: stop_rx,
            ended: ended_rx,
            accepted: self.clock.now(),
        };
        self.spawner.spawn_unboxed(async move {
            pump.run().await;
            let _ = done.send(());
        });

        Ok((
            SttSinkHandle { audio: audio_tx, control },
            SttSourceHandle {
                events: event_rx,
                done: false,
                stop,
                joined: done_rx,
                failure: failure_rx,
                clock: self.clock.clone(),
            },
        ))
    }
}

/// `_core.InferenceStt`: the generated adapter calls this.
impl<E: Env> lk_agents_abi::stt::inference::InferenceStt for NativeStt<E> {
    type Audio = SttSinkHandle;
    type Events = SttSourceHandle;

    fn new(
        model: Option<String>,
        language: Option<String>,
        encoding: String,
        sample_rate: u32,
        base_url: String,
        api_key: String,
        api_secret: String,
        extra_kwargs: serde_json::Map<String, serde_json::Value>,
        fallback: Option<Vec<lk_agents_abi::inference::fallback::SttFallbackModel>>,
        conn_options: ApiConnectOptions,
        vad: Option<Arc<dyn crate::abi::Vad>>,
    ) -> Result<Self, ApiError> {
        let spawner = E::spawner().map_err(|e| ApiError::new(e).with_retryable(false))?;
        // Not `parse_model_into`: the host already parsed `provider/model:language`; a second
        // parse would strip a legitimate `:` from the model name.
        let opts = SttOptions {
            model,
            language,
            encoding,
            sample_rate,
            base_url,
            api_key,
            api_secret,
            extra_kwargs,
            fallback: fallback.unwrap_or_default(),
            conn_options,
        };
        let stt =
            NativeStt::new(opts, Arc::new(lk_agents_net::websocket::TungsteniteTransport), spawner)
                .with_clock(E::clock());
        stt.set_vad(vad);
        Ok(stt)
    }

    fn set_vad(&self, vad: Option<Arc<dyn crate::abi::Vad>>) {
        NativeStt::set_vad(self, vad);
    }

    fn identified(&self, headers: Vec<(String, String)>, reporter: Option<Arc<dyn Stt>>) -> Self {
        NativeStt::identified(self, headers.into_iter().collect(), reporter)
    }

    fn update_options(
        &self,
        model: Option<String>,
        language: Option<String>,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        NativeStt::update_options(self, model.as_deref(), language.as_deref(), extra.as_ref());
    }

    fn update_keyterms(&self, keyterms: Vec<String>) {
        NativeStt::update_keyterms(self, &keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        NativeStt::push_conversation_item(self, role, text);
    }

    fn stream(
        &self,
        language: Option<String>,
        start_time_offset: Option<f64>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<(SttSinkHandle, SttSourceHandle), ProviderError> {
        self.configured(conn_options)
            .stream_native(language.as_deref(), start_time_offset.unwrap_or(0.0))
    }
}

impl<E: Env> Stt for NativeStt<E> {
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    /// The provider's `error` event, on the host provider this view serves.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(reporter) = &self.reporter {
            reporter.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        "livekit.agents.inference.stt.STT".into()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId {
            label: Stt::label(self),
            model: self.model(),
            provider: "livekit".into(),
        }
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        NativeStt::update_keyterms(self, keyterms);
    }

    fn push_conversation_item(&self, role: &str, text: &str) {
        NativeStt::push_conversation_item(self, role, text);
    }

    fn capabilities(&self) -> SttCapabilities {
        *self.capabilities.lock().unwrap()
    }

    /// "LiveKit Inference STT does not support batch recognition, use stream() instead".
    fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&lk_agents_abi::tokenize::language::LanguageCode>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<SpeechEvent, ProviderError>> {
        Box::pin(async move {
            Err(ProviderError::Local(ApiError::new(
                "LiveKit Inference STT does not support batch recognition, use stream() instead",
            )))
        })
    }

    /// Always `Some`: gateway STT declares `streaming=True`.
    fn stream(
        &self,
        language: Option<&lk_agents_abi::tokenize::language::LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<Option<(crate::abi::SttSink, crate::abi::SttSource)>, ProviderError>,
    > {
        let language = language.cloned();
        Box::pin(async move {
            let offset = start_time_offset.unwrap_or(0.0);
            self.stream_native(language.as_ref().map(|l| l.as_str()), offset).map(
                |(sink, source)| {
                    Some((
                        Box::new(sink) as crate::abi::SttSink,
                        Box::new(source) as crate::abi::SttSource,
                    ))
                },
            )
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The two halves
// ---------------------------------------------------------------------------------------------

/// `end_input(&self)` cannot drop the sender, so end-of-input is an explicit message.
#[derive(Debug, Clone, PartialEq)]
enum SinkCmd {
    Frame(AudioFrame),
    /// The `_FlushSentinel`: emit whatever the re-framer is holding, without ending the session.
    Flush,
}

/// A keyterm change that arrived mid-utterance, deferred to `END_OF_SPEECH`: "the gateway may
/// reconnect upstream when the keyterms change". One slot, latest wins.
#[derive(Debug, Default)]
struct DeferredExtra {
    speaking: std::sync::atomic::AtomicBool,
    pending: Mutex<Option<serde_json::Map<String, serde_json::Value>>>,
}

pub struct SttSinkHandle {
    audio: mpsc::Sender<SinkCmd>,
    control: Arc<StreamControl>,
}

impl lk_agents_abi::stt::inference::InferenceSttAudio for SttSinkHandle {
    fn update_options(
        &self,
        model: Option<String>,
        language: Option<String>,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        self.control.update_options(model.as_deref(), language.as_deref(), extra.as_ref());
    }
}

/// What reaches a live stream from its client: `session.update` frames and deferred keyterms.
struct StreamControl {
    stop: tokio::sync::watch::Sender<bool>,
    ended: tokio::sync::watch::Sender<bool>,
    failure: tokio::sync::watch::Sender<Option<ApiError>>,
    updates: mpsc::Sender<serde_json::Value>,
    deferred: Arc<DeferredExtra>,
}

impl StreamControl {
    /// A mid-stream `session.update`; unsupported providers ignore it.
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        if extra.is_some() {
            // An explicit `extra` clears any deferred keyterm change (`_pending_extra = None`).
            *self.deferred.pending.lock().unwrap() = None;
        }
        if let Some(frame) = wire::session_update(model, language, extra)
            && self.updates.try_send(frame).is_err()
        {
            self.fail_overflow();
        }
    }

    fn fail_overflow(&self) {
        self.failure.send_replace(Some(
            ApiError::new("STT input queue capacity exceeded").with_retryable(false),
        ));
        self.stop.send_replace(true);
    }

    /// Now if the user is silent, at end of speech if not.
    pub fn apply_keyterms(&self, extra: serde_json::Map<String, serde_json::Value>) {
        if self.deferred.speaking.load(std::sync::atomic::Ordering::SeqCst) {
            *self.deferred.pending.lock().unwrap() = Some(extra);
            return;
        }
        self.update_options(None, None, Some(&extra));
    }
}

impl StreamSink<AudioFrame> for SttSinkHandle {
    fn push(&self, item: AudioFrame) -> Result<(), ProviderError> {
        self.send(SinkCmd::Frame(item))
    }

    fn flush(&self) -> Result<(), ProviderError> {
        self.send(SinkCmd::Flush)
    }

    fn end_input(&self) {
        // A stream that is already gone has already ended its input.
        self.control.ended.send_replace(true);
    }
}

impl SttSinkHandle {
    fn closed_error(&self) -> ProviderError {
        self.control
            .failure
            .borrow()
            .clone()
            .map(ProviderError::from)
            .unwrap_or_else(|| stream_closed(WHAT))
    }

    fn send(&self, cmd: SinkCmd) -> Result<(), ProviderError> {
        if *self.control.ended.borrow() || *self.control.stop.borrow() {
            return Err(self.closed_error());
        }
        self.audio.try_send(cmd).map_err(|error| {
            if matches!(error, mpsc::error::TrySendError::Full(_)) {
                self.control.fail_overflow();
            }
            self.closed_error()
        })
    }
}

/// The gateway stream's drain half, handed to the host as `InferenceSttStream` beside its sink.
pub struct SttSourceHandle {
    events: mpsc::Receiver<Result<SpeechEvent, ApiError>>,
    done: bool,
    stop: tokio::sync::watch::Sender<bool>,
    joined: tokio::sync::oneshot::Receiver<()>,
    failure: tokio::sync::watch::Receiver<Option<ApiError>>,
    clock: Arc<dyn Clock>,
}

impl StreamSource<SpeechEvent> for SttSourceHandle {
    fn next(&mut self) -> lk_agents_abi::BoxFuture<'_, Result<Option<SpeechEvent>, ProviderError>> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            match self.events.recv().await {
                Some(Ok(event)) => Ok(Some(event)),
                Some(Err(error)) => {
                    self.done = true;
                    Err(error.into())
                }
                None => {
                    self.done = true;
                    match self.failure.borrow().clone() {
                        Some(error) => Err(error.into()),
                        None => Ok(None),
                    }
                }
            }
        })
    }

    fn close(
        mut self: Box<Self>,
        deadline: lk_agents_abi::Ts,
    ) -> lk_agents_abi::BoxFuture<'static, lk_agents_abi::Closed> {
        Box::pin(async move {
            self.done = true;
            self.stop.send_replace(true);
            self.events.close();
            let mut settled = false;
            let mut closed = lk_agents_abi::Closed::bounded(self.clock.as_ref(), deadline, async {
                settled = (&mut self.joined).await.is_ok();
            })
            .await;
            if !settled {
                closed = lk_agents_abi::Closed::TimedOut;
            }
            if closed == lk_agents_abi::Closed::TimedOut {
                log::error!("STT pump cleanup incomplete; pump remains owned until settlement");
            }
            closed
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The pump
// ---------------------------------------------------------------------------------------------

struct Pump {
    ws: Arc<dyn WsTransport>,
    identification: Option<Headers>,
    clock: Arc<dyn Clock>,
    create_frame: serde_json::Value,
    opts: SttOptions,
    vad: Option<Arc<dyn crate::abi::Vad>>,
    /// Seconds added to every timestamp: the caller's, plus the time spent on failed attempts, so
    /// transcripts stay on one linear timeline across reconnects.
    offset: f64,
    audio_rx: mpsc::Receiver<SinkCmd>,
    update_rx: mpsc::Receiver<serde_json::Value>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::Sender<serde_json::Value>,
    event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    failure: tokio::sync::watch::Sender<Option<ApiError>>,
    stop: tokio::sync::watch::Receiver<bool>,
    ended: tokio::sync::watch::Receiver<bool>,
    accepted: lk_agents_abi::Ts,
}

impl Pump {
    async fn failed(&self, error: ApiError) {
        self.failure.send_replace(Some(error.clone()));
        let _ = self.event_tx.send(Err(error)).await;
    }

    /// `RecognizeStream._main_task`: each failed attempt reconnects, under `conn_options`, with the
    /// audio pushed meanwhile still queued. An attempt that outlived the connect limit had
    /// connected, so the budget restarts.
    async fn run(mut self) {
        let mut policy = RequestPolicy::from_options(&self.opts.conn_options);
        // The socket adapter observes phase progress within each attempt. Reopening it has one
        // retry owner and retains the operation's original acceptance deadline.
        policy.connect_timeout = None;
        policy.write_timeout = None;
        policy.read_timeout = None;
        let mut steps = Vec::new();
        let mut attempts = Attempts::start(policy, self.accepted, &mut steps);
        let mut last_start = self.clock.now();
        loop {
            if *self.stop.borrow() {
                attempts.cancel(self.clock.now(), &mut Vec::new());
                return;
            }
            let started = self.clock.now();
            attempts.wake(started, None, &mut steps);
            if let Some(error) = steps.iter().find_map(|step| match step {
                AttemptStep::GiveUp(error) => Some(error.clone()),
                _ => None,
            }) {
                self.failed(error).await;
                return;
            }
            steps.clear();
            let Some(id) = attempts.current() else { return };
            self.offset += started.saturating_duration_since(last_start).as_secs_f64();
            last_start = started;
            let Err(error) = self.attempt().await else {
                attempts.succeeded(id, self.clock.now());
                attempts.closed(id, self.clock.now(), &mut Vec::new());
                return;
            };
            let now = self.clock.now();
            if self
                .opts
                .conn_options
                .attempt_connected(now.saturating_duration_since(started).as_secs_f64(), &error)
            {
                attempts.reset_retries();
            }
            steps.clear();
            attempts.failed_error(id, error.clone(), now, &mut steps);
            attempts.closed(id, now, &mut steps);
            if !steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. })) {
                let error = match lk_agents_net::attempts::terminal_failure(
                    lk_agents_abi::types::retry::RequestKind::SttStream,
                    &error,
                    self.opts.conn_options.max_retry,
                    attempts.retries(),
                ) {
                    lk_agents_abi::types::retry::RetryDecision::Exhausted(message) => {
                        ApiError::connection(Some(format!("{message}: {error}")))
                    }
                    _ => error,
                };
                self.failed(error).await;
                return;
            }
            let at = attempts.wake_at(None).expect("authorized retry wake");
            use futures_util::future::{Either, select};
            if matches!(
                select(Box::pin(self.clock.wait_until(at)), Box::pin(self.stop.changed())).await,
                Either::Right(_)
            ) {
                attempts.cancel(self.clock.now(), &mut Vec::new());
                return;
            }
        }
    }

    /// One connection. `Ok` when the stream ended, the caller gone or the input finished.
    async fn attempt(&mut self) -> Result<(), ApiError> {
        let headers = gateway_headers(
            self.identification
                .as_ref()
                .map(|h| {
                    let mut h = h.clone();
                    h.set(HEADER_SESSION_ID, crate::ids::shortuuid("inference_"));
                    h
                })
                .as_ref(),
            None,
            None,
            &self.opts.api_key,
            &self.opts.api_secret,
        )?;
        // The reference reads the session id back out of the header map (a job's
        // `inference_headers` cannot override it) and uses it as the request id.
        let request_id = headers
            .get(HEADER_SESSION_ID)
            .map(str::to_owned)
            .unwrap_or_else(|| crate::ids::shortuuid("stt_request_"));
        let url = format!(
            "{}/stt?model={}",
            to_ws_url(&self.opts.base_url),
            self.opts.model.as_deref().unwrap_or("")
        );
        // `_speech_duration`: written by send, read by recv.
        let machine = Arc::new(Mutex::new(SttProtocol::new(
            0,
            lk_agents_abi::resources::ResourceId(0),
            request_id,
            self.opts.language.as_deref(),
            FINALIZATION_TIMEOUT,
            self.opts
                .conn_options
                .total_timeout
                .map(|seconds| self.accepted.saturating_add(Duration::from_secs_f64(seconds))),
        )));
        let opening = machine.lock().unwrap().handle(SttEvent::Start, self.clock.now());
        if !opening.iter().any(|effect| matches!(effect, SttEffect::SttStreamOpen { .. })) {
            return opening
                .into_iter()
                .find_map(|effect| match effect {
                    SttEffect::Finished(result) => Some(result),
                    _ => None,
                })
                .unwrap_or(Ok(()));
        }
        let (sender, mut receiver) = open_ws(
            self.ws.as_ref(),
            &url,
            headers.into_vec(),
            &*self.clock,
            &self.opts.conn_options,
            WHAT,
            &self.create_frame,
        )
        .await?;

        // Built before the loops so the send half can push into it while the finalize half drains it.
        let vad_stream = match self.vad.as_ref() {
            Some(vad) => vad.stream().await.ok(),
            None => None,
        };
        let (vad_sink, mut vad_source) = match vad_stream {
            Some((sink, source)) => (Some(sink), Some(source)),
            None => (None, None),
        };

        let send = Self::send_loop(
            sender.clone(),
            &mut self.audio_rx,
            &mut self.update_rx,
            &mut self.ended,
            vad_sink,
            self.opts.sample_rate,
            machine.clone(),
            self.clock.clone(),
        );
        let recv = Self::recv_loop(
            receiver.as_mut(),
            machine.clone(),
            RecvCtx {
                event_tx: self.event_tx.clone(),
                deferred: self.deferred.clone(),
                update_tx: self.update_tx.clone(),
                clock: self.clock.clone(),
                ack_deadline: self.opts.conn_options.read_deadline(self.clock.now()),
                offset: self.offset,
            },
        );
        let finalize = Self::vad_finalize_loop(sender.clone(), &mut vad_source);

        let deadline = machine.lock().unwrap().wake_at();
        let total_timer = async {
            match deadline {
                Some(deadline) => self.clock.wait_until(deadline).await,
                None => std::future::pending().await,
            }
        };
        let outcome = tokio::select! {
            _ = self.stop.wait_for(|stop| *stop) => Ok(()),
            _ = total_timer => {
                machine.lock().unwrap().handle(SttEvent::Wake, self.clock.now()).into_iter().find_map(|effect| match effect {
                    SttEffect::Finished(result) => Some(result), _ => None
                }).unwrap_or_else(|| Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Total).with_retryable(false)))
            },
            r = send => r,
            r = recv => r,
            r = finalize => r,
        };
        sender.close().await;
        if let Some(source) = vad_source {
            let deadline = self.clock.now().saturating_add(lk_agents_abi::HOST_TEARDOWN_TIMEOUT);
            if source.close(deadline).await == lk_agents_abi::Closed::TimedOut {
                log::error!("STT VAD cleanup incomplete");
            }
        }
        outcome.map_err(|error| {
            if machine.lock().unwrap().replay_safe() { error } else { error.with_retryable(false) }
        })
    }

    async fn send_loop(
        sender: Arc<dyn WsSender>,
        audio_rx: &mut mpsc::Receiver<SinkCmd>,
        update_rx: &mut mpsc::Receiver<serde_json::Value>,
        ended: &mut tokio::sync::watch::Receiver<bool>,
        mut vad_sink: Option<crate::abi::VadSink>,
        sample_rate: u32,
        machine: Arc<Mutex<SttProtocol>>,
        clock: Arc<dyn Clock>,
    ) -> Result<(), ApiError> {
        // 50 ms frames: `samples_per_channel=self._opts.sample_rate // 20`.
        let mut bstream = AudioByteStream::new(sample_rate, 1, Some(sample_rate / 20), false);
        // `RecognizeStream.push_frame` resamples to the rate the gateway was told.
        let mut resampler = FrameResampler::new(sample_rate, Quality::High, OnChange::Drain);

        if *ended.borrow() {
            audio_rx.close();
        }
        let mut ending = false;
        let mut vad_push_warned = false;
        let mut audio = String::new();
        loop {
            let item = tokio::select! {
                // Biased so a pending `session.update` is not starved by continuous audio.
                biased;
                changed = ended.changed(), if !audio_rx.is_closed() => {
                    if changed.is_err() || *ended.borrow() { audio_rx.close(); }
                    continue;
                }
                Some(frame) = update_rx.recv() => {
                    // "failed to send session.update, ws may be closing" -- swallowed, as in the
                    // reference.
                    if !sender.is_closed() {
                        let _ = sender.send_text(frame.to_string()).await;
                    }
                    continue;
                }
                item = audio_rx.recv() => item,
            };

            let frames = match item {
                Some(SinkCmd::Frame(frame)) => {
                    if let Some(sink) = vad_sink.as_mut()
                        && let Err(error) = sink.push(frame.clone())
                        && !vad_push_warned
                    {
                        vad_push_warned = true;
                        log::warn!(
                            target: "livekit.agents",
                            "STT VAD refused audio: {error}; suppressing repeats"
                        );
                    }
                    let samples = usize::try_from(frame.sample_rate / 20).unwrap_or(0).max(1);
                    let stride = usize::from(frame.num_channels) * 2;
                    if stride == 0 || frame.sample_rate == 0 || frame.data.len() % stride != 0 {
                        return Err(ApiError::new("Invalid STT audio frame").with_retryable(false));
                    }
                    let mut frames = Vec::new();
                    for bytes in frame.data.chunks(samples * stride) {
                        let chunk = AudioFrame::new(
                            bytes,
                            frame.sample_rate,
                            frame.num_channels,
                            (bytes.len() / stride) as u32,
                        );
                        let resampled = resampler.push(&chunk).map_err(|error| {
                            ApiError::new(error.to_string()).with_retryable(false)
                        })?;
                        frames.extend(resampled.iter().flat_map(|frame| bstream.push(&frame.data)));
                        tokio::task::yield_now().await;
                    }
                    frames
                }
                Some(SinkCmd::Flush) => {
                    let tail = resampler
                        .flush()
                        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
                    let mut frames: Vec<_> =
                        tail.iter().flat_map(|f| bstream.push(&f.data)).collect();
                    frames.extend(bstream.flush());
                    frames
                }
                // `end_input()` or a dropped sink: both end the send half.
                None => {
                    ending = true;
                    let tail = resampler
                        .flush()
                        .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
                    let mut frames: Vec<_> =
                        tail.iter().flat_map(|frame| bstream.push(&frame.data)).collect();
                    frames.extend(bstream.flush());
                    frames
                }
            };

            for frame in frames {
                machine.lock().unwrap().handle(
                    SttEvent::AudioWritten(
                        f64::from(frame.samples_per_channel) / f64::from(sample_rate),
                    ),
                    clock.now(),
                );
                audio.clear();
                base64::engine::general_purpose::STANDARD.encode_string(&frame.data, &mut audio);
                if let Err(err) = sender.send_text(wire::input_audio(&audio).to_string()).await {
                    return Err(match err {
                        TransportError::Closed => ApiError::connection(Some(format!(
                            "{WHAT} connection closed unexpectedly"
                        ))),
                        other => ApiError::connection(Some(other.to_string())),
                    });
                }
            }
            if ending {
                break;
            }
        }

        if let Some(sink) = vad_sink.as_mut() {
            sink.end_input();
        }
        let effects = machine.lock().unwrap().handle(SttEvent::InputEof, clock.now());
        if effects.iter().any(|effect| matches!(effect, SttEffect::SttFinalize { .. })) {
            sender
                .send_text(wire::session_finalize().to_string())
                .await
                .map_err(|error| connect_error(WHAT, error))?;
        }
        let at = machine.lock().unwrap().wake_at().expect("input EOF arms settlement");
        clock.wait_until(at).await;
        let effects = machine.lock().unwrap().handle(SttEvent::Wake, clock.now());
        if let Some(result) = effects.into_iter().find_map(|effect| match effect {
            SttEffect::Finished(result) => Some(result),
            _ => None,
        }) {
            return result;
        }
        // Completion may already be observed while its transcript is blocked on delivery.
        std::future::pending().await
    }

    async fn recv_loop(
        receiver: &mut dyn lk_agents_net::transport::WsReceiver,
        machine: Arc<Mutex<SttProtocol>>,
        ctx: RecvCtx,
    ) -> Result<(), ApiError> {
        let RecvCtx { event_tx, deferred, update_tx, clock, ack_deadline, offset } = ctx;
        // Only the first read (the `session.create` ack) is a response the session waits for;
        // later silence is the user's.
        let mut awaiting_first = true;
        loop {
            let recv = receiver.recv();
            let received =
                if awaiting_first { within(&*clock, ack_deadline, recv).await } else { recv.await };
            let msg = match received {
                Ok(msg) => msg,
                Err(TransportError::Timeout) if awaiting_first => {
                    return Err(ApiError::timed_out(lk_agents_abi::errors::TimeoutPhase::Read));
                }
                Err(err) => return Err(connect_error(WHAT, err)),
            };
            match msg {
                WsMessage::Closed => {
                    let effects = machine.lock().unwrap().handle(SttEvent::ServerEof, clock.now());
                    return effects
                        .into_iter()
                        .find_map(|effect| match effect {
                            SttEffect::Finished(result) => Some(result),
                            _ => None,
                        })
                        .unwrap_or(Ok(()));
                }
                // "unexpected LiveKit Inference STT message type" -- logged and skipped.
                WsMessage::Binary(_) | WsMessage::Other => continue,
                WsMessage::Text(text) => {
                    let incoming = wire::decode(&text).ok_or_else(|| {
                        ApiError::new("Malformed STT gateway message").with_retryable(false)
                    })?;
                    if matches!(incoming, wire::Incoming::SessionCreated) {
                        awaiting_first = false;
                    }
                    let effects = {
                        let mut machine = machine.lock().unwrap();
                        let effects = machine.handle(SttEvent::Message(incoming), clock.now());
                        deferred
                            .speaking
                            .store(machine.is_speaking(), std::sync::atomic::Ordering::SeqCst);
                        effects
                    };
                    for effect in effects {
                        match effect {
                            SttEffect::Deliver(event) => {
                                if event_tx.send(Ok(shifted(event, offset))).await.is_err() {
                                    return Ok(());
                                }
                            }
                            SttEffect::EndOfSpeech => {
                                let extra = deferred.pending.lock().unwrap().take();
                                if let Some(extra) = extra
                                    && let Some(frame) =
                                        wire::session_update(None, None, Some(&extra))
                                {
                                    update_tx.try_send(frame).map_err(|_| {
                                        ApiError::new("STT update queue capacity exceeded")
                                            .with_retryable(false)
                                    })?;
                                }
                            }
                            SttEffect::Finished(result) => return result,
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// `vad_task`: every end of speech from the client-side VAD sends `session.finalize`.
    async fn vad_finalize_loop(
        sender: Arc<dyn WsSender>,
        source: &mut Option<crate::abi::VadSource>,
    ) -> Result<(), ApiError> {
        let Some(source) = source.as_mut() else {
            // No VAD: this arm must never resolve, or `select!` tears the socket down at once.
            std::future::pending::<()>().await;
            return Ok(());
        };
        while let Ok(Some(event)) = source.next().await {
            if event.kind != lk_agents_abi::VadEventType::EndOfSpeech {
                continue;
            }
            if sender.is_closed() {
                return Ok(());
            }
            if sender.send_text(wire::session_finalize().to_string()).await.is_err() {
                return Ok(());
            }
        }
        std::future::pending::<()>().await;
        Ok(())
    }
}

struct RecvCtx {
    event_tx: mpsc::Sender<Result<SpeechEvent, ApiError>>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::Sender<serde_json::Value>,
    clock: Arc<dyn Clock>,
    /// `read_timeout` from the session-create frame: the acknowledgement is a response.
    ack_deadline: Option<lk_agents_abi::Ts>,
    offset: f64,
}

/// The event on the caller's timeline: every alternative's and word's times moved by `offset`.
fn shifted(mut event: SpeechEvent, offset: f64) -> SpeechEvent {
    for alt in &mut event.alternatives {
        alt.start_time += offset;
        alt.end_time += offset;
        for word in alt.words.iter_mut().flatten() {
            word.start_time = Some(word.start_time.unwrap_or(0.0) + offset);
            word.end_time = Some(word.end_time.unwrap_or(0.0) + offset);
            word.start_time_offset = Some(offset);
        }
    }
    event
}

#[cfg(test)]
mod tests;
