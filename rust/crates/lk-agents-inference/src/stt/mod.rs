//! `inference/stt.py` -- the gateway STT, WebSocket + JSON (§1.4). `SpeechStream._run`'s three
//! tasks (send, recv and the client-side VAD finalize) are independent loops; `select!` only
//! decides whose termination tears the socket down.

pub mod machine;
pub mod options;
#[cfg(feature = "python")]
mod python;
pub mod wire;

use crate::abi::{AudioBuffer, AudioFrame, StreamSink, StreamSource, Stt, SttCapabilities};
use crate::headers::{BoundIdentity, HEADER_SESSION_ID, Headers, gateway_headers};
use crate::transport::{
    TransportError, WsMessage, WsSender, WsTransport, connect_error, next_terminal, open_ws,
    stream_closed, to_ws_url, within,
};
use async_trait::async_trait;
use base64::Engine as _;
use lk_agents_abi::{Spawner, SpawnerExt};
use lk_agents_abi::AudioByteStream;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{ApiConnectOptions, Clock, RealClock, SpeechEvent};
use machine::TranscriptMachine;
use options::FallbackModel;
use lk_agents_abi::types::retry::{RequestKind, RetryDecision, decide};
use lk_agents_media::{FrameResampler, OnChange, Quality};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::mpsc;

const WHAT: &str = "LiveKit Inference STT";

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
    /// Not an `Option`: `__init__` substitutes `DEFAULT_API_CONNECT_OPTIONS`; its `timeout` also
    /// bounds the handshake and the session acknowledgement.
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

pub struct NativeStt {
    opts: Mutex<SttOptions>,
    capabilities: Mutex<SttCapabilities>,
    /// Framework-managed keyterms, merged into `extra` at connect and by `session.update`.
    session_keyterms: Mutex<Vec<String>>,
    keyterms_unsupported_warned: std::sync::atomic::AtomicBool,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    identity: BoundIdentity,
    clock: Arc<dyn Clock>,
    /// The Speechmatics path: a client-side VAD whose end-of-speech sends `session.finalize`.
    vad: Mutex<Option<Arc<dyn crate::abi::Vad>>>,
    /// Every open stream, which `update_options` and keyterm changes reach mid-stream.
    live: Mutex<Vec<Weak<StreamControl>>>,
}

/// `inference/stt.py`'s WebSocket, over `lk-agents-inference::stt::NativeStt`.
///
/// Options, capability rules and model parsing stay host-side; the socket, frame pump, transcript
/// machine and reconnect cross. The stream is a sink/source pair driven by different host tasks:
/// `push` is a synchronous send, `recv` the only awaiting method. The keyword constructor is
/// `python::InferenceStt::new`.
#[lk_agents_abi::host(object(
    name = "InferenceStt",
    cell = "arc",
    rust_name = InferenceStt,
    as_dyn(Stt = as_stt),
    stub = "def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
    stub = "def set_vad(self, vad: Any | None = None) -> None: ...",
    stub = "def bind_identification(self, headers: list[tuple[str, str]]) -> None: ..."
))]
impl NativeStt {
    pub fn new(opts: SttOptions, ws: Arc<dyn WsTransport>, spawner: Arc<dyn Spawner>) -> Self {
        let capabilities =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        NativeStt {
            opts: Mutex::new(opts),
            capabilities: Mutex::new(capabilities),
            session_keyterms: Mutex::new(Vec::new()),
            keyterms_unsupported_warned: Default::default(),
            ws,
            spawner,
            identity: BoundIdentity::default(),
            clock: Arc::new(RealClock::new()),
            vad: Mutex::new(None),
            live: Mutex::new(Vec::new()),
        }
    }

    /// `get_inference_headers()`: the job's `X-LiveKit-*` identity; each connection gets a fresh
    /// session id.
    pub fn with_identification(self, headers: Headers) -> Self {
        self.identity.bind(headers);
        self
    }

    /// The identity of the job the client now serves, bound by the host that hands it to a session.
    pub fn bind_identification(&self, headers: Headers) {
        self.identity.bind(headers);
    }

    /// The client as the provider a session holds directly.
    pub fn as_stt(self: &Arc<Self>) -> Arc<dyn Stt> {
        Arc::clone(self) as Arc<dyn Stt>
    }

    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
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

    /// `update_options` on the STT. Live streams are updated through [`SttSinkHandle::update_options`].
    #[host(method(kw_only))]
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&crate::extra::ExtraKwargs>,
    ) {
        let mut opts = self.opts.lock().unwrap();
        if let Some(model) = model {
            let (name, parsed_language) = options::parse_model_string(model);
            opts.model = Some(name.to_owned());
            if let Some(parsed) = parsed_language
                && language.is_none()
            {
                opts.language =
                    Some(lk_agents_abi::tokenize::language::LanguageCode::new(parsed).as_str().to_owned());
            }
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
        // Recompute every capability together; per-branch updates are how `__init__` and
        // `update_options` drifted apart.
        *self.capabilities.lock().unwrap() =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        // A user `extra` re-merges the session keyterms, so the update does not drop them.
        let extra = extra.map(|extra| {
            let mut merged = extra.0.clone();
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
    #[host(method(name = "update_keyterms"))]
    pub fn update_keyterms_for_host(&self, keyterms: Vec<String>) {
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
        if let Some(extra) = self.update_session_keyterms(&keyterms) {
            for stream in self.live_streams() {
                stream.apply_keyterms(extra.clone());
            }
        }
    }

    /// `_push_conversation_item`: an assistant turn's text, its tail, biases recognition as
    /// `agent_context` on a model that carries it.
    #[host(method)]
    pub fn push_conversation_item(&self, role: &str, text: &str) {
        if role != "assistant" || text.is_empty() || !self.capabilities.lock().unwrap().chat_context {
            return;
        }
        let tail: String = {
            let chars: Vec<char> = text.chars().collect();
            chars[chars.len().saturating_sub(MAX_AGENT_CONTEXT_CHARS)..].iter().collect()
        };
        let mut extra = serde_json::Map::new();
        extra.insert("agent_context".into(), tail.into());
        self.update_options(None, None, Some(&crate::extra::ExtraKwargs(extra)));
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
        if let Some(language) = language {
            opts.language =
                Some(lk_agents_abi::tokenize::language::LanguageCode::new(language).as_str().to_owned());
        }
        opts
    }

    /// Open a stream. Synchronous: the pump task opens the socket. `start_time_offset` shifts its
    /// timestamps onto the caller's input timeline.
    #[host(method(name = "stream", stream = InferenceSttStream, kw_only))]
    pub fn stream_for_host(&self, language: Option<&str>, start_time_offset: Option<f64>) -> Result<(SttSinkHandle, SttSourceHandle), ProviderError> {
        self.stream_native(language, start_time_offset.unwrap_or(0.0))
    }

    /// The unboxed pair, for callers that need the live stream's `update_options`.
    pub fn stream_native(
        &self,
        language: Option<&str>,
        start_time_offset: f64,
    ) -> Result<(SttSinkHandle, SttSourceHandle), ProviderError> {
        let opts = self.sanitize(language);
        let keyterms = self.session_keyterms.lock().unwrap().clone();

        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<SinkCmd>();
        let (event_tx, event_rx) = mpsc::unbounded_channel::<Result<SpeechEvent, ApiError>>();
        let (update_tx, update_rx) = mpsc::unbounded_channel::<serde_json::Value>();
        let control = Arc::new(StreamControl {
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
        };
        self.spawner.spawn_unboxed(pump.run());

        Ok((
            SttSinkHandle { audio: audio_tx, control },
            SttSourceHandle { events: event_rx, done: false },
        ))
    }
}

#[async_trait]
impl Stt for NativeStt {
    fn label(&self) -> String {
        "livekit.agents.inference.stt.STT".into()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId { label: Stt::label(self), model: self.model(), provider: "livekit".into() }
    }

    fn update_keyterms(&self, keyterms: &[String]) {
        self.update_keyterms_for_host(keyterms.to_vec());
    }

    fn capabilities(&self) -> SttCapabilities {
        *self.capabilities.lock().unwrap()
    }

    /// "LiveKit Inference STT does not support batch recognition, use stream() instead".
    async fn recognize(
        &self,
        _buffer: AudioBuffer<'_>,
        _language: Option<&lk_agents_abi::tokenize::language::LanguageCode>,
    ) -> Result<SpeechEvent, ProviderError> {
        Err(ProviderError::Local(ApiError::new(
            "LiveKit Inference STT does not support batch recognition, use stream() instead",
        )))
    }

    /// Always `Some`: gateway STT declares `streaming=True`.
    async fn stream(
        &self,
        language: Option<&lk_agents_abi::tokenize::language::LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> Option<Result<(crate::abi::SttSink, crate::abi::SttSource), ProviderError>> {
        let offset = start_time_offset.unwrap_or(0.0);
        Some(self.stream_native(language.map(|l| l.as_str()), offset).map(|(sink, source)| {
            (Box::new(sink) as crate::abi::SttSink, Box::new(source) as crate::abi::SttSource)
        }))
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
    End,
}

/// A keyterm change that arrived mid-utterance, deferred to `END_OF_SPEECH`: "the gateway may
/// reconnect upstream when the keyterms change". One slot, latest wins.
#[derive(Debug, Default)]
struct DeferredExtra {
    speaking: std::sync::atomic::AtomicBool,
    pending: Mutex<Option<serde_json::Map<String, serde_json::Value>>>,
}

pub struct SttSinkHandle {
    audio: mpsc::UnboundedSender<SinkCmd>,
    control: Arc<StreamControl>,
}

impl SttSinkHandle {
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        self.control.update_options(model, language, extra);
    }

    pub fn apply_keyterms(&self, extra: serde_json::Map<String, serde_json::Value>) {
        self.control.apply_keyterms(extra);
    }
}

/// What reaches a live stream from its client: `session.update` frames and deferred keyterms.
struct StreamControl {
    updates: mpsc::UnboundedSender<serde_json::Value>,
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
        if let Some(frame) = wire::session_update(model, language, extra) {
            let _ = self.updates.send(frame);
        }
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
        let _ = self.audio.send(SinkCmd::End);
    }
}

impl SttSinkHandle {
    fn send(&self, cmd: SinkCmd) -> Result<(), ProviderError> {
        self.audio.send(cmd).map_err(|_| stream_closed(WHAT))
    }
}

/// The gateway stream's drain half, handed to the host as `InferenceSttStream` beside its sink.
#[lk_agents_abi::host(stream(name = "InferenceSttStream", doc = "One live recognition session.", sink = crate::stt::SttSinkHandle,
    input = AudioFrame, input_py = "rtc.AudioFrame",
    stub_import = "from livekit import rtc", end_input, close_ends_input,
    output = SpeechEvent,
    stub = "def apply_keyterms(self, extra: dict[str, Any]) -> None: ...",
    stub = "def update_options(\n    self,\n    *,\n    model: str | None = None,\n    language: str | None = None,\n    extra: dict[str, Any] | None = None,\n) -> None: ..."))]
pub struct SttSourceHandle {
    events: mpsc::UnboundedReceiver<Result<SpeechEvent, ApiError>>,
    done: bool,
}

#[async_trait]
impl StreamSource<SpeechEvent> for SttSourceHandle {
    async fn next(&mut self) -> Result<Option<SpeechEvent>, ProviderError> {
        next_terminal(&mut self.events, &mut self.done).await
    }

    /// Nothing here awaits the network, so the deadline is not consulted.
    async fn close(mut self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
        self.done = true;
        self.events.close();
        lk_agents_abi::Closed::Clean
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
    audio_rx: mpsc::UnboundedReceiver<SinkCmd>,
    update_rx: mpsc::UnboundedReceiver<serde_json::Value>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::UnboundedSender<serde_json::Value>,
    event_tx: mpsc::UnboundedSender<Result<SpeechEvent, ApiError>>,
}

impl Pump {
    /// `RecognizeStream._main_task`: each failed attempt reconnects, under `conn_options`, with the
    /// audio pushed meanwhile still queued. An attempt that outlived the connect timeout had
    /// connected, so the budget restarts.
    async fn run(mut self) {
        let timeout = Duration::from_secs_f64(self.opts.conn_options.timeout.max(0.0));
        let mut retries = 0u32;
        let mut last_start = self.clock.now();
        loop {
            let started = self.clock.now();
            self.offset += started.saturating_duration_since(last_start).as_secs_f64();
            last_start = started;
            let Err(err) = self.attempt(timeout).await else { return };
            if self.clock.now().saturating_duration_since(started) > timeout {
                retries = 0;
            }
            let decision = decide(
                RequestKind::SttStream,
                &self.opts.conn_options,
                retries,
                err.retryable,
                err.status_code,
                false,
            );
            match decision {
                RetryDecision::Retry(delay) => {
                    log::warn!(
                        target: "livekit.agents",
                        stt = "livekit.agents.inference.stt.STT", attempt = retries;
                        "failed to recognize speech: {err}, retrying in {delay}s"
                    );
                    let at = self.clock.now().saturating_add(Duration::from_secs_f64(delay.max(0.0)));
                    self.clock.wait_until(at).await;
                    retries += 1;
                }
                RetryDecision::Exhausted(message) => {
                    let _ = self.event_tx.send(Err(ApiError::connection(Some(message))));
                    return;
                }
                RetryDecision::Stop | RetryDecision::Raise => {
                    let _ = self.event_tx.send(Err(err));
                    return;
                }
            }
        }
    }

    /// One connection. `Ok` when the stream ended, the caller gone or the input finished.
    async fn attempt(&mut self, timeout: Duration) -> Result<(), ApiError> {
        let headers = gateway_headers(
            self.identification.as_ref().map(|h| {
                let mut h = h.clone();
                h.set(HEADER_SESSION_ID, crate::ids::shortuuid("inference_"));
                h
            }).as_ref(),
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
        let (sender, mut receiver) = open_ws(
            self.ws.as_ref(),
            &url,
            headers.into_vec(),
            &*self.clock,
            timeout,
            WHAT,
            &self.create_frame,
        )
        .await?;

        // Built before the loops so the send half can push into it while the finalize half drains it.
        let vad_stream = match self.vad.as_ref() {
            Some(vad) => vad.stream().await.ok(),
            None => None,
        };
        let (vad_sink, vad_source) = match vad_stream {
            Some((sink, source)) => (Some(sink), Some(source)),
            None => (None, None),
        };

        // `_speech_duration`: written by send, read by recv.
        let machine = Arc::new(Mutex::new(TranscriptMachine::new(
            request_id,
            self.opts.language.as_deref(),
        )));
        let closing = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let send = Self::send_loop(
            sender.clone(),
            &mut self.audio_rx,
            &mut self.update_rx,
            vad_sink,
            self.opts.sample_rate,
            machine.clone(),
            closing.clone(),
        );
        let recv = Self::recv_loop(
            receiver.as_mut(),
            machine,
            RecvCtx {
                event_tx: self.event_tx.clone(),
                closing: closing.clone(),
                deferred: self.deferred.clone(),
                update_tx: self.update_tx.clone(),
                clock: self.clock.clone(),
                first_message_timeout: timeout,
                offset: self.offset,
            },
        );
        let finalize = Self::vad_finalize_loop(sender.clone(), vad_source);

        // The reference gathers the tasks and cancels the rest when one finishes.
        let outcome = tokio::select! {
            r = send => r,
            r = recv => r,
            r = finalize => r,
        };
        sender.close().await;
        outcome
    }

    async fn send_loop(
        sender: Arc<dyn WsSender>,
        audio_rx: &mut mpsc::UnboundedReceiver<SinkCmd>,
        update_rx: &mut mpsc::UnboundedReceiver<serde_json::Value>,
        mut vad_sink: Option<crate::abi::VadSink>,
        sample_rate: u32,
        machine: Arc<Mutex<TranscriptMachine>>,
        closing: Arc<std::sync::atomic::AtomicBool>,
    ) -> Result<(), ApiError> {
        // 50 ms frames: `samples_per_channel=self._opts.sample_rate // 20`.
        let mut bstream = AudioByteStream::new(sample_rate, 1, Some(sample_rate / 20), false);
        // `RecognizeStream.push_frame` resamples to the rate the gateway was told.
        let mut resampler = FrameResampler::new(sample_rate, Quality::High, OnChange::Drain);

        loop {
            let item = tokio::select! {
                // Biased so a pending `session.update` is not starved by continuous audio.
                biased;
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
                    if let Some(sink) = vad_sink.as_mut() {
                        let _ = sink.push(frame.clone());
                    }
                    let resampled = resampler
                        .push(&frame)
                        .map_err(|e| ApiError::new(format!("{WHAT}: {e}")).with_retryable(false))?;
                    resampled.iter().flat_map(|f| bstream.push(&f.data)).collect()
                }
                Some(SinkCmd::Flush) => {
                    let tail = resampler.flush().ok().flatten();
                    let mut frames: Vec<_> = tail.iter().flat_map(|f| bstream.push(&f.data)).collect();
                    frames.extend(bstream.flush());
                    frames
                }
                // `end_input()` or a dropped sink: both end the send half.
                Some(SinkCmd::End) | None => break,
            };

            for frame in frames {
                machine
                    .lock()
                    .unwrap()
                    .record_audio(f64::from(frame.samples_per_channel) / f64::from(sample_rate));
                let audio = base64::engine::general_purpose::STANDARD.encode(&frame.data);
                if let Err(err) = sender.send_text(wire::input_audio(&audio).to_string()).await {
                    // `if closing_ws or http_session.closed: return` -- a send failure during
                    // teardown is normal.
                    if closing.load(std::sync::atomic::Ordering::SeqCst) {
                        return Ok(());
                    }
                    return Err(match err {
                        TransportError::Closed => ApiError::connection(Some(format!(
                            "{WHAT} connection closed unexpectedly"
                        ))),
                        other => ApiError::connection(Some(other.to_string())),
                    });
                }
            }
        }

        if let Some(sink) = vad_sink.as_mut() {
            sink.end_input();
        }
        closing.store(true, std::sync::atomic::Ordering::SeqCst);
        let _ = sender.send_text(wire::session_finalize().to_string()).await;
        // The gateway still owes transcripts for audio already sent; park until recv is done.
        std::future::pending::<()>().await;
        Ok(())
    }

    async fn recv_loop(
        receiver: &mut dyn crate::transport::WsReceiver,
        machine: Arc<Mutex<TranscriptMachine>>,
        ctx: RecvCtx,
    ) -> Result<(), ApiError> {
        let RecvCtx { event_tx, closing, deferred, update_tx, clock, first_message_timeout, offset } =
            ctx;
        // Only the first read (the `session.create` ack) is bounded; later silence is the user's.
        let mut awaiting_first = true;
        loop {
            let recv = receiver.recv();
            let received = if awaiting_first {
                awaiting_first = false;
                within(&*clock, first_message_timeout, recv).await
            } else {
                recv.await
            };
            let msg = match received {
                Ok(msg) => msg,
                Err(err) => return Err(connect_error(WHAT, err)),
            };
            match msg {
                WsMessage::Closed => {
                    // The reference distinguishes "we asked for this" from "the gateway hung up".
                    if closing.load(std::sync::atomic::Ordering::SeqCst) {
                        return Ok(());
                    }
                    return Err(ApiError::status(
                        format!("{WHAT} connection closed unexpectedly"),
                        -1,
                        None,
                        None,
                    ));
                }
                // "unexpected LiveKit Inference STT message type" -- logged and skipped.
                WsMessage::Other => continue,
                WsMessage::Text(text) => {
                    let Some(incoming) = wire::decode(&text) else { continue };
                    let processed = {
                        let mut machine = machine.lock().unwrap();
                        let processed = machine.process(incoming);
                        deferred
                            .speaking
                            .store(machine.is_speaking(), std::sync::atomic::Ordering::SeqCst);
                        processed
                    };
                    // `_on_end_of_speech`: release a keyterm change that was parked mid-utterance.
                    if processed.end_of_speech
                        && let Some(extra) = deferred.pending.lock().unwrap().take()
                        && let Some(frame) = wire::session_update(None, None, Some(&extra))
                    {
                        let _ = update_tx.send(frame);
                    }
                    for event in processed.events {
                        if event_tx.send(Ok(shifted(event, offset))).is_err() {
                            return Ok(());
                        }
                    }
                    if let Some(err) = processed.error {
                        return Err(err);
                    }
                }
            }
        }
    }

    /// `vad_task`: every end of speech from the client-side VAD sends `session.finalize`.
    async fn vad_finalize_loop(
        sender: Arc<dyn WsSender>,
        source: Option<crate::abi::VadSource>,
    ) -> Result<(), ApiError> {
        let Some(mut source) = source else {
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
    event_tx: mpsc::UnboundedSender<Result<SpeechEvent, ApiError>>,
    closing: Arc<std::sync::atomic::AtomicBool>,
    deferred: Arc<DeferredExtra>,
    update_tx: mpsc::UnboundedSender<serde_json::Value>,
    clock: Arc<dyn Clock>,
    first_message_timeout: Duration,
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
mod tests {
    use super::*;
    use crate::testkit::{FakeWs, text};
    use lk_agents_abi::TokioSpawner;
    use lk_agents_abi::SpeechEventType;
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

    fn stt(ws: Arc<FakeWs>) -> NativeStt {
        let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
        NativeStt::new(opts(), ws, spawner)
    }

    /// A client whose streams do not retry, for tests about one connection's failure.
    fn stt_once(ws: Arc<FakeWs>) -> NativeStt {
        let mut opts = opts();
        opts.conn_options.max_retry = 0;
        let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
        NativeStt::new(opts, ws, spawner)
    }

    // Catches: a failed connection ending the stream instead of reconnecting under
    // `conn_options`, or the exhausted error not naming the attempts.
    #[tokio::test]
    async fn a_failed_connection_is_retried_then_reported_with_its_attempts() {
        let ws = Arc::new(FakeWs::failing(TransportError::Connect("refused".into())));
        let mut opts = opts();
        opts.conn_options = ApiConnectOptions { max_retry: 2, retry_interval: 0.0, timeout: 10.0 };
        let spawner = Arc::new(TokioSpawner::new(tokio::runtime::Handle::current()));
        let stt = NativeStt::new(opts, ws.clone(), spawner);
        let (_sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
        let err = tokio::time::timeout(Duration::from_secs(5), source.next())
            .await
            .expect("the retries settle")
            .expect_err("every attempt failed");
        assert_eq!(ws.connect_count(), 3, "the first attempt and two retries");
        assert!(err.message().contains("after 2 attempts"), "{}", err.message());
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
        stt.update_options(None, Some("fr"), Some(&crate::extra::ExtraKwargs(extra)));
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
        assert_eq!((word.start_time, word.end_time, word.start_time_offset), (Some(11.0), Some(11.5), Some(10.0)));
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
        let drained = tokio::time::timeout(std::time::Duration::from_millis(500), async {
            while let Some(event) = source.next().await.expect("no provider error") {
                let kind = event.kind;
                out.push(event);
                if kind == stop {
                    return;
                }
            }
        })
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
        let ws = Arc::new(
            FakeWs::new(vec![
                text(json!({"type": "session.created"})),
                text(json!({"type": "final_transcript", "transcript": "hello"})),
            ])
            .keep_open(),
        );
        let stt = stt(ws.clone());
        let (sink, mut source) = stt.stream(None, None).await.unwrap().unwrap();
        sink.push(frame(800)).unwrap();
        let event = source.next().await.unwrap().unwrap();
        assert_eq!(event.kind, SpeechEventType::StartOfSpeech);

        let sent = ws.sent_json();
        assert_eq!(sent[0]["type"], json!("session.create"));
        assert_eq!(sent[1]["type"], json!("input_audio"));
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

    // Catches: finalizing on the wrong VAD event (only END_OF_SPEECH), or on none.
    #[tokio::test]
    async fn only_the_vads_end_of_speech_finalizes_the_session() {
        use crate::abi::{Vad, VadCapabilities, VadSink, VadSource};
        use lk_agents_abi::{VadEvent, VadEventType};

        struct ScriptedVad(Mutex<Option<Vec<VadEventType>>>);
        #[async_trait]
        impl Vad for ScriptedVad {
            fn label(&self) -> String {
                "test".into()
            }
            fn capabilities(&self) -> VadCapabilities {
                VadCapabilities::default()
            }
            async fn stream(&self) -> Result<(VadSink, VadSource), ProviderError> {
                let kinds = self.0.lock().unwrap().take().unwrap_or_default();
                Ok((Box::new(NullSink), Box::new(ScriptedSource(kinds.into_iter().collect()))))
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
        #[async_trait]
        impl StreamSource<VadEvent> for ScriptedSource {
            async fn next(&mut self) -> Result<Option<VadEvent>, ProviderError> {
                Ok(self.0.pop_front().map(|kind| VadEvent::new(kind, 0, 0.0, 0.0, 0.0)))
            }
            async fn close(self: Box<Self>, _deadline: lk_agents_abi::Ts) -> lk_agents_abi::Closed {
                lk_agents_abi::Closed::Clean
            }
        }

        // Three inference-done events and one end-of-speech: exactly one finalize.
        let ws = Arc::new(FakeWs::new(vec![]).keep_open());
        let vad = Arc::new(ScriptedVad(Mutex::new(Some(vec![
            VadEventType::InferenceDone,
            VadEventType::InferenceDone,
            VadEventType::StartOfSpeech,
            VadEventType::EndOfSpeech,
            VadEventType::InferenceDone,
        ]))));
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

    // Catches: `model` returning "" when unset, or `capabilities()` returning the trait default.
    #[test]
    fn the_label_and_capabilities_come_from_the_configured_model() {
        let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let _guard = rt.enter();
        let stt = stt(Arc::new(FakeWs::new(vec![])));
        assert_eq!(stt.model(), "deepgram/nova-3");
        assert_eq!(stt.label(), "livekit.agents.inference.stt.STT");
        assert_eq!((stt.id().model.as_str(), stt.id().provider.as_str()), ("deepgram/nova-3", "livekit"));
        let caps = Stt::capabilities(&stt);
        assert!(caps.streaming && caps.interim_results && caps.keyterms);
        assert!(!caps.offline_recognize, "the default is true; this provider has no batch path");
        assert_eq!(caps.aligned_transcript, crate::abi::AlignedTranscript::Word);

        let mut opts = opts();
        opts.model = None;
        let unnamed = NativeStt::new(
            opts,
            Arc::new(FakeWs::new(vec![])),
            Arc::new(TokioSpawner::new(tokio::runtime::Handle::current())),
        );
        assert_eq!(unnamed.model(), "unknown");
    }

    // Catches: `recognize()` quietly returning an empty event.
    #[tokio::test]
    async fn batch_recognition_reports_that_it_is_unsupported() {
        let stt = stt(Arc::new(FakeWs::new(vec![])));
        let err = stt.recognize(&[], None).await.expect_err("no batch path");
        assert!(err.message().contains("does not support batch recognition"), "{}", err.message());
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
        sink.update_options(Some("deepgram/nova-2"), None, None);
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
        sink.apply_keyterms(extra);
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
}
