mod executor;
pub(crate) mod gaps;
pub(crate) mod input;
pub mod machine;
pub mod options;
mod provider_options;
pub mod retrying;
pub use retrying::{recognize, stream};
mod stream;
pub mod stream_adapter;
pub mod wire;

use crate::abi::{AudioFrame, StreamInput, StreamOutput, Stt, SttCapabilities};
use crate::headers::{BoundIdentity, Headers, gateway_headers};
use crate::telemetry;
use lk_agents_core::request::{Policy, Request, Step};
use lk_agents_net::task::TaskSet;
use lk_agents_net::transport::{
    TransportError, WsMessage, WsSender, WsTransport, connect_error, join_url, open_ws,
    stream_closed, to_ws_url, within,
};

use base64::Engine as _;
use executor::SttStreamExecutor;
use input::{AudioInput, Flush};
use lk_agents_abi::Spawner;
use lk_agents_abi::errors::{ApiError, Invalid, ProviderError};
use lk_agents_abi::inference::fallback::{SttFallbackEntry, SttFallbacks};
use lk_agents_abi::inference::models::InferenceProvider;
use lk_agents_abi::tokenize::language::LanguageCode;
use lk_agents_abi::types::provider::NativeLabel;
use lk_agents_abi::vad::inference::VadModels;
use lk_agents_abi::{ApiConnectOptions, Clock, SpeechEvent};
use lk_agents_media::AudioByteStream;
use lk_agents_media::{AudioResampler, OnChange, Quality};
use machine::{SttStream, SttStreamEvent};
use options::{FallbackModel, VadResolution};
use std::sync::{Arc, Weak};
use std::time::Duration;
use stream::{DeferredExtra, StreamControl};

use parking_lot::Mutex;
pub use stream::{RecognitionInput, RecognitionOutput};

use lk_agents_net::env::ProcessEnv;
use tokio::sync::mpsc;

const WHAT: &str = "LiveKit Inference STT";

/// A stream's event queue, in items.
pub(crate) const STREAM_QUEUE: usize = 50;

/// Quiet period after audio EOF, or after a later interim, that completes an unacknowledged stream.
pub(super) const FINALIZATION_TIMEOUT: Duration = Duration::from_secs(30);

/// `_MAX_AGENT_CONTEXT_CHARS`: AssemblyAI caps `agent_context`, so the tail is kept.
const MAX_AGENT_CONTEXT_CHARS: usize = 1750;

/// Builds the default VAD of a model that endpoints client-side.
type DefaultVad = Arc<dyn Fn() -> Arc<dyn crate::abi::Vad> + Send + Sync>;

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
pub struct GatewayStt {
    shared: Arc<SttClient>,
    identity: Arc<BoundIdentity>,
    conn_options: Option<ApiConnectOptions>,
    /// The host provider this view serves: its `error` event is each failure the core reports.
    reporter: Option<Arc<dyn lk_agents_abi::Stt>>,
}

impl Clone for GatewayStt {
    fn clone(&self) -> Self {
        GatewayStt {
            shared: Arc::clone(&self.shared),
            identity: Arc::clone(&self.identity),
            conn_options: self.conn_options,
            reporter: self.reporter.clone(),
        }
    }
}

/// What every session's view of one client shares: its options, connections and open streams.
pub struct SttClient {
    opts: Mutex<SttOptions>,
    capabilities: Mutex<SttCapabilities>,
    /// Framework-managed keyterms, merged into `extra` at connect and by `session.update`.
    session_keyterms: Mutex<Vec<String>>,
    keyterms_unsupported_warned: std::sync::atomic::AtomicBool,
    ws: Arc<dyn WsTransport>,
    spawner: Arc<dyn Spawner>,
    /// Stream executors: each ends on its own once its handles are gone.
    tasks: Arc<TaskSet>,
    clock: Arc<dyn Clock>,
    /// The Speechmatics path: a client-side VAD whose end-of-speech sends `session.finalize`.
    vad: Mutex<Option<Arc<dyn crate::abi::Vad>>>,
    /// The VAD of a model that endpoints client-side when the host gave none.
    default_vad: std::sync::OnceLock<DefaultVad>,
    /// Every open stream, which `update_options` and keyterm changes reach mid-stream.
    live: Mutex<Vec<Weak<StreamControl>>>,
}

impl GatewayStt {
    fn configured(&self, conn_options: Option<ApiConnectOptions>) -> Self {
        Self { conn_options: conn_options.or(self.conn_options), ..self.clone() }
    }

    pub fn new(
        opts: SttOptions,
        ws: Arc<dyn WsTransport>,
        spawner: Arc<dyn Spawner>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let capabilities =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        let shared = SttClient {
            opts: Mutex::new(opts),
            capabilities: Mutex::new(capabilities),
            session_keyterms: Mutex::new(Vec::new()),
            keyterms_unsupported_warned: Default::default(),
            ws,
            spawner,
            tasks: Arc::default(),
            clock,
            vad: Mutex::new(None),
            default_vad: std::sync::OnceLock::new(),
            live: Mutex::new(Vec::new()),
        };
        GatewayStt {
            shared: Arc::new(shared),
            identity: Arc::default(),
            conn_options: None,
            reporter: None,
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
        let view = GatewayStt {
            shared: Arc::clone(&self.shared),
            identity: Arc::default(),
            conn_options: self.conn_options,
            reporter,
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
    #[cfg(test)]
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
        *self.shared.vad.lock() = vad;
    }

    /// `vad` as the model and its fallbacks need it (`_resolve_vad_for_model`): a model that
    /// endpoints client-side keeps it, or gets a default one; a model that endpoints server-side
    /// drops it, with a warning. Later streams use it.
    fn resolve_vad(&self, vad: Option<Arc<dyn crate::abi::Vad>>) {
        let opts = self.shared.opts.lock().clone();
        let candidates = opts.model.iter().map(String::as_str);
        let mut candidates = candidates.chain(opts.fallback.iter().map(|f| f.model.as_str()));
        let model = candidates
            .find(|m| {
                options::resolve_vad_for_model(Some(m), false) == VadResolution::CreateDefault
            })
            .or(opts.model.as_deref());
        let vad = match options::resolve_vad_for_model(model, vad.is_some()) {
            VadResolution::DropAndWarn => {
                log::warn!(
                    target: lk_agents_abi::LOG_TARGET,
                    "`vad` will be ignored: model {:?} handles endpointing server-side.",
                    opts.model.as_deref().unwrap_or_default()
                );
                None
            }
            VadResolution::CreateDefault => self.shared.default_vad.get().map(|create| create()),
            VadResolution::Keep | VadResolution::None => vad,
        };
        self.set_vad(vad);
    }

    /// The open streams; a dropped one is pruned on the way.
    fn live_streams(&self) -> Vec<Arc<StreamControl>> {
        let mut live = self.shared.live.lock();
        live.retain(|w| w.strong_count() > 0);
        live.iter().filter_map(Weak::upgrade).collect()
    }

    pub fn model(&self) -> String {
        // `"unknown"`, not "": a blank model name in metrics looks like a dropped field.
        self.shared.opts.lock().model.clone().unwrap_or_else(|| "unknown".to_owned())
    }

    /// `update_options` on the STT; the live streams get it as a mid-stream `session.update`.
    pub fn update_options(
        &self,
        model: Option<&str>,
        language: Option<&str>,
        extra: Option<&serde_json::Map<String, serde_json::Value>>,
    ) {
        let mut opts = self.shared.opts.lock();
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
        *self.shared.capabilities.lock() =
            options::capabilities(opts.model.as_deref(), &opts.fallback, Some(&opts.extra_kwargs));
        // A user `extra` re-merges the session keyterms, so the update does not drop them.
        let extra = extra.map(|extra| {
            let mut merged = extra.clone();
            let terms = self.shared.session_keyterms.lock().clone();
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

    /// The `extra` entry a keyterm change applies, or `None` when unsupported or unchanged.
    pub fn update_session_keyterms(
        &self,
        keyterms: &[String],
    ) -> Option<serde_json::Map<String, serde_json::Value>> {
        {
            let existing = self.shared.session_keyterms.lock();
            if *existing == keyterms {
                return None;
            }
        }
        let opts = self.shared.opts.lock();
        let extra = options::keyterms_extra_for_model(
            opts.model.as_deref(),
            Some(&opts.extra_kwargs),
            keyterms,
        )?;
        drop(opts);
        *self.shared.session_keyterms.lock() = keyterms.to_vec();
        Some(extra)
    }

    /// `_sanitize_options`: a per-stream copy, so a later `update_options` on the STT cannot
    /// mutate a stream that is already running.
    fn sanitize(&self, language: Option<&str>) -> SttOptions {
        let mut opts = self.shared.opts.lock().clone();
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
    ) -> Result<lk_agents_abi::DuplexStream<RecognitionInput, RecognitionOutput>, ProviderError>
    {
        let opts = self.sanitize(language);
        if opts
            .model
            .as_deref()
            .into_iter()
            .chain(opts.fallback.iter().map(|entry| entry.model.as_str()))
            .any(|model| {
                lk_agents_abi::inference::models::InferenceProvider::of_model(model)
                    == Some(lk_agents_abi::inference::models::InferenceProvider::Openai)
            })
            && opts.sample_rate != options::OPENAI_SAMPLE_RATE
        {
            return Err(ApiError::new("OpenAI STT requires a sample rate of 24000 Hz")
                .with_retryable(false)
                .into());
        }
        opts.conn_options
            .validate()
            .map_err(|error| ApiError::new(error.to_string()).with_retryable(false))?;
        let keyterms = self.shared.session_keyterms.lock().clone();

        let (audio, audio_rx) = AudioInput::new(self.shared.clock.clone());
        let (event_tx, event_rx) = mpsc::channel::<Result<SpeechEvent, ApiError>>(STREAM_QUEUE);
        let (update_tx, update_rx) = mpsc::unbounded_channel::<serde_json::Value>();
        let stop = tokio_util::sync::CancellationToken::new();
        let stop_guard = Arc::new(stop.clone().drop_guard());
        let (failure, failure_rx) = tokio::sync::watch::channel(None);
        let (done, done_rx) = tokio::sync::oneshot::channel();
        let control = Arc::new(StreamControl {
            stop: Arc::clone(&stop_guard),
            failure,
            updates: update_tx.clone(),
            deferred: Arc::new(DeferredExtra {
                options: Mutex::new(stream::LiveOptions {
                    model: opts.model.clone(),
                    language: opts.language.clone(),
                }),
                ..Default::default()
            }),
        });
        self.shared.live.lock().push(Arc::downgrade(&control));

        let executor = SttStreamExecutor {
            ws: self.shared.ws.clone(),
            identification: self.identity.for_connection(),
            clock: self.shared.clock.clone(),
            create_frame: wire::session_create(&opts, &keyterms)?,
            vad: self.shared.vad.lock().clone(),
            offset: start_time_offset,
            opts,
            audio_rx,
            update_rx,
            deferred: control.deferred.clone(),
            update_tx,
            event_tx,
            failure: control.failure.clone(),
            stop,
            accepted: self.shared.clock.now(),
        };
        self.shared.tasks.spawn(self.shared.spawner.as_ref(), "stt stream", async move {
            executor.run().await;
            let _ = done.send(());
        });

        Ok(lk_agents_abi::DuplexStream::new(
            RecognitionInput { audio, control },
            RecognitionOutput {
                events: event_rx,
                done: false,
                stop: stop_guard,
                joined: done_rx,
                failure: failure_rx,
                clock: self.shared.clock.clone(),
            },
        ))
    }
}

/// `_core.InferenceStt`: the generated adapter calls this.
impl lk_agents_abi::stt::inference::InferenceStt for GatewayStt {
    type Audio = RecognitionInput;
    type Events = RecognitionOutput;
    type Env = ProcessEnv;

    fn new(
        env: &ProcessEnv,
        vad_models: Arc<dyn VadModels>,
        model: Option<String>,
        language: Option<String>,
        base_url: Option<String>,
        encoding: Option<String>,
        sample_rate: Option<u32>,
        api_key: Option<String>,
        api_secret: Option<String>,
        extra_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
        fallback: Option<SttFallbacks>,
        conn_options: Option<ApiConnectOptions>,
        vad: Option<Arc<dyn crate::abi::Vad>>,
    ) -> Result<Self, Invalid> {
        use lk_agents_abi::inference::InferenceCredentials as _;
        let credentials =
            crate::credentials::Credentials::inference_credentials(base_url, api_key, api_secret)?;
        let spawner = (env.spawner)().map_err(|e| Invalid(e.to_string()))?;
        let (model, parsed_language) = match &model {
            Some(model) => {
                let (name, language) = options::parse_model_string(model);
                (Some(name.to_owned()), language.map(str::to_owned))
            }
            None => (None, None),
        };
        let entries = fallback.map_or_else(Vec::new, SttFallbacks::entries);
        let fallback: Vec<FallbackModel> = entries
            .into_iter()
            .map(|entry| match entry {
                SttFallbackEntry::Model(model) => model,
                SttFallbackEntry::Name(name) => options::normalize_fallback_str(&name),
            })
            .collect();
        let openai = model
            .iter()
            .map(String::as_str)
            .chain(fallback.iter().map(|f| f.model.as_str()))
            .any(|m| InferenceProvider::of_model(m) == Some(InferenceProvider::Openai));
        let opts = SttOptions {
            model,
            language: language
                .or(parsed_language)
                .map(|l| LanguageCode::new(&l).as_str().to_owned()),
            encoding: encoding.unwrap_or_else(|| options::DEFAULT_ENCODING.to_owned()),
            sample_rate: sample_rate.unwrap_or(match openai {
                true => options::OPENAI_SAMPLE_RATE,
                false => options::DEFAULT_SAMPLE_RATE,
            }),
            base_url: credentials.base_url,
            api_key: credentials.api_key,
            api_secret: credentials.api_secret,
            extra_kwargs: extra_kwargs.unwrap_or_default(),
            fallback,
            conn_options: conn_options.unwrap_or(lk_agents_abi::DEFAULT_API_CONNECT_OPTIONS),
        };
        let ws = Arc::new(lk_agents_net::websocket::TungsteniteTransport);
        let stt = GatewayStt::new(opts, ws, spawner, env.clock.clone());
        let env = env.clone();
        let default_vad: DefaultVad = Arc::new(move || {
            let opts = crate::vad::machine::VadOptions::default();
            Arc::new(crate::vad::LocalVad::with_host_models(opts, vad_models.clone(), &env))
        });
        let _ = stt.shared.default_vad.set(default_vad);
        stt.resolve_vad(vad);
        Ok(stt)
    }

    fn model(&self) -> String {
        GatewayStt::model(self)
    }

    fn capabilities(&self) -> SttCapabilities {
        *self.shared.capabilities.lock()
    }

    fn conn_options(&self) -> ApiConnectOptions {
        self.shared.opts.lock().conn_options
    }

    fn identified(&self, headers: Headers, reporter: Option<Arc<dyn Stt>>) -> Self {
        GatewayStt::identified(self, headers, reporter)
    }

    fn update_options(
        &self,
        model: Option<String>,
        language: Option<String>,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) {
        GatewayStt::update_options(self, model.as_deref(), language.as_deref(), extra.as_ref());
        if model.is_some() {
            let vad = self.shared.vad.lock().clone();
            self.resolve_vad(vad);
        }
    }

    fn stream(
        &self,
        language: Option<String>,
        start_time_offset: Option<f64>,
        conn_options: Option<ApiConnectOptions>,
    ) -> Result<lk_agents_abi::DuplexStream<RecognitionInput, RecognitionOutput>, ProviderError>
    {
        self.configured(conn_options)
            .stream_native(language.as_deref(), start_time_offset.unwrap_or(0.0))
    }
}

impl Stt for GatewayStt {
    fn with_conn_options(&self, conn: &ApiConnectOptions) -> Option<Arc<dyn Stt>> {
        Some(Arc::new(self.configured(Some(*conn))))
    }

    fn conn_options(&self) -> Option<ApiConnectOptions> {
        self.conn_options
    }

    /// The provider's `error` event, on the host provider this view serves.
    fn failed(&self, error: ProviderError, recoverable: bool) {
        if let Some(reporter) = &self.reporter {
            reporter.failed(error, recoverable);
        }
    }

    fn label(&self) -> String {
        NativeLabel::InferenceStt.as_str().into()
    }

    fn id(&self) -> lk_agents_abi::ProviderId {
        lk_agents_abi::ProviderId {
            label: Stt::label(self),
            model: self.model(),
            provider: "livekit".into(),
        }
    }

    /// `_update_session_keyterms`: a warning once for a model without keyterm prompting, else the
    /// keyterms reach every open stream (now if the user is silent, at end of speech if not).
    fn update_keyterms(&self, keyterms: &[String]) {
        if !self.shared.capabilities.lock().keyterms {
            if !self
                .shared
                .keyterms_unsupported_warned
                .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                log::warn!(
                    target: lk_agents_abi::LOG_TARGET,
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
    fn push_conversation_item(&self, role: &str, text: &str) {
        if role != "assistant" || text.is_empty() || !self.shared.capabilities.lock().chat_context {
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

    fn capabilities(&self) -> SttCapabilities {
        *self.shared.capabilities.lock()
    }

    /// "LiveKit Inference STT does not support batch recognition, use stream() instead".
    fn recognize(
        &self,
        _cx: &telemetry::Cx,
        _buffer: &[AudioFrame],
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
        cx: &telemetry::Cx,
        language: Option<&lk_agents_abi::tokenize::language::LanguageCode>,
        start_time_offset: Option<f64>,
    ) -> lk_agents_abi::BoxFuture<
        '_,
        Result<
            Option<lk_agents_abi::DuplexStream<crate::abi::SttSink, crate::abi::SttSource>>,
            ProviderError,
        >,
    > {
        let language = language.cloned();
        let cx = cx.clone();
        Box::pin(async move {
            let offset = start_time_offset.unwrap_or(0.0);
            let options = self.conn_options.unwrap_or_else(|| self.shared.opts.lock().conn_options);
            let provider = self.configured(Some(ApiConnectOptions { max_retry: 0, ..options }));
            provider.stream_native(language.as_ref().map(|l| l.as_str()), offset).map(|stream| {
                let (sink, source) = stream.split();
                sink.audio.observe(cx);
                Some(lk_agents_abi::DuplexStream::new(
                    Box::new(sink) as crate::abi::SttSink,
                    Box::new(source) as crate::abi::SttSource,
                ))
            })
        })
    }
}

#[cfg(test)]
#[path = "stt_tests.rs"]
mod tests;
