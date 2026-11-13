/// `AvailabilityChangedEvent`, identical in all three fallback adapters: an entry, by position in
/// the host's list, went unavailable or recovered. The host maps the index back to its instance.
#[derive(Debug, Clone, PartialEq, Eq)]
#[crate::host(record(module = "llm_fallback", emit))]
pub struct AvailabilityChanged {
    pub index: usize,
    pub label: String,
    pub available: bool,
}

/// What a fallback chain reports to the adapter that built it (each host `FallbackAdapter`, the
/// realtime one included): the adapter maps the index back to its instance.
#[crate::host(proxy(host = "livekit.agents._bindings.FallbackListener", weak))]
pub trait ChainObserver: Send + Sync {
    #[host(notify, name = "_core_availability_changed")]
    fn availability_changed(&self, ev: AvailabilityChanged);

    /// `logger.error("all X are unavailable, retrying..")`. The chain then tries every entry
    /// anyway rather than failing fast, which is what `all_failed` does in all three adapters.
    #[host(notify, name = "_core_all_unavailable")]
    fn all_unavailable(&self) {}

    /// The adapter's `error` event: a failed attempt of a call over the chain, `recoverable`
    /// while the call is retried.
    #[host(notify, name = "_core_failed")]
    fn failed(&self, _error: crate::errors::ProviderError, _recoverable: bool) {}
}

/// The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks.
#[crate::host(class(name = "FallbackLlm", native = crate::Llm))]
pub trait FallbackLlm: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(source(item = crate::LlmChunk, class = "FallbackLlmChunks"))]
    type Chunks: crate::StreamOutput<crate::LlmChunk> + 'static;

    /// `llm` in order; `observer` hears the chain's availability and failures.
    #[host(init)]
    #[allow(clippy::too_many_arguments, reason = "mirrors the Python constructor's arguments")]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        observer: std::sync::Arc<dyn ChainObserver>,
        llm: Vec<std::sync::Arc<dyn crate::Llm>>,
        #[host(kw, default = 5.0)] attempt_timeout: f64,
        #[host(default = 0)] max_retry_per_llm: u32,
        #[host(default = 0.5)] retry_interval: f64,
        #[host(default = false)] sticky: bool,
    ) -> Result<Self, crate::errors::Invalid>;

    /// Which entry most recently served, as a position in the host's list.
    #[host(getter, name = "_core_active_index")]
    fn active_index(&self) -> usize;

    #[host(await)]
    async fn aclose(&self);

    /// A completion of the host `chat_ctx`, offering `tools`; the chain picks an entry on the
    /// first pull.
    #[host(await, name = "_core_native_chat")]
    #[allow(clippy::too_many_arguments, reason = "mirrors the Python method's keyword arguments")]
    async fn chat(
        &self,
        cx: crate::telemetry::Cx,
        #[host(kw)] chat_ctx: HostChatContext,
        #[host(default = None)] tools: Option<HostTools>,
        #[host(default = None)] parallel_tool_calls: Option<bool>,
        #[host(default = None)] tool_choice: Option<serde_json::Value>,
        #[host(default = None)] extra_kwargs: Option<crate::Extra>,
        #[host(default = None)] conn_options: Option<crate::ApiConnectOptions>,
    ) -> Result<Self::Chunks, crate::errors::ProviderError>;
}

/// A host tool list (`list[Tool | Toolset]`) as a native chat takes it: the specs a native model
/// is offered, and the list itself, which a host model is handed.
#[crate::host(record(module = "llm_fallback", host_type = "builtins.object", boundary, extract))]
#[derive(Debug, Clone)]
pub struct HostTools {
    #[host(own_value, extract_with = "py_values::tool_specs")]
    pub specs: Vec<crate::ToolSpec>,
    #[host(own_value, extract_with = "py_values::host_ref")]
    pub host: crate::HostRef,
}

/// A host `llm.ChatContext` as a native chat takes it: its items, and the context itself, which a
/// host model is handed.
#[crate::host(record(module = "llm_fallback", host_type = "builtins.object", boundary, extract))]
#[derive(Debug, Clone)]
pub struct HostChatContext {
    pub items: Vec<crate::ChatItem>,
    #[host(own_value, extract_with = "py_values::host_ref")]
    pub host: crate::HostRef,
}

/// `llm.FallbackAdapter`: several LLMs behind one; a failed one is skipped until it recovers.
#[crate::host(facade(
    path = "livekit.agents.llm",
    extended,
    name = "FallbackAdapter",
    base = "llm.LLM",
    native = FallbackLlm,
    refuse(chat),
    node = "session.llm_node",
    availability(
        observer = "observer",
        members = "llm",
        kind = "LLM",
        event = "llm_availability_changed",
        record = ".llm_availability.AvailabilityChangedEvent"
    )
))]
pub trait FallbackLlmAdapter {
    /// Several LLMs behind one: a failed one is skipped until it recovers.
    ///
    /// Args:
    ///     llm: The LLMs, in order of preference.
    ///     attempt_timeout: The timeout of one attempt.
    ///     max_retry_per_llm: Retries of each before moving to the next.
    ///     retry_interval: The wait between retries.
    ///     sticky: Keep the LLM that last served until it fails, even once a preferred one
    ///         recovers; on failure the rest are tried in order.
    #[host(new)]
    fn new(
        llm: Vec<std::sync::Arc<dyn crate::Llm>>,
        #[host(kw, default = 5.0)] attempt_timeout: f64,
        #[host(default = 0)] max_retry_per_llm: u32,
        #[host(default = 0.5)] retry_interval: f64,
        #[host(default = false)] sticky: bool,
    ) -> Self;
}

/// `llm.FallbackAdapter.chat()`: the chain's completion, which retries and falls back.
#[crate::host(facade(
    path = "livekit.agents.llm",
    base = "llm.LLMStream",
    native = FallbackLlm,
    stream(
        op = chat,
        owner = "_llm",
        native_retries,
        method = "chat",
        base_args(chat_ctx, tools, conn_options),
        default(tools = "builtins.None", conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS"),
        annotation(
            chat_ctx = "livekit.agents.llm.ChatContext",
            tools = "list[livekit.agents.llm.Tool] | None",
            tool_choice = "livekit.agents.llm.ToolChoice",
            extra_kwargs = "dict[str, Any]"
        ),
        not_given
    )
))]
pub trait FallbackLLMStream {}

/// `tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[crate::host(class(name = "FallbackTts", native = crate::Tts))]
pub trait FallbackTts: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `tts` in order; `sample_rate` defaults to the highest of theirs, the others resampled.
    /// `observer` hears the chain's availability and failures.
    #[host(init)]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        observer: std::sync::Arc<dyn ChainObserver>,
        tts: Vec<std::sync::Arc<dyn crate::Tts>>,
        #[host(kw, default = 2)] max_retry_per_tts: u32,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = None)] sample_rate: Option<u32>,
    ) -> Result<Self, crate::errors::Invalid>;
    /// Which entry most recently served, as a position in the host's list.
    #[host(getter, name = "_core_active_index")]
    fn active_index(&self) -> usize;
    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(getter)]
    fn num_channels(&self) -> u16;
    #[host(await)]
    async fn aclose(&self);
}

/// `tts.FallbackAdapter`: several TTS behind one; a failed one is skipped until it recovers.
#[crate::host(facade(
    path = "livekit.agents.tts",
    extended,
    name = "FallbackAdapter",
    base = "tts.TTS",
    native = FallbackTts,
    refuse(_synthesize, _stream),
    node = "session.tts_node",
    base_args(capabilities, sample_rate, num_channels),
    availability(
        observer = "observer",
        members = "tts",
        kind = "TTS",
        event = "tts_availability_changed",
        record = ".tts_availability.AvailabilityChangedEvent"
    )
))]
pub trait FallbackTtsAdapter {
    /// Several TTS behind one: a failed one is skipped until it recovers.
    ///
    /// Args:
    ///     tts: The TTS, in order of preference.
    ///     max_retry_per_tts: Retries of each before moving to the next.
    ///     attempt_timeout: Caps each limit of one attempt; ``None`` keeps the caller's
    ///         ``conn_options`` limits.
    ///     sample_rate: The output rate; the highest of the TTS by default, the others
    ///         resampled.
    #[host(new)]
    fn new(
        tts: Vec<std::sync::Arc<dyn crate::Tts>>,
        #[host(kw, default = 2)] max_retry_per_tts: u32,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = None)] sample_rate: Option<u32>,
    ) -> Self;
}

/// A realtime model as the core takes it (a voice session's, a fallback member): a native model
/// or a host provider. A fallback opens a host provider afresh on every attempt, so a failed
/// session is never reused by a later attempt.
#[crate::host(record(module = "llm_fallback", extract, transparent))]
#[derive(Clone)]
pub enum RealtimeLeg {
    Native(std::sync::Arc<dyn crate::Realtime>),
    Host(std::sync::Arc<dyn crate::llm::realtime::HostRealtime>),
}

/// The realtime fallback's `model` name, which labels the error of a full event queue.
pub const REALTIME_FALLBACK_MODEL: &str = "RealtimeModelFallbackAdapter";
/// The error code of a reply the held backlog refused during a swap.
pub const REALTIME_FALLBACK_BACKLOG_EXCEEDED: &str = "realtime_fallback_backlog_exceeded";
/// The error code of a session failed because its event queue filled during a swap.
pub const REALTIME_FALLBACK_EVENT_BACKLOG_EXCEEDED: &str =
    "realtime_fallback_event_backlog_exceeded";

/// `llm.RealtimeModelFallbackAdapter`'s engine: the generated facade holds native resources; host
/// provider objects use the shared bridge.
#[crate::host(class(name = "RealtimeFallbackModel", native = crate::Realtime))]
pub trait RealtimeFallbackModel: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    /// `models` in order; `observer` hears the chain's availability and failures.
    #[host(init)]
    fn new(
        env: &Self::Env,
        #[host(label)] label: String,
        observer: std::sync::Arc<dyn ChainObserver>,
        models: Vec<RealtimeLeg>,
        #[host(kw, default = 10.0)] cooldown: f64,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = true)] regenerate_on_swap: bool,
    ) -> Result<Self, crate::llm::realtime::RealtimeConfigurationError>;
    /// Which entry most recently served, as a position in the host's list.
    #[host(getter, name = "_core_active_index")]
    fn active_index(&self) -> usize;
    /// The members' capabilities, merged: the pipeline-shaping ones agree, the others are ANDed.
    #[host(getter)]
    fn capabilities(&self) -> crate::llm::realtime::RealtimeCapabilities;
    #[host(await, name = "_core_restart")]
    async fn restart(&self, #[host(kw, default = false)] switch_model: bool);
}

/// `llm.RealtimeModelFallbackAdapter`: falls back between realtime models (or restarts one),
/// preserving chat context and handlers. A failed model is skipped until its cooldown ends, the
/// conversation is replayed onto the next one, and a reply cut by the swap is asked for again.
#[crate::host(facade(
    path = "livekit.agents.llm.realtime",
    extended,
    name = "RealtimeModelFallbackAdapter",
    base = "llm.realtime.RealtimeModel",
    native = RealtimeFallbackModel,
    refuse(session),
    node = "session.llm_node",
    base_args(capabilities),
    availability(
        observer = "observer",
        members = "models",
        kind = "RealtimeModel",
        event = "realtime_availability_changed",
        record = ".realtime_availability.RealtimeAvailabilityChangedEvent"
    )
))]
pub trait RealtimeFallbackAdapter {
    /// Fall back between realtime models while preserving chat context.
    ///
    /// Args:
    ///     models: Ordered models; the first is primary, the rest fallbacks. All must agree on
    ///         the ``audio_output`` and ``turn_detection`` capabilities.
    ///     cooldown: Seconds a failed model stays unavailable before it can be preferred again.
    ///     regenerate_on_swap: Re-issue the reply on the new session if one was in progress.
    ///     attempt_timeout: Seconds one model's session may take to open before the next is
    ///         tried; ``None`` waits for the model's own connection limits.
    ///
    /// Raises:
    ///     ValueError: If no models are given or their hard capabilities disagree.
    #[host(new)]
    fn new(
        models: Vec<std::sync::Arc<dyn crate::llm::realtime::HostRealtime>>,
        #[host(kw, default = 10.0)] cooldown: f64,
        #[host(default = true)] regenerate_on_swap: bool,
        #[host(default = None)] attempt_timeout: Option<f64>,
    ) -> Self;

    /// Bring up a fresh underlying session, preserving chat context and bound handlers.
    ///
    /// Args:
    ///     switch_model: Bring the new session up on the next available model instead of the
    ///         current one.
    #[host(native = _core_restart)]
    async fn restart_session(&self, #[host(kw, default = false)] switch_model: bool);
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(module = "llm_availability", path = "livekit.agents.llm", dataclass))]
pub struct AvailabilityChangedEvent {
    #[host(annotation = ".llm.LLM")]
    pub llm: crate::HostRef,
    pub available: bool,
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "realtime_availability",
    path = "livekit.agents.llm.realtime",
    dataclass
))]
pub struct RealtimeAvailabilityChangedEvent {
    #[host(annotation = ".realtime.RealtimeModel")]
    pub realtime_model: crate::HostRef,
    pub available: bool,
}
