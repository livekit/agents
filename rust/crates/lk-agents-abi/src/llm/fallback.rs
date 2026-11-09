//! The fallback adapters: `AvailabilityChangedEvent` (all of them), the chain's report to its
//! adapter, and the native LLM, TTS and realtime chains.

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
#[crate::host(proxy(host = "livekit.agents._bindings.FallbackListener"))]
pub trait ChainObserver: Send + Sync {
    #[host(notify, name = "_core_availability_changed")]
    fn availability_changed(&self, ev: AvailabilityChanged);

    /// `logger.error("all X are unavailable, retrying..")`. The chain then tries every entry
    /// anyway rather than failing fast, which is what `all_failed` does in all three adapters.
    #[host(notify, name = "_core_all_unavailable")]
    fn all_unavailable(&self) {}
}

/// The native LLM chain behind `llm.FallbackAdapter`. Runtime owns its live providers and tasks.
#[crate::host(class(name = "FallbackLlm", native = crate::Llm))]
pub trait FallbackLlm: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(source(item = crate::LlmChunk, class = "FallbackLlmChunks"))]
    type Chunks: crate::StreamOutput<crate::LlmChunk> + 'static;

    /// `llm` in order.
    #[host(new)]
    fn new(
        env: &Self::Env,
        llm: Vec<std::sync::Arc<dyn crate::Llm>>,
        #[host(kw, default = 5.0)] attempt_timeout: f64,
        #[host(default = 0)] max_retry_per_llm: u32,
        #[host(default = 0.5)] retry_interval: f64,
        #[host(default = false)] sticky: bool,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::errors::Invalid>;

    #[host(getter)]
    fn active_index(&self) -> usize;

    #[host(await)]
    async fn aclose(&self);

    /// A completion of the host `chat_ctx`, offering `tools`; the chain picks an entry on the
    /// first pull.
    #[host(await)]
    async fn chat(
        &self,
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
    legs(chat),
    availability(
        observer = "on_availability",
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
    #[host(new, resolve(llm = ".._bindings.core_legs"))]
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

    #[host(class(name = "TtsStream"))]
    type Stream;
    /// `tts` in order; `sample_rate` defaults to the highest of theirs, the others resampled.
    #[host(new)]
    fn new(
        env: &Self::Env,
        tts: Vec<std::sync::Arc<dyn crate::Tts>>,
        #[host(kw, default = 2)] max_retry_per_tts: u32,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::errors::Invalid>;
    #[host(getter)]
    fn active_index(&self) -> usize;
    #[host(getter)]
    fn capabilities(&self) -> crate::TtsCapabilities;
    #[host(getter)]
    fn sample_rate(&self) -> u32;
    #[host(getter)]
    fn num_channels(&self) -> u16;
    #[host(await)]
    async fn aclose(&self);
    #[host(await)]
    async fn synthesize(
        &self,
        input_text: String,
        #[host(kw, default = None)] conn_options: Option<crate::ApiConnectOptions>,
    ) -> Result<Self::Stream, crate::errors::ProviderError>;
    #[host(await)]
    async fn stream(
        &self,
        #[host(kw, default = None)] conn_options: Option<crate::ApiConnectOptions>,
        #[host(default = false)] expressive: bool,
    ) -> Result<Self::Stream, crate::errors::ProviderError>;
}

/// `tts.FallbackAdapter`: several TTS behind one; a failed one is skipped until it recovers.
#[crate::host(facade(
    path = "livekit.agents.tts",
    extended,
    name = "FallbackAdapter",
    base = "tts.TTS",
    native = FallbackTts,
    legs(synthesize, stream),
    base_args(capabilities, sample_rate, num_channels),
    availability(
        observer = "on_availability",
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
    ///     attempt_timeout: Caps each phase of one attempt; ``None`` keeps the caller's
    ///         ``conn_options`` limits.
    ///     sample_rate: The output rate; the highest of the TTS by default, the others
    ///         resampled.
    #[host(new, resolve(tts = ".._bindings.core_legs"))]
    fn new(
        tts: Vec<std::sync::Arc<dyn crate::Tts>>,
        #[host(kw, default = 2)] max_retry_per_tts: u32,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = None)] sample_rate: Option<u32>,
    ) -> Self;
}

/// `tts.FallbackAdapter.synthesize()`: the chain's synthesis, which retries and falls back.
#[crate::host(facade(
    path = "livekit.agents.tts",
    base = "tts.ChunkedStream",
    native = FallbackTts,
    stream(
        op = synthesize,
        owner = "_tts",
        native_retries,
        method = "synthesize",
        base_args(input_text, conn_options),
        rename(input_text = "text"),
        default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS")
    )
))]
pub trait FallbackChunkedStream {}

/// `tts.FallbackAdapter.stream()`: the chain's synthesis, which retries and falls back.
#[crate::host(facade(
    path = "livekit.agents.tts",
    base = "tts.SynthesizeStream",
    native = FallbackTts,
    stream(
        op = stream,
        owner = "_tts",
        native_retries,
        method = "stream",
        base_args(conn_options),
        state(expressive),
        default(conn_options = "..types.DEFAULT_API_CONNECT_OPTIONS")
    )
))]
pub trait FallbackSynthesizeStream {}

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

    #[host(new)]
    fn new(
        env: &Self::Env,
        models: Vec<RealtimeLeg>,
        #[host(kw, default = 10.0)] cooldown: f64,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = true)] regenerate_on_swap: bool,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::llm::realtime::RealtimeConfigurationError>;
    #[host(getter)]
    fn active_index(&self) -> usize;
    /// The members' capabilities, merged: the pipeline-shaping ones agree, the others are ANDed.
    #[host(getter)]
    fn capabilities(&self) -> crate::llm::realtime::RealtimeCapabilities;
    #[host(await)]
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
    base_args(capabilities),
    availability(
        observer = "on_availability",
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
    #[host(new, resolve(models = ".._bindings.core_legs"))]
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
    #[host(native = restart)]
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
