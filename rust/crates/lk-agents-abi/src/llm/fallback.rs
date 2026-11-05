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

    #[host(new)]
    fn new(
        env: &Self::Env,
        llms: Vec<std::sync::Arc<dyn crate::Llm>>,
        #[host(kw, default = 5.0)] attempt_timeout: f64,
        #[host(default = 0)] max_retry_per_llm: u32,
        #[host(default = 0.5)] retry_interval: f64,
        #[host(default = false)] sticky: bool,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::llm::inference::InvalidPayload>;

    #[host(getter)]
    fn active_index(&self) -> usize;

    #[host(await)]
    async fn aclose(&self);

    #[host(await)]
    #[allow(clippy::too_many_arguments, reason = "Preserve the public Python signature")]
    async fn chat(
        &self,
        chat_ctx_host: crate::HostRef,
        #[host(kw)] tools: Vec<crate::ToolSpec>,
        #[host(default = None)] tools_host: Option<crate::HostRef>,
        #[host(default = None)] parallel_tool_calls: Option<bool>,
        #[host(default = None)] tool_choice: Option<serde_json::Value>,
        #[host(default = None)] extra_kwargs: Option<crate::Extra>,
        #[host(default = None)] conn_options: Option<crate::ApiConnectOptions>,
    ) -> Result<Self::Chunks, crate::errors::ProviderError>;
}

/// `tts.FallbackAdapter`'s engine as the host builds it; the public class stays Python.
#[crate::host(class(name = "FallbackTts", native = crate::Tts))]
pub trait FallbackTts: Send + Sync + Sized + 'static {
    /// What the hosting process lends the class.
    #[host(env)]
    type Env;

    #[host(class(name = "TtsStream"))]
    type Stream;
    #[host(new)]
    fn new(
        env: &Self::Env,
        ttss: Vec<std::sync::Arc<dyn crate::Tts>>,
        #[host(kw, default = 2)] max_retry_per_tts: u32,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = None)] sample_rate: Option<u32>,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::llm::inference::InvalidPayload>;
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
        text: String,
        #[host(default = None)] conn_options: Option<crate::ApiConnectOptions>,
    ) -> Result<Self::Stream, crate::errors::ProviderError>;
    #[host(await)]
    async fn stream(
        &self,
        #[host(default = None)] conn_options: Option<crate::ApiConnectOptions>,
        #[host(default = false)] expressive: bool,
    ) -> Result<Self::Stream, crate::errors::ProviderError>;
}

/// One realtime fallback member: a native model or a host provider. Host providers open a fresh
/// adapter on every attempt, so a failed session is never reused by a later attempt.
#[crate::host(record(module = "llm_fallback", extract, transparent))]
#[derive(Clone)]
pub enum RealtimeFallbackMember {
    Native(std::sync::Arc<dyn crate::Realtime>),
    Host(std::sync::Arc<dyn crate::llm::realtime::HostRealtime>),
}

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
        models: Vec<RealtimeFallbackMember>,
        #[host(kw, default = 10.0)] cooldown: f64,
        #[host(default = None)] attempt_timeout: Option<f64>,
        #[host(default = true)] regenerate_on_swap: bool,
        #[host(default = None)] on_availability: Option<std::sync::Arc<dyn ChainObserver>>,
    ) -> Result<Self, crate::llm::realtime::RealtimeConfigurationError>;
    #[host(getter)]
    fn active_index(&self) -> usize;
    #[host(call)]
    fn capabilities(&self) -> crate::llm::realtime::RealtimeCapabilities;
    #[host(await)]
    async fn restart(&self, #[host(kw, default = false)] switch_model: bool);
}

/// The pushed-token log as the host's `SynthesizeStream` keeps it (`_core.TtsTokenLog`): the text
/// a recovering stream replays.
///
/// `self._pushed_tokens` plus the live child sink. The text leg may not drop (§1.2d), so nothing
/// here is lossy or bounded.
#[crate::host(class(name = "TtsTokenLog"))]
pub trait TtsTokenLog: Send + Sync + Sized + 'static {
    #[host(new)]
    fn new() -> Self;
    #[host(call)]
    fn push(&self, text: String);
    #[host(call)]
    fn flush(&self);
    #[host(call)]
    fn end(&self);
    #[host(call)]
    fn snapshot(&self) -> (Vec<Option<String>>, bool);
}

// Public data contracts. Python behavior remains in its adapters.

#[crate::host(record(
    module = "llm_availability",
    path = "livekit.agents.llm",
    dataclass
))]
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
    #[host(annotation = "livekit.agents.llm.realtime.RealtimeModel")]
    pub realtime_model: crate::HostRef,
    pub available: bool,
}
