//! usage.md's records: what an operation consumed. A `///` is API (generated pydantic); Rust
//! notes go in `//`. The accounting that builds them is `lk_agents_core::usage`'s.

use serde::{Deserialize, Serialize};

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "usage", path = "livekit.agents"))]
const _: () = ();

/// Token counts in one vocabulary. Totals are inclusive: the cache, reasoning and modality
/// breakdowns are subsets of them and are never added to them. A breakdown is ``None`` when the
/// provider did not report it; a reported zero stays zero.
#[crate::host(record(module = "usage", pydantic, frozen, emit, extract))]
#[python(property(
    name = "total_tokens",
    ret = "int",
    hook = "token_usage_total",
    doc = "``input_tokens + output_tokens``."
))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TokenUsage {
    /// All input tokens, including cache reads and writes and every modality.
    #[host(default = 0)]
    pub input_tokens: u64,
    /// All output tokens, including hidden reasoning and every modality.
    #[host(default = 0)]
    pub output_tokens: u64,
    /// Input tokens read from cache.
    #[host(default = None)]
    pub cache_read_tokens: Option<u64>,
    /// Input tokens written to cache.
    #[host(default = None)]
    pub cache_write_tokens: Option<u64>,
    /// Reasoning tokens included in output.
    #[host(default = None)]
    pub output_reasoning_tokens: Option<u64>,
    #[host(default = None)]
    pub input_text_tokens: Option<u64>,
    #[host(default = None)]
    pub input_audio_tokens: Option<u64>,
    #[host(default = None)]
    pub input_image_tokens: Option<u64>,
    #[host(default = None)]
    pub output_text_tokens: Option<u64>,
    #[host(default = None)]
    pub output_audio_tokens: Option<u64>,
    #[host(default = None)]
    pub output_image_tokens: Option<u64>,
    #[host(default = None)]
    pub cache_read_text_tokens: Option<u64>,
    #[host(default = None)]
    pub cache_read_audio_tokens: Option<u64>,
    #[host(default = None)]
    pub cache_read_image_tokens: Option<u64>,
}

impl TokenUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// An LLM's usage, realtime models included.
#[crate::host(record(
    module = "usage",
    name = "LLMUsage",
    pydantic,
    frozen,
    constant(type = "llm"),
    emit,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmUsage {
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = 0)]
    pub requests: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// Metered session-connection seconds.
    #[host(default = None)]
    pub connection_duration: Option<f64>,
}

/// A speech-to-text model's usage.
#[crate::host(record(
    module = "usage",
    name = "STTUsage",
    pydantic,
    frozen,
    constant(type = "stt"),
    emit,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttUsage {
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = 0)]
    pub requests: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// Processed input audio seconds.
    #[host(default = None)]
    pub audio_duration: Option<f64>,
}

/// A text-to-speech model's usage.
#[crate::host(record(
    module = "usage",
    name = "TTSUsage",
    pydantic,
    frozen,
    constant(type = "tts"),
    emit,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsUsage {
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = 0)]
    pub requests: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// Synthesized output audio seconds.
    #[host(default = None)]
    pub audio_duration: Option<f64>,
    /// Synthesized characters.
    #[host(default = None)]
    pub characters_count: Option<u64>,
}

/// An end-of-turn model's usage, priced per request.
#[crate::host(record(
    module = "usage",
    name = "EOTUsage",
    pydantic,
    frozen,
    constant(type = "eot"),
    emit,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EotUsage {
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = 0)]
    pub requests: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
}

/// An interruption detector's usage, priced per request.
#[crate::host(record(
    module = "usage",
    pydantic,
    frozen,
    constant(type = "interruption"),
    emit,
    extract
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterruptionUsage {
    #[host(default = None)]
    pub provider: Option<String>,
    #[host(default = None)]
    pub model: Option<String>,
    #[host(default = 0)]
    pub requests: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
}

/// One model's usage, keyed by ``(type, provider, model)``.
#[crate::host(record(module = "usage", transparent, discriminator = "type", emit, extract))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelUsage {
    Llm(LlmUsage),
    Stt(SttUsage),
    Tts(TtsUsage),
    Eot(EotUsage),
    Interruption(InterruptionUsage),
}

/// What an operation consumed: a text run, a text-session turn or a voice session. An immutable
/// snapshot; the operation's accessor returns the latest one.
#[crate::host(record(module = "usage", pydantic, frozen, emit, extract))]
#[python(
    property(
        name = "input_tokens",
        ret = "int",
        hook = "usage_input_tokens",
        doc = "``tokens.input_tokens``."
    ),
    property(
        name = "output_tokens",
        ret = "int",
        hook = "usage_output_tokens",
        doc = "``tokens.output_tokens``."
    ),
    property(
        name = "total_tokens",
        ret = "int",
        hook = "usage_total_tokens",
        doc = "``tokens.total_tokens``."
    )
)]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    /// Per-model usage, in first-observed order.
    #[host(default_factory = "tuple()", annotation = "tuple[ModelUsage, ...]")]
    pub models: Vec<ModelUsage>,
    /// The aggregate of ``models``' tokens.
    #[host(default_factory = "TokenUsage()")]
    pub tokens: TokenUsage,
    /// The sum of the per-model request counts.
    #[host(default = 0)]
    pub requests: u64,
    /// Tool-body executions started in this scope.
    #[host(default = 0)]
    pub tool_calls: u64,
    /// Requests whose expected usage measurement is unavailable.
    #[host(default = 0)]
    pub unmeasured_requests: u64,
    /// The operation finished local execution and accounting cleanup.
    #[host(default = false)]
    pub settled: bool,
    /// The operation is settled and no expected usage measurement is missing.
    #[host(default = false)]
    pub complete: bool,
}

impl Usage {
    pub fn input_tokens(&self) -> u64 {
        self.tokens.input_tokens
    }

    pub fn output_tokens(&self) -> u64 {
        self.tokens.output_tokens
    }

    pub fn total_tokens(&self) -> u64 {
        self.tokens.total_tokens()
    }
}
