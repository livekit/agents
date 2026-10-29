//! `metrics/usage.py`'s records. A `///` is API (generated pydantic); Rust notes go in `//`. Counts
//! are session-long sums, hence `u64`.

use serde::{Deserialize, Serialize};

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "usage", path = "livekit.agents.metrics"))]
const _: () = ();

// `_BaseModelUsage`. Python-only: a zero-eliding `__repr__`, and the base the gate records.
#[crate::host(record(
    module = "usage",
    name = "_BaseModelUsage",
    pydantic,
    repr = "model_usage_repr",
))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BaseModelUsage {}

// `usage.py:27`. LLM and realtime-model usage both fold in here.
/// Usage summary for LLM models.
#[crate::host(record(
    module = "usage",
    name = "LLMModelUsage",
    pydantic,
    base = "_BaseModelUsage",
    constant(type = "llm_usage"),
    emit,
    extract,
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmModelUsage {
    /// The provider name (e.g., 'openai', 'anthropic').
    pub provider: String,
    /// The model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
    pub model: String,
    /// Total input tokens.
    #[host(default = 0)]
    pub input_tokens: u64,
    /// Input tokens served from cache.
    #[host(default = 0)]
    pub input_cached_tokens: u64,
    // Fed only by `LlmMetrics::cache_creation_tokens`; zero for realtime.
    /// Input tokens used to write to the prompt cache (e.g. Anthropic cache writes).
    #[host(default = 0)]
    pub input_cache_creation_tokens: u64,
    /// Input audio tokens (for multimodal models).
    #[host(default = 0)]
    pub input_audio_tokens: u64,
    /// Cached input audio tokens.
    #[host(default = 0)]
    pub input_cached_audio_tokens: u64,
    /// Input text tokens.
    #[host(default = 0)]
    pub input_text_tokens: u64,
    /// Cached input text tokens.
    #[host(default = 0)]
    pub input_cached_text_tokens: u64,
    /// Input image tokens (for multimodal models).
    #[host(default = 0)]
    pub input_image_tokens: u64,
    /// Cached input image tokens.
    #[host(default = 0)]
    pub input_cached_image_tokens: u64,
    /// Total output tokens.
    #[host(default = 0)]
    pub output_tokens: u64,
    /// Output audio tokens (for multimodal models).
    #[host(default = 0)]
    pub output_audio_tokens: u64,
    /// Output text tokens.
    #[host(default = 0)]
    pub output_text_tokens: u64,
    /// Output tokens spent on hidden reasoning. Already counted in ``output_tokens``.
    #[host(default = 0)]
    pub output_reasoning_tokens: u64,
    /// Total session connection duration in seconds (for session-based billing like xAI).
    #[host(default = 0.0)]
    pub session_duration: f64,
}

impl LlmModelUsage {
    pub const TYPE: &'static str = "llm_usage";
}

// `usage.py:65`.
/// Usage summary for TTS models.
#[crate::host(record(
    module = "usage",
    name = "TTSModelUsage",
    pydantic,
    base = "_BaseModelUsage",
    constant(type = "tts_usage"),
    emit,
    extract,
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsModelUsage {
    /// The provider name (e.g., 'elevenlabs', 'cartesia').
    pub provider: String,
    /// The model name (e.g., 'eleven_turbo_v2', 'sonic').
    pub model: String,
    /// Input text tokens (for token-based TTS billing, e.g., OpenAI TTS).
    #[host(default = 0)]
    pub input_tokens: u64,
    /// Output audio tokens (for token-based TTS billing, e.g., OpenAI TTS).
    #[host(default = 0)]
    pub output_tokens: u64,
    /// Number of characters synthesized (for character-based TTS billing).
    #[host(default = 0)]
    pub characters_count: u64,
    /// Duration of generated audio in seconds.
    #[host(default = 0.0)]
    pub audio_duration: f64,
}

impl TtsModelUsage {
    pub const TYPE: &'static str = "tts_usage";
}

// `usage.py:83`.
/// Usage summary for STT models.
#[crate::host(record(
    module = "usage",
    name = "STTModelUsage",
    pydantic,
    base = "_BaseModelUsage",
    constant(type = "stt_usage"),
    emit,
    extract,
))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttModelUsage {
    /// The provider name (e.g., 'deepgram', 'assemblyai').
    pub provider: String,
    /// The model name (e.g., 'nova-2', 'best').
    pub model: String,
    /// Total input tokens, including both audio and text (for token-based STT billing).
    #[host(default = 0)]
    pub input_tokens: u64,
    /// Audio input tokens, a subset of input_tokens when reported by the provider.
    #[host(default = 0)]
    pub input_audio_tokens: u64,
    /// Output text tokens (for token-based STT billing).
    #[host(default = 0)]
    pub output_tokens: u64,
    /// Duration of processed audio in seconds.
    #[host(default = 0.0)]
    pub audio_duration: f64,
}

impl SttModelUsage {
    pub const TYPE: &'static str = "stt_usage";
}

// `usage.py:100`. One counter only: the interruption model is billed per request.
/// Usage summary for interruption detection models.
#[crate::host(record(
    module = "usage",
    pydantic,
    base = "_BaseModelUsage",
    constant(type = "interruption_usage"),
    emit,
    extract,
))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterruptionModelUsage {
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'adaptive').
    pub model: String,
    /// Total number of requests sent to the interruption detection model.
    #[host(default = 0)]
    pub total_requests: u64,
}

impl InterruptionModelUsage {
    pub const TYPE: &'static str = "interruption_usage";
}

// `usage.py:112`.
/// Usage summary for end-of-turn detection models.
#[crate::host(record(
    module = "usage",
    name = "EOTModelUsage",
    pydantic,
    base = "_BaseModelUsage",
    constant(type = "eot_usage"),
    emit,
    extract,
))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EotModelUsage {
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'turn-detector-v1').
    pub model: String,
    /// Total number of inference requests sent to the EOT model.
    #[host(default = 0)]
    pub total_requests: u64,
}

impl EotModelUsage {
    pub const TYPE: &'static str = "eot_usage";
}

// `usage.py:124`'s `ModelUsage`: a plain PEP-604 union, not a discriminated one.
/// Union type for all model usage types.
#[crate::host(record(module = "usage", transparent, emit, extract))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModelUsage {
    #[serde(rename = "llm_usage")]
    Llm(LlmModelUsage),
    #[serde(rename = "tts_usage")]
    Tts(TtsModelUsage),
    #[serde(rename = "stt_usage")]
    Stt(SttModelUsage),
    #[serde(rename = "interruption_usage")]
    Interruption(InterruptionModelUsage),
    #[serde(rename = "eot_usage")]
    Eot(EotModelUsage),
}

impl ModelUsage {
    pub fn type_literal(&self) -> &'static str {
        match self {
            ModelUsage::Llm(_) => LlmModelUsage::TYPE,
            ModelUsage::Tts(_) => TtsModelUsage::TYPE,
            ModelUsage::Stt(_) => SttModelUsage::TYPE,
            ModelUsage::Interruption(_) => InterruptionModelUsage::TYPE,
            ModelUsage::Eot(_) => EotModelUsage::TYPE,
        }
    }
}

// `usage.py:129`. The host's `AgentSessionUsage` is a hand-written dataclass (`metrics.py`); this
// is the session report's reading of it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AgentSessionUsage {
    pub model_usage: Vec<ModelUsage>,
}
