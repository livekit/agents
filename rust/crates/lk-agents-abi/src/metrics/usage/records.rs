//! `metrics/usage.py`'s records. A `///` is API (generated pydantic); Rust notes go in `//`. Counts
//! are session-long sums, hence `u64`.

use lk_agents_host_derive::Host;

// The Python `module` and `path` of every record here that names none of its own.
#[crate::host(file(module = "usage", path = "livekit.agents.metrics.usage"))]
const _: () = ();

// `_BaseModelUsage`. Python-only: a zero-eliding `__repr__`, and the base the gate records.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "_BaseModelUsage",
    repr = "model_usage_repr"
)]
pub struct BaseModelUsage {}

// `usage.py:27`. LLM and realtime-model usage both fold in here.
/// Usage summary for LLM models.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "LLMModelUsage",
    base = "_BaseModelUsage",
    tag = "type",
    tag_const = TYPE
)]
// Both directions; the read side is all-`default` (a missing counter is zero), as the reference's
// `.get(k, 0)` reads were.
#[binding(emit, extract, from_items, default)]
pub struct LlmModelUsage {
    /// The provider name (e.g., 'openai', 'anthropic').
    pub provider: String,
    /// The model name (e.g., 'gpt-4o', 'claude-3-5-sonnet').
    pub model: String,
    /// Total input tokens.
    pub input_tokens: u64,
    /// Input tokens served from cache.
    pub input_cached_tokens: u64,
    // Fed only by `LlmMetrics::cache_creation_tokens`; zero for realtime.
    /// Input tokens used to write to the prompt cache (e.g. Anthropic cache writes).
    pub input_cache_creation_tokens: u64,
    /// Input audio tokens (for multimodal models).
    pub input_audio_tokens: u64,
    /// Cached input audio tokens.
    pub input_cached_audio_tokens: u64,
    /// Input text tokens.
    pub input_text_tokens: u64,
    /// Cached input text tokens.
    pub input_cached_text_tokens: u64,
    /// Input image tokens (for multimodal models).
    pub input_image_tokens: u64,
    /// Cached input image tokens.
    pub input_cached_image_tokens: u64,
    /// Total output tokens.
    pub output_tokens: u64,
    /// Output audio tokens (for multimodal models).
    pub output_audio_tokens: u64,
    /// Output text tokens.
    pub output_text_tokens: u64,
    /// Total session connection duration in seconds (for session-based billing like xAI).
    pub session_duration: f64,
}

impl LlmModelUsage {
    pub const TYPE: &'static str = "llm_usage";
}

// `usage.py:65`.
/// Usage summary for TTS models.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "TTSModelUsage",
    base = "_BaseModelUsage",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct TtsModelUsage {
    /// The provider name (e.g., 'elevenlabs', 'cartesia').
    pub provider: String,
    /// The model name (e.g., 'eleven_turbo_v2', 'sonic').
    pub model: String,
    /// Input text tokens (for token-based TTS billing, e.g., OpenAI TTS).
    pub input_tokens: u64,
    /// Output audio tokens (for token-based TTS billing, e.g., OpenAI TTS).
    pub output_tokens: u64,
    /// Number of characters synthesized (for character-based TTS billing).
    pub characters_count: u64,
    /// Duration of generated audio in seconds.
    pub audio_duration: f64,
}

impl TtsModelUsage {
    pub const TYPE: &'static str = "tts_usage";
}

// `usage.py:83`.
/// Usage summary for STT models.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    pydantic,
    zero_defaults,
    name = "STTModelUsage",
    base = "_BaseModelUsage",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct SttModelUsage {
    /// The provider name (e.g., 'deepgram', 'assemblyai').
    pub provider: String,
    /// The model name (e.g., 'nova-2', 'best').
    pub model: String,
    /// Input audio tokens (for token-based STT billing).
    pub input_tokens: u64,
    /// Output text tokens (for token-based STT billing).
    pub output_tokens: u64,
    /// Duration of processed audio in seconds.
    pub audio_duration: f64,
}

impl SttModelUsage {
    pub const TYPE: &'static str = "stt_usage";
}

// `usage.py:100`. One counter only: the interruption model is billed per request.
/// Usage summary for interruption detection models.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "InterruptionModelUsage",
    base = "_BaseModelUsage",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct InterruptionModelUsage {
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'adaptive').
    pub model: String,
    /// Total number of requests sent to the interruption detection model.
    #[python(default = "0")]
    pub total_requests: u64,
}

impl InterruptionModelUsage {
    pub const TYPE: &'static str = "interruption_usage";
}

// `usage.py:112`.
/// Usage summary for end-of-turn detection models.
#[derive(Debug, Clone, Default, PartialEq, Eq, Host)]
#[python(
    pydantic,
    name = "EOTModelUsage",
    base = "_BaseModelUsage",
    tag = "type",
    tag_const = TYPE
)]
#[binding(emit, extract, from_items, default)]
pub struct EotModelUsage {
    /// The provider name (e.g., 'livekit').
    pub provider: String,
    /// The model name (e.g., 'turn-detector-v1').
    pub model: String,
    /// Total number of inference requests sent to the EOT model.
    #[python(default = "0")]
    pub total_requests: u64,
}

impl EotModelUsage {
    pub const TYPE: &'static str = "eot_usage";
}

// `usage.py:124`'s `ModelUsage`: a plain PEP-604 union, not a discriminated one.
/// Union type for all model usage types.
#[derive(Debug, Clone, PartialEq, Host)]
#[python(union, name = "ModelUsage")]
#[binding(emit, extract, from_items)]
pub enum ModelUsage {
    Llm(LlmModelUsage),
    Tts(TtsModelUsage),
    Stt(SttModelUsage),
    Interruption(InterruptionModelUsage),
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

// `usage.py:129`. A `@dataclass` in the reference, not a pydantic model: it has no `model_dump`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    dataclass,
    name = "AgentSessionUsage"
)]
pub struct AgentSessionUsage {
    pub model_usage: Vec<ModelUsage>,
}

/// Each dict read by its own `type`: an unknown one raises rather than silently leaving the bill.
#[cfg(feature = "python")]
impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AgentSessionUsage {
    type Error = pyo3::PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::types::PyAnyMethods as _;
        if obj.is_none() {
            return Ok(AgentSessionUsage::default());
        }
        let model_usage = obj.try_iter()?.map(|item| item?.extract::<ModelUsage>()).collect::<pyo3::PyResult<_>>()?;
        Ok(AgentSessionUsage { model_usage })
    }
}
