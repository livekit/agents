//! `UsageSummary`: a `@dataclass`, not a pydantic model, as in the reference.

use lk_agents_host_derive::Host;

/// .. deprecated::
///     Use :class:`LLMModelUsage`, :class:`TTSModelUsage`, or :class:`STTModelUsage` instead.
//
// The `///` is the reference's docstring only. `llm_input_tokens`/`llm_output_tokens` alias the
// `llm_prompt_*`/`llm_completion_*` fields; on the Rust side they are `input_tokens`/`output_tokens`.
#[derive(Debug, Clone, Default, PartialEq, Host)]
#[python(
    dataclass,
    zero_defaults,
    module = "usage_collector",
    path = "livekit.agents.metrics.usage_collector",
    name = "UsageSummary",
    post_init = "usage_summary_post_init",
    property(
        name = "llm_input_tokens",
        ret = "int",
        hook = "usage_summary_llm_input_tokens",
        setter = "usage_summary_set_llm_input_tokens",
        doc = "Alias for ``llm_prompt_tokens``: prompt and input are one counter under two names."
    ),
    property(
        name = "llm_output_tokens",
        ret = "int",
        hook = "usage_summary_llm_output_tokens",
        setter = "usage_summary_set_llm_output_tokens",
        doc = "Alias for ``llm_completion_tokens``."
    )
)]
#[binding(emit)]
pub struct UsageSummary {
    pub llm_prompt_tokens: u64,
    pub llm_prompt_cached_tokens: u64,
    pub llm_input_audio_tokens: u64,
    pub llm_input_cached_audio_tokens: u64,
    pub llm_input_text_tokens: u64,
    pub llm_input_cached_text_tokens: u64,
    pub llm_input_image_tokens: u64,
    pub llm_input_cached_image_tokens: u64,
    pub llm_completion_tokens: u64,
    pub llm_output_audio_tokens: u64,
    // Deprecated upstream on the metric side, still summed here, deliberately.
    pub llm_output_image_tokens: u64,
    pub llm_output_text_tokens: u64,
    pub tts_characters_count: u64,
    pub tts_audio_duration: f64,
    pub stt_audio_duration: f64,
}
