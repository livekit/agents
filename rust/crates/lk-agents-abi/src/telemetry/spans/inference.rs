//! Model requests: LLM and TTS requests, their attempts and adapters, and the usage groups.

use super::common::GenAiOperation;
use crate::telemetry::decl::{
    Attrs, Content, Current, GenAiContent, Json, Measured, Seconds, span,
};
use crate::telemetry::semconv::otel;

/// One request to an LLM provider, retries included, wherever it is made.
#[span(
    name = "llm_request",
    parent = Current,
    kind = Client,
    end = LlmRequestEnd,
    outlives_parent = "a recovery probe (lk.fallback.recovery) runs in the background past the request"
)]
#[derive(Attrs)]
pub struct LlmRequest {
    /// `chat`.
    #[attr(otel::GEN_AI_OPERATION_NAME, label)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_PROVIDER_NAME, label)]
    pub provider: String,
    #[attr(otel::GEN_AI_REQUEST_MODEL, label)]
    pub model: String,
    #[attr(otel::GEN_AI_REQUEST_STREAM)]
    pub stream: bool,
    /// `text`.
    #[attr(otel::GEN_AI_OUTPUT_TYPE)]
    pub output_type: String,
    #[attr(otel::GEN_AI_SYSTEM_INSTRUCTIONS)]
    pub system_instructions: Option<GenAiContent<Json<serde_json::Value>>>,
    #[attr(otel::GEN_AI_INPUT_MESSAGES)]
    pub input_messages: Option<GenAiContent<Json<serde_json::Value>>>,
    #[attr(otel::GEN_AI_TOOL_DEFINITIONS)]
    pub tool_definitions: Option<GenAiContent<Json<serde_json::Value>>>,
    #[attrs]
    pub input: Option<InputDelta>,
    /// The request is a fallback adapter's recovery probe.
    #[attr("lk.fallback.recovery")]
    pub recovery: Option<bool>,
}

#[derive(Attrs)]
pub struct LlmRequestEnd {
    /// The response's metrics: `ttft`, `prompt_tokens`, `prompt_cached_tokens`,
    /// `completion_tokens`, `duration`, ...
    #[attr("lk.llm_metrics")]
    pub legacy: Option<Json<serde_json::Value>>,
    #[attrs]
    pub usage: GenAiUsage,
    #[attr(otel::GEN_AI_RESPONSE_ID)]
    pub response_id: Option<String>,
    #[attr(otel::GEN_AI_RESPONSE_MODEL, label)]
    pub response_model: Option<String>,
    /// `stop`, `tool_call` or `error`.
    #[attr(otel::GEN_AI_RESPONSE_FINISH_REASONS)]
    pub finish_reasons: Vec<String>,
    #[attr(
        otel::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        metric = "gen_ai.client.operation.time_to_first_chunk"
    )]
    pub time_to_first_chunk: Measured<Seconds>,
    #[attr(otel::GEN_AI_OUTPUT_MESSAGES)]
    pub output_messages: Option<GenAiContent<Json<serde_json::Value>>>,
    /// The first chunk's time, as a JSON-quoted ISO 8601 string, for Langfuse.
    #[attr("langfuse.observation.completion_start_time")]
    pub completion_start_time: Option<String>,
}

/// Token usage of an LLM response.
#[derive(Attrs)]
pub struct GenAiUsage {
    #[attr(otel::GEN_AI_USAGE_INPUT_TOKENS)]
    pub input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_OUTPUT_TOKENS)]
    pub output_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)]
    pub cache_read_input_tokens: Option<u64>,
    /// Cached input tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.input_cached_tokens")]
    pub input_cached_tokens: Option<u64>,
    #[attr(otel::GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS)]
    pub cache_write_input_tokens: Option<u64>,
    #[attr(otel::GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)]
    pub reasoning_output_tokens: Option<u64>,
    /// Reasoning tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.reasoning_tokens")]
    pub reasoning_tokens: Option<u64>,
}

/// A request's input as a delta over an earlier span's, with message capture.
#[derive(Attrs)]
pub struct InputDelta {
    /// The span's input record is a delta over its base span.
    #[attr("lk.input.delta")]
    pub delta: bool,
    /// Span whose input record this one continues.
    #[attr("lk.input.base_span_id")]
    pub base_span_id: String,
    /// Items of the base record not carried into this one.
    #[attr("lk.input.dropped_from_base")]
    pub dropped_from_base: u32,
}

/// One attempt of an LLM request.
#[span(name = "llm_request_run", parent = LlmRequest, end = AttemptEnd)]
#[derive(Attrs)]
pub struct LlmRequestRun {
    #[attrs]
    pub attempt: Attempt,
}

/// The provider, model and retry of one attempt.
#[derive(Attrs)]
pub struct Attempt {
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: String,
    #[attr(otel::GEN_AI_REQUEST_MODEL)]
    pub model: String,
    /// Retries before this attempt.
    #[attr("lk.retry_count")]
    pub retry_count: u32,
}

#[derive(Attrs)]
pub struct AttemptEnd {
    #[attr("lk.provider_request_ids")]
    pub provider_request_ids: Vec<String>,
}

/// An LLM fallback adapter; one `llm_request` child per instance tried.
#[span(name = "llm_fallback_adapter", parent = Current, end = Option<LlmFallbackChoice>)]
#[derive(Attrs)]
pub struct LlmFallbackAdapter {
    /// The next instance's provider and model, and the retry count.
    #[attrs]
    pub attempt: Attempt,
    #[attr(otel::GEN_AI_REQUEST_STREAM)]
    pub stream: bool,
    #[attr(otel::GEN_AI_OUTPUT_TYPE)]
    pub output_type: String,
}

/// The instance that served a fallback adapter's request.
#[derive(Attrs)]
pub struct FallbackChoice {
    /// Label of the instance that served the request.
    #[attr("lk.fallback.label")]
    pub label: String,
    /// Index of that instance in the adapter.
    #[attr("lk.fallback.index")]
    pub index: u32,
    #[attr(otel::GEN_AI_RESPONSE_MODEL)]
    pub response_model: String,
}

#[derive(Attrs)]
pub struct LlmFallbackChoice {
    #[attrs]
    pub choice: FallbackChoice,
    /// The provider that served the request.
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: String,
}

/// The TTS a request is for.
#[derive(Attrs)]
pub struct TtsTarget {
    /// The TTS streams its input.
    #[attr("lk.tts.streaming")]
    pub streaming: bool,
    /// Label of the TTS instance.
    #[attr("lk.tts.label")]
    pub label: String,
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: String,
    #[attr(otel::GEN_AI_REQUEST_MODEL)]
    pub model: String,
}

/// One request to a TTS provider, retries included, wherever it is made.
#[span(
    name = "tts_request",
    parent = Current,
    kind = Client,
    end = TtsRequestEnd,
    outlives_parent = "a recovery probe (lk.fallback.recovery) runs in the background past the request"
)]
#[derive(Attrs)]
pub struct TtsRequest {
    #[attrs]
    pub target: TtsTarget,
    #[attr("lk.fallback.recovery")]
    pub recovery: Option<bool>,
}

#[derive(Attrs)]
pub struct TtsRequestEnd {
    /// The synthesized text, after success.
    #[attr("lk.pii.input_text")]
    pub input_text: Option<Content<String>>,
    /// The request's metrics: `ttfb`, `characters_count`, `audio_duration`, ...
    #[attr("lk.tts_metrics")]
    pub legacy: Option<Json<serde_json::Value>>,
}

/// One attempt of a TTS request, or of an adapter's.
#[span(
    name = "tts_request_run",
    parent = TtsRequest | TtsFallbackAdapter | TtsStreamAdapter,
    end = AttemptEnd
)]
#[derive(Attrs)]
pub struct TtsRequestRun {
    #[attrs]
    pub attempt: Attempt,
}

/// A request to a TTS fallback adapter, in place of its `tts_request`; under each attempt, one
/// `tts_request` per instance tried.
#[span(name = "tts_fallback_adapter", parent = Current, kind = Client, end = TtsRequestEnd)]
#[derive(Attrs)]
pub struct TtsFallbackAdapter {
    #[attrs]
    pub target: TtsTarget,
}

/// A request to a stream adapter around a non-streaming TTS, in place of its `tts_request`.
#[span(name = "tts_stream_adapter", parent = Current, kind = Client, end = TtsRequestEnd)]
#[derive(Attrs)]
pub struct TtsStreamAdapter {
    #[attrs]
    pub target: TtsTarget,
}

/// A realtime response's metrics, on `realtime_inference` or a late `realtime_metrics`.
#[derive(Attrs)]
pub struct RealtimeResponse {
    /// The response's metrics; `ttft` is `-1` when unmeasured.
    #[attr("lk.realtime_model_metrics")]
    pub legacy: Json<serde_json::Value>,
    #[attr(otel::GEN_AI_RESPONSE_MODEL, label)]
    pub response_model: Option<String>,
    #[attr(otel::GEN_AI_RESPONSE_ID)]
    pub response_id: Option<String>,
    #[attrs]
    pub usage: RealtimeUsage,
    #[attr(
        otel::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        metric = "gen_ai.client.operation.time_to_first_chunk"
    )]
    pub time_to_first_chunk: Measured<Seconds>,
    #[attr("langfuse.observation.completion_start_time")]
    pub completion_start_time: Option<String>,
}

/// Token usage of a realtime response, split by modality.
#[derive(Attrs)]
pub struct RealtimeUsage {
    #[attr(otel::GEN_AI_USAGE_INPUT_TOKENS)]
    pub input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_OUTPUT_TOKENS)]
    pub output_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_TEXT_INPUT_TOKENS)]
    pub text_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_AUDIO_INPUT_TOKENS)]
    pub audio_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_IMAGE_INPUT_TOKENS)]
    pub image_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_TEXT_OUTPUT_TOKENS)]
    pub text_output_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS)]
    pub audio_output_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)]
    pub cache_read_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS)]
    pub text_cache_read_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS)]
    pub audio_cache_read_input_tokens: u64,
    #[attr(otel::GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS)]
    pub image_cache_read_input_tokens: u64,
    /// Text input tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.input_text_tokens")]
    pub input_text_tokens: u64,
    /// Audio input tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.input_audio_tokens")]
    pub input_audio_tokens: u64,
    #[attr("gen_ai.usage.input_cached_tokens")]
    pub input_cached_tokens: u64,
    /// Text output tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.output_text_tokens")]
    pub output_text_tokens: u64,
    /// Audio output tokens, under Langfuse's spelling.
    #[attr("gen_ai.usage.output_audio_tokens")]
    pub output_audio_tokens: u64,
}
