//! Model requests: LLM and TTS requests, their attempts and adapters, and the usage groups.

use serde::Serialize;

use super::common::{GenAiOperation, OutputType};
use crate::metrics::base::{LlmMetrics, Metadata, RealtimeModelMetrics, TtsMetrics};
use crate::telemetry::decl::{
    Attrs, Content, Current, GenAiContent, Json, Measured, Seconds, SpanValue, span,
};
use crate::telemetry::instruments::GenAiTimeToFirstChunk;
use crate::telemetry::semconv::otel;

/// What a request to a provider records as: the provider says, its caller never guesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestSpan {
    /// `llm_request` or `tts_request`.
    Request,
    /// A fallback adapter: the LLM chain records its own spans; a TTS request is a
    /// `tts_fallback_adapter`.
    FallbackAdapter,
    /// A sentence stream adapter: a TTS request is a `tts_stream_adapter`.
    StreamAdapter,
    /// A session's registered node: it makes its own request, if any, so its caller opens none.
    Node,
}

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
    #[attr(otel::GEN_AI_OUTPUT_TYPE)]
    pub output_type: OutputType,
    #[attr(otel::GEN_AI_SYSTEM_INSTRUCTIONS)]
    pub system_instructions: Option<GenAiContent<Json<Vec<MessagePart>>>>,
    #[attr(otel::GEN_AI_INPUT_MESSAGES)]
    pub input_messages: Option<GenAiContent<Json<Vec<InputMessage>>>>,
    #[attr(otel::GEN_AI_TOOL_DEFINITIONS)]
    pub tool_definitions: Option<GenAiContent<Json<Vec<ToolDefinition>>>>,
    /// The request is a fallback adapter's recovery probe.
    #[attr("lk.fallback.recovery")]
    pub recovery: Option<bool>,
}

#[derive(Attrs)]
pub struct LlmRequestEnd {
    /// The response's metrics, as 1.8.10's `LLMMetrics`.
    #[attr("lk.llm_metrics")]
    pub metrics: Option<Json<LlmMetricsSummary>>,
    #[attrs]
    pub usage: GenAiUsage,
    #[attr(otel::GEN_AI_RESPONSE_ID)]
    pub response_id: Option<String>,
    #[attr(otel::GEN_AI_RESPONSE_MODEL, label)]
    pub response_model: Option<String>,
    #[attr(otel::GEN_AI_RESPONSE_FINISH_REASONS)]
    pub finish_reasons: Vec<FinishReason>,
    #[attr(
        otel::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        metric = GenAiTimeToFirstChunk
    )]
    pub time_to_first_chunk: Measured<Seconds>,
    #[attr(otel::GEN_AI_OUTPUT_MESSAGES)]
    pub output_messages: Option<GenAiContent<Json<Vec<OutputMessage>>>>,
    /// The first chunk's time, for Langfuse.
    #[attr("langfuse.observation.completion_start_time")]
    pub completion_start_time: Option<Json<UtcInstant>>,
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
    #[attr(otel::GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS)]
    pub cache_write_input_tokens: Option<u64>,
    #[attr(otel::GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)]
    pub reasoning_output_tokens: Option<u64>,
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
    /// Provider request ids the span covers, for joining against provider logs; absent when the
    /// provider named none.
    #[attr("lk.provider_request_ids")]
    pub provider_request_ids: Option<Vec<String>>,
}

impl AttemptEnd {
    pub fn new(provider_request_ids: Vec<String>) -> Self {
        Self { provider_request_ids: Some(provider_request_ids).filter(|ids| !ids.is_empty()) }
    }
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
    pub output_type: OutputType,
}

/// The instance that served a fallback adapter's request.
#[derive(Attrs)]
pub struct LlmFallbackChoice {
    /// Label of the instance that served the request.
    #[attr("lk.fallback.label")]
    pub label: String,
    /// Index of that instance in the adapter.
    #[attr("lk.fallback.index")]
    pub index: u32,
    #[attr(otel::GEN_AI_RESPONSE_MODEL)]
    pub response_model: String,
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
    /// The request's metrics, as 1.8.10's `TTSMetrics`.
    #[attr("lk.tts_metrics")]
    pub metrics: Option<Json<TtsMetricsSummary>>,
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
    /// The response's metrics, as 1.8.10's `RealtimeModelMetrics`.
    #[attr("lk.realtime_model_metrics")]
    pub metrics: Json<RealtimeMetricsSummary>,
    #[attr(otel::GEN_AI_RESPONSE_MODEL, label)]
    pub response_model: Option<String>,
    #[attr(otel::GEN_AI_RESPONSE_ID)]
    pub response_id: Option<String>,
    #[attrs]
    pub usage: RealtimeUsage,
    #[attr(
        otel::GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
        metric = GenAiTimeToFirstChunk
    )]
    pub time_to_first_chunk: Measured<Seconds>,
    /// The first chunk's time, for Langfuse.
    #[attr("langfuse.observation.completion_start_time")]
    pub completion_start_time: Option<Json<UtcInstant>>,
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
}

/// How a model response ended.
#[derive(SpanValue, Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    ToolCall,
    /// The response was interrupted.
    Error,
}

/// A message of `gen_ai.input.messages`, as the GenAI conventions' input message schema.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct InputMessage {
    pub role: MessageRole,
    pub parts: Vec<MessagePart>,
}

/// The message of `gen_ai.output.messages`, as the GenAI conventions' output message schema.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct OutputMessage {
    pub role: MessageRole,
    pub parts: Vec<MessagePart>,
    pub finish_reason: FinishReason,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// One part of a GenAI message; media payloads are never recorded, only their kind and URL or
/// transcript.
#[derive(Serialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessagePart {
    Text(TextPart),
    ToolCall(ToolCallPart),
    ToolCallResponse(ToolCallResponsePart),
    Uri(UriPart),
    Blob(BlobPart),
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct TextPart {
    pub content: String,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ToolCallPart {
    pub id: String,
    pub name: String,
    /// The model's arguments: their JSON when they parse, else their text.
    pub arguments: serde_json::Value,
}

#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ToolCallResponsePart {
    pub id: String,
    /// The tool's output: its JSON when it parses, else its text.
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// An image, by URL; inline data is left out.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct UriPart {
    pub modality: Modality,
    pub mime_type: Option<String>,
    pub uri: String,
}

/// Audio, without its samples: `content` is always empty.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct BlobPart {
    pub modality: Modality,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    Audio,
}

/// One tool of `gen_ai.tool.definitions`: name and description; schemas are left out for their
/// size.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// `function`, or a provider tool's name.
    #[serde(rename = "type")]
    pub kind: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// An instant as ISO 8601 UTC with microseconds (`2023-11-14T22:13:20.250000+00:00`), as Python's
/// `datetime.isoformat()` writes it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UtcInstant {
    pub unix_seconds: f64,
}

impl Serialize for UtcInstant {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let micros = (self.unix_seconds * 1e6).round() as i64;
        let at = chrono::DateTime::from_timestamp_micros(micros).unwrap_or_default();
        serializer.collect_str(&at.format("%Y-%m-%dT%H:%M:%S%.6f+00:00"))
    }
}

/// The model a metrics summary is for.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelMetadata {
    pub model_name: Option<String>,
    pub model_provider: Option<String>,
}

impl From<&Metadata> for ModelMetadata {
    fn from(m: &Metadata) -> Self {
        ModelMetadata { model_name: m.model_name.clone(), model_provider: m.model_provider.clone() }
    }
}

/// `lk.llm_metrics`: `LLMMetrics` as 1.8.10's `model_dump_json()` wrote it, which Cloud parses.
/// Unmeasured durations are `-1`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "llm_metrics")]
pub struct LlmMetricsSummary {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub duration: f64,
    pub ttft: f64,
    pub cancelled: bool,
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    pub prompt_cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    pub total_tokens: u64,
    pub tokens_per_second: f64,
    pub speech_id: Option<String>,
    pub metadata: Option<ModelMetadata>,
}

impl From<&LlmMetrics> for LlmMetricsSummary {
    fn from(m: &LlmMetrics) -> Self {
        let t = &m.tokens;
        LlmMetricsSummary {
            label: m.label.clone(),
            request_id: m.request_id.clone(),
            timestamp: m.timestamp,
            duration: m.duration,
            ttft: m.ttft.unwrap_or(-1.0),
            cancelled: m.cancelled,
            completion_tokens: t.output_tokens,
            prompt_tokens: t.input_tokens,
            prompt_cached_tokens: t.cache_read_tokens.unwrap_or(0),
            cache_creation_tokens: t.cache_write_tokens.unwrap_or(0),
            reasoning_tokens: t.output_reasoning_tokens.unwrap_or(0),
            total_tokens: t.total_tokens(),
            tokens_per_second: m.tokens_per_second,
            speech_id: m.speech_id.clone(),
            metadata: m.metadata.as_ref().map(ModelMetadata::from),
        }
    }
}

/// `lk.tts_metrics`: `TTSMetrics` as 1.8.10's `model_dump_json()` wrote it. An unmeasured `ttfb`
/// is `-1`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "tts_metrics")]
pub struct TtsMetricsSummary {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub ttfb: f64,
    pub duration: f64,
    pub audio_duration: f64,
    pub cancelled: bool,
    pub characters_count: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub streamed: bool,
    pub acquire_time: f64,
    pub connection_reused: bool,
    pub segment_id: Option<String>,
    pub speech_id: Option<String>,
    pub metadata: Option<ModelMetadata>,
}

impl From<&TtsMetrics> for TtsMetricsSummary {
    fn from(m: &TtsMetrics) -> Self {
        TtsMetricsSummary {
            label: m.label.clone(),
            request_id: m.request_id.clone(),
            timestamp: m.timestamp,
            ttfb: m.ttfb.unwrap_or(-1.0),
            duration: m.duration,
            audio_duration: m.audio_duration,
            cancelled: m.cancelled,
            characters_count: m.characters_count,
            input_tokens: m.tokens.input_tokens,
            output_tokens: m.tokens.output_tokens,
            streamed: m.streamed,
            acquire_time: m.acquire_time.unwrap_or(0.0),
            connection_reused: m.connection_reused,
            segment_id: m.segment_id.clone(),
            speech_id: m.speech_id.clone(),
            metadata: m.metadata.as_ref().map(ModelMetadata::from),
        }
    }
}

/// `lk.realtime_model_metrics`: `RealtimeModelMetrics` as 1.8.10's `model_dump_json()` wrote it,
/// its token details nested. An unmeasured `ttft` is `-1`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename = "realtime_model_metrics")]
pub struct RealtimeMetricsSummary {
    pub label: String,
    pub request_id: String,
    pub timestamp: f64,
    pub duration: f64,
    pub session_duration: f64,
    pub ttft: f64,
    pub cancelled: bool,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub tokens_per_second: f64,
    pub input_token_details: InputTokenDetails,
    pub output_token_details: OutputTokenDetails,
    pub acquire_time: f64,
    pub connection_reused: bool,
    pub metadata: Option<ModelMetadata>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InputTokenDetails {
    pub audio_tokens: u64,
    pub text_tokens: u64,
    pub image_tokens: u64,
    pub cached_tokens: u64,
    /// Absent unless the provider split its cached tokens by modality.
    pub cached_tokens_details: Option<CachedTokenDetails>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CachedTokenDetails {
    pub audio_tokens: u64,
    pub text_tokens: u64,
    pub image_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OutputTokenDetails {
    pub text_tokens: u64,
    pub audio_tokens: u64,
    pub image_tokens: u64,
}

impl From<&RealtimeModelMetrics> for RealtimeMetricsSummary {
    fn from(m: &RealtimeModelMetrics) -> Self {
        let t = &m.tokens;
        let n = |v: Option<u64>| v.unwrap_or(0);
        let cached_split = t.cache_read_text_tokens.is_some()
            || t.cache_read_audio_tokens.is_some()
            || t.cache_read_image_tokens.is_some();
        RealtimeMetricsSummary {
            label: m.label.clone(),
            request_id: m.request_id.clone(),
            timestamp: m.timestamp,
            duration: m.duration,
            session_duration: m.connection_duration.unwrap_or(0.0),
            ttft: m.ttft.unwrap_or(-1.0),
            cancelled: m.cancelled,
            input_tokens: t.input_tokens,
            output_tokens: t.output_tokens,
            total_tokens: t.total_tokens(),
            tokens_per_second: m.tokens_per_second,
            input_token_details: InputTokenDetails {
                audio_tokens: n(t.input_audio_tokens),
                text_tokens: n(t.input_text_tokens),
                image_tokens: n(t.input_image_tokens),
                cached_tokens: n(t.cache_read_tokens),
                cached_tokens_details: cached_split.then(|| CachedTokenDetails {
                    audio_tokens: n(t.cache_read_audio_tokens),
                    text_tokens: n(t.cache_read_text_tokens),
                    image_tokens: n(t.cache_read_image_tokens),
                }),
            },
            output_token_details: OutputTokenDetails {
                text_tokens: n(t.output_text_tokens),
                audio_tokens: n(t.output_audio_tokens),
                image_tokens: n(t.output_image_tokens),
            },
            acquire_time: m.acquire_time.unwrap_or(0.0),
            connection_reused: m.connection_reused,
            metadata: m.metadata.as_ref().map(ModelMetadata::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a date or fraction misformatted for Langfuse.
    #[test]
    fn instants_are_iso_8601_utc() {
        let iso = |unix_seconds| serde_json::to_string(&UtcInstant { unix_seconds }).unwrap();
        assert_eq!(iso(0.0), "\"1970-01-01T00:00:00.000000+00:00\"");
        assert_eq!(iso(1_700_000_000.25), "\"2023-11-14T22:13:20.250000+00:00\"");
        assert_eq!(iso(951_782_400.000_001), "\"2000-02-29T00:00:00.000001+00:00\"");
        assert_eq!(iso(-0.5), "\"1969-12-31T23:59:59.500000+00:00\"");
    }

    // Catches: the `type` tag lost or moved, or an unmeasured ttft written as null.
    #[test]
    fn metrics_summaries_keep_the_reference_shape() {
        let m = LlmMetrics { ttft: None, metadata: None, ..crate::metrics::testkit::llm() };
        let v = serde_json::to_value(LlmMetricsSummary::from(&m)).unwrap();
        assert_eq!(v.as_object().unwrap().keys().next().map(String::as_str), Some("type"));
        assert_eq!((v["type"].as_str(), v["ttft"].as_f64()), (Some("llm_metrics"), Some(-1.0)));
        assert!(v["metadata"].is_null());
    }
}
