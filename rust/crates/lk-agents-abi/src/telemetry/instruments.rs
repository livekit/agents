//! The metric instruments. A span end field with `metric = Instrument` records one when its span
//! ends; the others are recorded where their value is computed.

use super::decl::{Attrs, BoolString, Label, SpanValue, metric};
use super::errors::ErrorType;
use super::semconv::otel;
use super::spans::common::GenAiOperation;
use super::spans::session::BlockingSeverity;

/// A duration histogram's bucket bounds, in seconds: the GenAI semantic conventions' advice.
pub const DURATION_BOUNDS: &[f64] =
    &[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92];

/// A token count histogram's bucket bounds: the GenAI semantic conventions' advice.
pub const TOKEN_BOUNDS: &[f64] = &[
    1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0, 4194304.0,
    16777216.0, 67108864.0,
];

/// The model label keys, which the turn spans' `label = ..` fields take too.
pub const MODEL_PROVIDER: &str = "model_provider";
pub const MODEL_NAME: &str = "model_name";

/// The model a measurement is about; what is not known is omitted.
#[derive(Attrs, Default)]
pub struct ModelLabels {
    /// The model's provider.
    #[attr(MODEL_PROVIDER)]
    pub provider: Option<Label>,
    /// The model's name.
    #[attr(MODEL_NAME)]
    pub model: Option<Label>,
}

impl ModelLabels {
    /// An empty provider or name is not known.
    pub fn of(provider: Option<&str>, model: Option<&str>) -> Self {
        let label = |v: Option<&str>| v.filter(|v| !v.is_empty()).map(|v| Label(v.to_owned()));
        ModelLabels { provider: label(provider), model: label(model) }
    }
}

/// The GenAI operation a measurement is about.
#[derive(Attrs)]
pub struct GenAiLabels {
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_PROVIDER_NAME)]
    pub provider: Label,
    #[attr(otel::GEN_AI_REQUEST_MODEL)]
    pub model: Label,
    #[attr(otel::GEN_AI_RESPONSE_MODEL)]
    pub response_model: Option<Label>,
}

/// End-to-end latency of an answered turn, labeled with its LLM.
#[metric(
    name = "lk.agents.turn.e2e_latency",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct TurnE2eLatency {
    #[attrs]
    pub model: ModelLabels,
}

/// Time to the first LLM token of a turn.
#[metric(name = "lk.agents.turn.llm_ttft", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct TurnLlmTtft {
    #[attrs]
    pub model: ModelLabels,
}

/// Time to the first TTS audio of a turn.
#[metric(name = "lk.agents.turn.tts_ttfb", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct TurnTtsTtfb {
    #[attrs]
    pub model: ModelLabels,
}

/// Delay from the end of speech to the final transcript, labeled with the STT.
#[metric(
    name = "lk.agents.turn.transcription_delay",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct TurnTranscriptionDelay {
    #[attrs]
    pub model: ModelLabels,
}

/// Delay from the end of speech to the turn decision, labeled with the STT.
#[metric(
    name = "lk.agents.turn.end_of_turn_delay",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct TurnEndOfTurnDelay {
    #[attrs]
    pub model: ModelLabels,
}

/// Duration of the `on_user_turn_completed` hook, labeled with the STT.
#[metric(
    name = "lk.agents.turn.on_user_turn_completed_delay",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct TurnOnUserTurnCompletedDelay {
    #[attrs]
    pub model: ModelLabels,
}

/// LLM input tokens.
#[metric(name = "lk.agents.usage.llm_input_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// LLM cached input tokens.
#[metric(name = "lk.agents.usage.llm_input_cached_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputCachedTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// LLM output tokens.
#[metric(name = "lk.agents.usage.llm_output_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmOutputTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Realtime-model audio input tokens.
#[metric(name = "lk.agents.usage.llm_input_audio_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputAudioTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Realtime-model text input tokens.
#[metric(name = "lk.agents.usage.llm_input_text_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmInputTextTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Realtime-model audio output tokens.
#[metric(name = "lk.agents.usage.llm_output_audio_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmOutputAudioTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Realtime-model text output tokens.
#[metric(name = "lk.agents.usage.llm_output_text_tokens", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct LlmOutputTextTokens {
    #[attrs]
    pub model: ModelLabels,
}

/// Realtime-model session seconds.
#[metric(name = "lk.agents.usage.llm_session_duration", kind = Counter, unit = "s")]
#[derive(Attrs)]
pub struct LlmSessionDuration {
    #[attrs]
    pub model: ModelLabels,
}

/// Seconds of audio synthesized.
#[metric(name = "lk.agents.usage.tts_audio_duration", kind = Counter, unit = "s")]
#[derive(Attrs)]
pub struct TtsAudioDuration {
    #[attrs]
    pub model: ModelLabels,
}

/// Characters synthesized.
#[metric(name = "lk.agents.usage.tts_characters", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct TtsCharacters {
    #[attrs]
    pub model: ModelLabels,
}

/// Seconds of audio transcribed.
#[metric(name = "lk.agents.usage.stt_audio_duration", kind = Counter, unit = "s")]
#[derive(Attrs)]
pub struct SttAudioDuration {
    #[attrs]
    pub model: ModelLabels,
}

/// Interruption-detector requests.
#[metric(name = "lk.agents.usage.interruption_num_requests", kind = Counter, unit = "")]
#[derive(Attrs)]
pub struct InterruptionNumRequests {
    #[attrs]
    pub model: ModelLabels,
}

/// Seconds to acquire a provider WebSocket connection.
#[metric(
    name = "lk.agents.connection.acquire_time",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct ConnectionAcquireTime {
    #[attrs]
    pub model: ModelLabels,
    /// A pooled connection was reused.
    #[attr("connection_reused")]
    pub reused: BoolString,
}

/// Duration of an event-loop stall.
#[metric(
    name = "lk.agents.event_loop.blocked_duration",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct EventLoopBlockedDuration {
    /// How severe the stall was.
    #[attr("severity")]
    pub severity: BlockingSeverity,
}

/// Seconds of audio an input channel dropped to keep its consumer within its policy.
#[metric(name = "lk.agents.audio.dropped", kind = Counter, unit = "s")]
#[derive(Attrs)]
pub struct AudioDropped {
    /// What the channel keeps when its consumer falls behind.
    #[attr("policy")]
    pub policy: AudioPolicy,
    /// The consumer the channel feeds.
    #[attr("consumer")]
    pub consumer: AudioConsumer,
}

/// How long a consumer took to drain the pre-connect audio replayed ahead of the live input:
/// longer than that audio lasted, it falls behind the call.
#[metric(name = "lk.agents.audio.catch_up_time", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct AudioCatchUp {
    /// The consumer the channel feeds.
    #[attr("consumer")]
    pub consumer: AudioConsumer,
}

/// Seconds of a duplex model's output audio queued ahead of the session when it takes the next
/// frame: how far the session runs behind the model's output.
#[metric(name = "lk.agents.duplex.output_lag", kind = Histogram, unit = "s", bounds = DURATION_BOUNDS)]
#[derive(Attrs)]
pub struct DuplexOutputLag {
    #[attrs]
    pub model: ModelLabels,
}

/// What an audio channel keeps when its consumer falls behind.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioPolicy {
    /// Everything up to its bound, the oldest dropped beyond it.
    CatchUp,
    /// Only the newest audio, once the consumer lags.
    Now,
}

/// The consumer an audio channel feeds.
#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioConsumer {
    /// The session's microphone, read from the room.
    RoomInput,
    /// A realtime model's input audio.
    Realtime,
    /// A recognition stream's input audio.
    Stt,
    /// A voice activity detector's input audio.
    Vad,
    /// The adaptive interruption detector's input audio.
    Interruption,
    /// The audio end-of-turn model's input audio.
    EndOfTurn,
    /// A custom `stt_node`'s input audio.
    SttNode,
    /// The session recording's microphone channel.
    Recording,
    /// A duplex model's input audio.
    Duplex,
}

/// Input tokens of each LLM response, one measurement per modality.
#[metric(name = otel::GEN_AI_CLIENT_INFERENCE_USAGE_INPUT_TOKENS, kind = Counter, unit = "{token}")]
#[derive(Attrs)]
pub struct GenAiInputTokens {
    #[attrs]
    pub operation: GenAiLabels,
    #[attr(otel::GEN_AI_TOKEN_MODALITY)]
    pub modality: TokenModality,
}

/// Output tokens of each LLM response, one measurement per modality.
#[metric(name = otel::GEN_AI_CLIENT_INFERENCE_USAGE_OUTPUT_TOKENS, kind = Counter, unit = "{token}")]
#[derive(Attrs)]
pub struct GenAiOutputTokens {
    #[attrs]
    pub operation: GenAiLabels,
    #[attr(otel::GEN_AI_TOKEN_MODALITY)]
    pub modality: TokenModality,
}

#[derive(super::decl::SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenModality {
    Text,
    Image,
    Audio,
    /// Tokens the provider did not break down by modality.
    Unknown,
}

/// Duration of each LLM and realtime response. Declared here: weaver cannot import it from the
/// GenAI registry while semconv v1.44.0 keeps an excluded deprecated copy.
#[metric(
    name = "gen_ai.client.operation.duration",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct GenAiOperationDuration {
    #[attrs]
    pub operation: GenAiLabels,
}

/// Time to the first chunk of a streaming response. Deprecated upstream; declared as emitted.
#[metric(
    name = "gen_ai.client.operation.time_to_first_chunk",
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct GenAiTimeToFirstChunk {
    #[attrs]
    pub operation: GenAiLabels,
}

/// Duration of each tool call.
#[metric(
    name = otel::GEN_AI_EXECUTE_TOOL_DURATION,
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct ExecuteToolDuration {
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_TOOL_NAME)]
    pub tool: Label,
    #[attr(otel::ERROR_TYPE)]
    pub error_type: Option<ErrorType>,
}

/// Duration of each agent turn.
#[metric(
    name = otel::GEN_AI_INVOKE_AGENT_DURATION,
    kind = Histogram,
    unit = "s",
    bounds = DURATION_BOUNDS
)]
#[derive(Attrs)]
pub struct InvokeAgentDuration {
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_AGENT_NAME)]
    pub agent: Label,
}
