//! The agent's turn: one speech, its nodes, realtime inference and tool calls.

use serde::Serialize;

use super::common::{GenAiOperation, OutputType, ParticipantAttrs, ToolType};
use super::inference::RealtimeResponse;
use super::session::AgentSession;
use super::text::AgentRun;
use crate::telemetry::decl::{
    Attrs, Content, GenAiContent, Json, Measured, Seconds, SpanValue, event, span,
};
use crate::telemetry::semconv::otel;

/// One speech, from its first generation to done.
#[span(
    name = "agent_turn",
    parent = AgentSession,
    end = AgentTurnEnd,
    events = [Generation, PreemptiveGenerationDiscarded],
    handle = Manual
)]
#[derive(Attrs)]
pub struct AgentTurn {
    /// Id of the speech, unique across sessions and processes.
    #[attr("lk.speech_id")]
    pub speech_id: String,
    /// `invoke_agent`.
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_AGENT_NAME)]
    pub agent_name: String,
    /// The agent's instructions, for a pipeline reply.
    #[attr("lk.pii.instructions")]
    pub instructions: Option<Content<String>>,
    /// The user input replied to, for a pipeline reply.
    #[attr("lk.pii.user_input")]
    pub user_input: Option<Content<String>>,
    #[attrs]
    pub participant: Option<ParticipantAttrs>,
}

#[derive(Attrs)]
pub struct AgentTurnEnd {
    /// Id of the last generation.
    #[attr("lk.generation_id")]
    pub generation_id: Option<String>,
    /// Generations (steps) of the speech.
    #[attr("lk.generation_count")]
    pub generation_count: u32,
    /// Seconds the speech waited in the queue before generation was authorized.
    #[attr("lk.speech.queue_wait")]
    pub queue_wait: Option<Seconds>,
    /// The speech was interrupted.
    #[attr("lk.interrupted")]
    pub interrupted: bool,
    /// What interrupted the speech.
    #[attr("lk.interruption.source")]
    pub interruption_source: Option<InterruptionSource>,
    /// Seconds of the speech played out when it was interrupted.
    #[attr("lk.playout.position")]
    pub playout_position: Option<Seconds>,
    /// The agent's response text.
    #[attr("lk.pii.response.text")]
    pub response_text: Option<Content<String>>,
    /// The function calls of a realtime response.
    #[attr("lk.pii.response.function_calls")]
    pub function_calls: Option<Content<Json<Vec<ResponseFunctionCall>>>>,
    /// Seconds from the user's end of speech to the agent's first audio.
    #[attr("lk.e2e_latency")]
    pub e2e_latency: Measured<Seconds>,
    #[attr("lk.end_of_turn_delay")]
    pub end_of_turn_delay: Option<Seconds>,
    #[attr("lk.transcription_delay")]
    pub transcription_delay: Option<Seconds>,
    /// Seconds the `on_user_turn_completed` hook took.
    #[attr("lk.on_user_turn_completed_delay")]
    pub on_user_turn_completed_delay: Option<Seconds>,
}

#[derive(SpanValue, Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptionSource {
    AudioActivity,
    UserTurn,
    Programmatic,
}

/// One generation step started, at the time it started.
#[event(name = "generation")]
#[derive(Attrs)]
pub struct Generation {
    #[attr("lk.generation_id")]
    pub generation_id: String,
    /// The generation this step continues.
    #[attr("lk.parent_generation_id")]
    pub parent_generation_id: Option<String>,
}

/// A preemptive generation for the speech was discarded.
#[event(name = "preemptive_generation_discarded")]
#[derive(Attrs)]
pub struct PreemptiveGenerationDiscarded {
    #[attr("lk.speech_id")]
    pub speech_id: String,
}

/// The agent speaking, a direct child of its turn.
#[span(name = "agent_speaking", parent = AgentTurn, handle = Manual)]
#[derive(Attrs)]
pub struct AgentSpeaking {
    #[attrs]
    pub participant: Option<ParticipantAttrs>,
}

/// The agent's LLM node for one step.
#[span(name = "llm_node", parent = AgentTurn, end = LlmNodeEnd)]
#[derive(Attrs)]
pub struct LlmNode {
    /// The chat context sent.
    #[attr("lk.pii.chat_ctx")]
    pub chat_ctx: Content<Json<ChatContextDict>>,
    /// Function tools offered.
    #[attr("lk.function_tools")]
    pub function_tools: Vec<String>,
    /// Provider tools offered.
    #[attr("lk.provider_tools")]
    pub provider_tools: Vec<String>,
    /// Toolsets offered.
    #[attr("lk.tool_sets")]
    pub tool_sets: Vec<String>,
}

// What the node yielded. The model and provider are its requests' (`llm_request`): a node makes
// one only when it runs the default.
#[derive(Attrs)]
pub struct LlmNodeEnd {
    #[attr("lk.pii.response.text")]
    pub response_text: Content<String>,
    #[attr("lk.pii.response.function_calls")]
    pub function_calls: Content<Json<Vec<ResponseFunctionCall>>>,
    /// Seconds to the first token, measured at the node.
    #[attr("lk.response.ttft")]
    pub ttft: Measured<Seconds>,
}

/// The agent's TTS node for one step.
// The model and provider are its requests' (`tts_request`): a node makes one only when it runs
// the default.
#[span(name = "tts_node", parent = AgentTurn, end = TtsNodeEnd)]
#[derive(Attrs)]
pub struct TtsNode;

#[derive(Attrs)]
pub struct TtsNodeEnd {
    /// Seconds from the text anchor to the first audio frame.
    #[attr("lk.response.ttfb")]
    pub ttfb: Measured<Seconds>,
}

/// One realtime model response; usage and timing on the span itself.
#[span(
    name = "realtime_inference",
    parent = AgentTurn,
    kind = Client,
    end = Option<RealtimeResponse>
)]
#[derive(Attrs)]
pub struct RealtimeInference {
    /// `generate_content`.
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
}

/// A realtime response's metrics that arrived after its inference ended.
#[span(
    name = "realtime_metrics",
    parent = RealtimeInference,
    outlives_parent = "the metrics arrive after the inference ended"
)]
#[derive(Attrs)]
pub struct RealtimeMetrics {
    /// `generate_content`.
    #[attr(otel::GEN_AI_OPERATION_NAME, label)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_PROVIDER_NAME, label)]
    pub provider: String,
    #[attr(otel::GEN_AI_REQUEST_MODEL, label)]
    pub model: String,
    #[attrs]
    pub response: RealtimeResponse,
}

/// One tool call.
#[span(name = "function_tool", parent = AgentTurn | AgentRun, end = FunctionToolEnd)]
#[derive(Attrs)]
pub struct FunctionTool {
    /// The tool call id.
    #[attr("lk.function_tool.id")]
    pub id: String,
    /// The tool's name.
    #[attr("lk.function_tool.name")]
    pub name: String,
    /// The call's arguments, as the model wrote them.
    #[attr("lk.pii.function_tool.arguments")]
    pub arguments: Content<String>,
    /// `execute_tool`.
    #[attr(otel::GEN_AI_OPERATION_NAME)]
    pub operation: GenAiOperation,
    #[attr(otel::GEN_AI_TOOL_NAME)]
    pub tool_name: String,
    #[attr(otel::GEN_AI_TOOL_TYPE)]
    pub tool_type: ToolType,
    #[attr(otel::GEN_AI_TOOL_CALL_ID)]
    pub call_id: String,
    #[attr(otel::GEN_AI_AGENT_NAME)]
    pub agent_name: String,
    #[attr(otel::GEN_AI_TOOL_DESCRIPTION)]
    pub description: Option<GenAiContent<String>>,
    #[attr(otel::GEN_AI_TOOL_CALL_ARGUMENTS)]
    pub call_arguments: Option<GenAiContent<String>>,
}

#[derive(Attrs)]
pub struct FunctionToolEnd {
    /// The tool's output.
    #[attr("lk.pii.function_tool.output")]
    pub output: Content<String>,
    /// The tool raised an error; the span records it as a `tool_error` failure.
    #[attr("lk.function_tool.is_error")]
    pub is_error: bool,
    #[attr(otel::GEN_AI_TOOL_CALL_RESULT)]
    pub result: Option<GenAiContent<String>>,
}

/// A function call a response made.
#[derive(Serialize, Debug, Clone, PartialEq)]
pub struct ResponseFunctionCall {
    pub call_id: String,
    pub name: String,
    /// The arguments, as the model wrote them.
    pub arguments: String,
}

/// A chat context in `ChatContext.to_dict()`'s form (`lk_agents_core::llm::chat_dict`).
#[derive(Serialize, Debug, Clone, PartialEq, Default)]
#[serde(transparent)]
pub struct ChatContextDict(pub serde_json::Map<String, serde_json::Value>);
