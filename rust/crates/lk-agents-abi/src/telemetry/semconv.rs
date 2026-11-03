//! Keys and names owned by the registries the agents registry depends on. A declaration names
//! these constants; it never redeclares their keys as literals (`lk-agents-codegen --check`).
//!
//! Hand-maintained mirror of the pinned registries (the pins are in `lk-agents-codegen`'s
//! telemetry module). Once livekit/protocol publishes `otel/model/shared` with generated Rust
//! constants, `protocol` becomes that generated file, and `otel` a weaver template over the
//! pinned OpenTelemetry and GenAI registries; the declarations do not change.

/// livekit/protocol `otel/model/shared`: correlation keys shared with the client SDKs.
pub mod protocol {
    pub const LK_ROOM_ID: &str = "lk.room_id";
    pub const LK_JOB_ID: &str = "lk.job_id";
    pub const LK_PARTICIPANT_ID: &str = "lk.participant_id";
    pub const LK_TRACK_ID: &str = "lk.track_id";
    pub const LK_PII_ROOM_NAME: &str = "lk.pii.room_name";
    pub const LK_PII_PARTICIPANT_IDENTITY: &str = "lk.pii.participant_identity";
    pub const LK_PII_REMOTE_PARTICIPANT_IDENTITY: &str = "lk.pii.remote_participant_identity";
    pub const LK_SDK_NAME: &str = "lk.sdk.name";
    pub const LK_SDK_VERSION: &str = "lk.sdk.version";
    pub const LK_TRACK_SOURCE: &str = "lk.track_source";
    pub const LK_DISCONNECT_REASON: &str = "lk.disconnect_reason";
    pub const LK_AGENT_NAME: &str = "lk.agent_name";
}

/// OpenTelemetry semantic conventions v1.44.0 and the GenAI registry.
pub mod otel {
    pub const ERROR_TYPE: &str = "error.type";
    pub const EXCEPTION_TYPE: &str = "exception.type";
    pub const EXCEPTION_MESSAGE: &str = "exception.message";
    pub const EXCEPTION_STACKTRACE: &str = "exception.stacktrace";
    pub const RPC_METHOD: &str = "rpc.method";

    pub const GEN_AI_AGENT_NAME: &str = "gen_ai.agent.name";
    pub const GEN_AI_CONVERSATION_ID: &str = "gen_ai.conversation.id";
    pub const GEN_AI_EVALUATION_EXPLANATION: &str = "gen_ai.evaluation.explanation";
    pub const GEN_AI_EVALUATION_NAME: &str = "gen_ai.evaluation.name";
    pub const GEN_AI_EVALUATION_SCORE_LABEL: &str = "gen_ai.evaluation.score.label";
    pub const GEN_AI_INPUT_MESSAGES: &str = "gen_ai.input.messages";
    pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";
    pub const GEN_AI_OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
    pub const GEN_AI_OUTPUT_TYPE: &str = "gen_ai.output.type";
    pub const GEN_AI_PROVIDER_NAME: &str = "gen_ai.provider.name";
    pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const GEN_AI_REQUEST_STREAM: &str = "gen_ai.request.stream";
    pub const GEN_AI_RESPONSE_FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    pub const GEN_AI_RESPONSE_ID: &str = "gen_ai.response.id";
    pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";
    pub const GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK: &str = "gen_ai.response.time_to_first_chunk";
    pub const GEN_AI_SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
    pub const GEN_AI_TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
    pub const GEN_AI_TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
    pub const GEN_AI_TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
    pub const GEN_AI_TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
    pub const GEN_AI_TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
    pub const GEN_AI_TOOL_NAME: &str = "gen_ai.tool.name";
    pub const GEN_AI_TOOL_TYPE: &str = "gen_ai.tool.type";
    pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";
    pub const GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: &str = "gen_ai.usage.cache_write.input_tokens";
    pub const GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";
    pub const GEN_AI_USAGE_TEXT_INPUT_TOKENS: &str = "gen_ai.usage.text.input_tokens";
    pub const GEN_AI_USAGE_AUDIO_INPUT_TOKENS: &str = "gen_ai.usage.audio.input_tokens";
    pub const GEN_AI_USAGE_IMAGE_INPUT_TOKENS: &str = "gen_ai.usage.image.input_tokens";
    pub const GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: &str = "gen_ai.usage.text.output_tokens";
    pub const GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: &str = "gen_ai.usage.audio.output_tokens";
    pub const GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS: &str =
        "gen_ai.usage.text.cache_read.input_tokens";
    pub const GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS: &str =
        "gen_ai.usage.audio.cache_read.input_tokens";
    pub const GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS: &str =
        "gen_ai.usage.image.cache_read.input_tokens";
    pub const GEN_AI_WORKFLOW_NAME: &str = "gen_ai.workflow.name";

    /// Metrics imported from the GenAI registry.
    pub const GEN_AI_EXECUTE_TOOL_DURATION: &str = "gen_ai.execute_tool.duration";
    pub const GEN_AI_INVOKE_AGENT_DURATION: &str = "gen_ai.invoke_agent.duration";
}
