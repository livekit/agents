"""Every telemetry string the package emitted before the vocabulary moved to the protocol
registry, pinned against the constant that now carries it. A registry change that renames, drops
or retypes one of them fails here, before it changes a byte on the wire."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import observability, types
from livekit.agents.telemetry import _semconv, otel_metrics, trace_types

pytestmark = pytest.mark.unit

# (`module.NAME` as it was before the change, the literal the source held): the 196 trace_types
# constants, the 20 `lk.agents.*` instruments, the telemetry metadata keys and the report tags.
PINS: list[tuple[str, str]] = [
    ("trace_types.ATTR_SPEECH_ID", "lk.speech_id"),
    ("trace_types.ATTR_AGENT_LABEL", "lk.agent_label"),
    ("trace_types.ATTR_RETRY_COUNT", "lk.retry_count"),
    ("trace_types.ATTR_PROVIDER_REQUEST_IDS", "lk.provider_request_ids"),
    ("trace_types.ATTR_PARTICIPANT_ID", "lk.participant_id"),
    ("trace_types.ATTR_PARTICIPANT_IDENTITY", "lk.pii.participant_identity"),
    ("trace_types.ATTR_PARTICIPANT_KIND", "lk.participant_kind"),
    ("trace_types.ATTR_JOB_ID", "lk.job_id"),
    ("trace_types.ATTR_AGENT_NAME", "lk.agent_name"),
    ("trace_types.ATTR_CLOUD_AGENT_ID", "lk.cloud_agent_id"),
    ("trace_types.ATTR_DEPLOYMENT_ID", "lk.deployment_id"),
    ("trace_types.ATTR_ROOM_NAME", "lk.pii.room_name"),
    ("trace_types.ATTR_ROOM_SID", "lk.room_sid"),
    ("trace_types.ATTR_DISPATCH_ID", "lk.dispatch_id"),
    ("trace_types.ATTR_WORKER_ID", "lk.job.worker_id"),
    ("trace_types.ATTR_JOB_AGENT_ID", "lk.job.agent_id"),
    ("trace_types.ATTR_SIP_PREFIX", "lk.sip."),
    ("trace_types.ATTR_SIP_PHONE_NUMBER", "lk.pii.sip.phoneNumber"),
    ("trace_types.ATTR_JOB_ACCEPT_LATENCY", "lk.job.accept_latency"),
    ("trace_types.ATTR_JOB_ASSIGNMENT_LATENCY", "lk.job.assignment_latency"),
    ("trace_types.ATTR_JOB_LAUNCH_LATENCY", "lk.job.launch_latency"),
    ("trace_types.ATTR_JOB_ENTRYPOINT_LATENCY", "lk.job.entrypoint_latency"),
    ("trace_types.ATTR_JOB_DISPATCH_LATENCY", "lk.job.dispatch_latency"),
    ("trace_types.ATTR_KEYTERMS_COUNT", "lk.keyterms.count"),
    ("trace_types.ATTR_KEYTERMS_ADDED", "lk.keyterms.added"),
    ("trace_types.ATTR_KEYTERMS_REMOVED", "lk.keyterms.removed"),
    ("trace_types.ATTR_ROOM_AUTO_SUBSCRIBE", "lk.room.auto_subscribe"),
    ("trace_types.ATTR_ROOM_E2EE", "lk.room.e2ee"),
    ("trace_types.ATTR_ROOM_REMOTE_PARTICIPANT_COUNT", "lk.room.remote_participant_count"),
    ("trace_types.ATTR_ROOM_IO_PARTICIPANT_FILTER", "lk.room_io.participant_filter"),
    ("trace_types.ATTR_TRACK_SID", "lk.track_sid"),
    ("trace_types.ATTR_TRACK_SOURCE", "lk.track_source"),
    ("trace_types.ATTR_FIRST_FRAME_DELAY", "lk.first_frame_delay"),
    ("trace_types.ATTR_PRE_CONNECT_AUDIO_DURATION", "lk.pre_connect_audio.duration"),
    ("trace_types.ATTR_CONNECTION_STATE", "lk.connection_state"),
    ("trace_types.ATTR_DISCONNECT_REASON", "lk.disconnect_reason"),
    ("trace_types.ATTR_OLD_STATE", "lk.old_state"),
    ("trace_types.ATTR_NEW_STATE", "lk.new_state"),
    ("trace_types.ATTR_RPC_METHOD", "rpc.method"),
    ("trace_types.ATTR_RPC_REQUEST_ID", "lk.rpc.request_id"),
    ("trace_types.ATTR_RPC_CALLER_IDENTITY", "lk.rpc.caller_identity"),
    ("trace_types.ATTR_RPC_DESTINATION_IDENTITY", "lk.rpc.destination_identity"),
    ("trace_types.ATTR_RPC_PAYLOAD", "lk.pii.rpc.payload"),
    ("trace_types.ATTR_RPC_PAYLOAD_SIZE", "lk.rpc.payload_size"),
    ("trace_types.ATTR_RPC_RESPONSE", "lk.pii.rpc.response"),
    ("trace_types.ATTR_RPC_RESPONSE_SIZE", "lk.rpc.response_size"),
    ("trace_types.ATTR_RPC_RESPONSE_TIMEOUT", "lk.rpc.response_timeout"),
    ("trace_types.ATTR_RPC_ERROR_CODE", "lk.rpc.error_code"),
    ("trace_types.ATTR_RPC_HANDLER_REGISTERED", "lk.rpc.handler_registered"),
    ("trace_types.ATTR_CLOSE_REASON", "lk.close_reason"),
    ("trace_types.ATTR_CLOSE_DRAIN", "lk.close.drain"),
    ("trace_types.ATTR_SHUTDOWN_REASON", "lk.shutdown.reason"),
    ("trace_types.ATTR_SHUTDOWN_USER_INITIATED", "lk.shutdown.user_initiated"),
    ("trace_types.ATTR_CALLBACK_NAME", "lk.callback.name"),
    ("trace_types.ATTR_AGENT_TURN_ID", "lk.generation_id"),
    ("trace_types.ATTR_AGENT_PARENT_TURN_ID", "lk.parent_generation_id"),
    ("trace_types.ATTR_GENERATION_COUNT", "lk.generation_count"),
    ("trace_types.ATTR_USER_INPUT", "lk.pii.user_input"),
    ("trace_types.ATTR_INSTRUCTIONS", "lk.pii.instructions"),
    ("trace_types.ATTR_SPEECH_INTERRUPTED", "lk.interrupted"),
    ("trace_types.ATTR_CHAT_CTX", "lk.pii.chat_ctx"),
    ("trace_types.ATTR_FUNCTION_TOOLS", "lk.function_tools"),
    ("trace_types.ATTR_PROVIDER_TOOLS", "lk.provider_tools"),
    ("trace_types.ATTR_TOOL_SETS", "lk.tool_sets"),
    ("trace_types.ATTR_RESPONSE_TEXT", "lk.pii.response.text"),
    ("trace_types.ATTR_RESPONSE_FUNCTION_CALLS", "lk.pii.response.function_calls"),
    ("trace_types.ATTR_RESPONSE_TTFT", "lk.response.ttft"),
    ("trace_types.ATTR_INPUT_DELTA", "lk.input.delta"),
    ("trace_types.ATTR_INPUT_BASE_SPAN_ID", "lk.input.base_span_id"),
    ("trace_types.ATTR_INPUT_DROPPED_FROM_BASE", "lk.input.dropped_from_base"),
    ("trace_types.ATTR_FUNCTION_TOOL_ID", "lk.function_tool.id"),
    ("trace_types.ATTR_FUNCTION_TOOL_NAME", "lk.function_tool.name"),
    ("trace_types.ATTR_FUNCTION_TOOL_ARGS", "lk.pii.function_tool.arguments"),
    ("trace_types.ATTR_FUNCTION_TOOL_IS_ERROR", "lk.function_tool.is_error"),
    ("trace_types.ATTR_FUNCTION_TOOL_OUTPUT", "lk.pii.function_tool.output"),
    ("trace_types.ATTR_TTS_INPUT_TEXT", "lk.pii.input_text"),
    ("trace_types.ATTR_TTS_STREAMING", "lk.tts.streaming"),
    ("trace_types.ATTR_TTS_LABEL", "lk.tts.label"),
    ("trace_types.ATTR_RESPONSE_TTFB", "lk.response.ttfb"),
    ("trace_types.ATTR_EOU_PROBABILITY", "lk.eou.probability"),
    ("trace_types.ATTR_EOU_UNLIKELY_THRESHOLD", "lk.eou.unlikely_threshold"),
    ("trace_types.ATTR_EOU_DELAY", "lk.eou.endpointing_delay"),
    ("trace_types.ATTR_EOU_LANGUAGE", "lk.eou.language"),
    ("trace_types.ATTR_USER_TRANSCRIPT", "lk.pii.user_transcript"),
    ("trace_types.ATTR_STT_EVENTS", "lk.stt.events"),
    ("trace_types.ATTR_TRANSCRIPT_CONFIDENCE", "lk.transcript_confidence"),
    ("trace_types.ATTR_TRANSCRIPTION_DELAY", "lk.transcription_delay"),
    ("trace_types.ATTR_END_OF_TURN_DELAY", "lk.end_of_turn_delay"),
    ("trace_types.ATTR_EOU_SOURCE", "lk.eou.source"),
    ("trace_types.ATTR_EOU_DETECTION_DELAY", "lk.eou.detection_delay"),
    ("trace_types.ATTR_EOU_FROM_CACHE", "lk.eou.from_cache"),
    ("trace_types.ATTR_EOU_OUTCOME", "lk.eou.outcome"),
    ("trace_types.ATTR_EOU_WAIT_DURATION", "lk.eou.wait_duration"),
    ("trace_types.ATTR_EOU_REARM_COUNT", "lk.eou.rearm_count"),
    ("trace_types.ATTR_EOU_NOT_COMMITTED_COUNT", "lk.eou.not_committed_count"),
    ("trace_types.ATTR_EOU_RESUME_COUNT", "lk.eou.resume_count"),
    ("trace_types.ATTR_ON_USER_TURN_COMPLETED_DELAY", "lk.on_user_turn_completed_delay"),
    ("trace_types.ATTR_SPEECH_QUEUE_WAIT", "lk.speech.queue_wait"),
    ("trace_types.ATTR_LLM_METRICS", "lk.llm_metrics"),
    ("trace_types.ATTR_TTS_METRICS", "lk.tts_metrics"),
    ("trace_types.ATTR_REALTIME_MODEL_METRICS", "lk.realtime_model_metrics"),
    ("trace_types.ATTR_E2E_LATENCY", "lk.e2e_latency"),
    ("trace_types.ATTR_GEN_AI_OPERATION_NAME", "gen_ai.operation.name"),
    ("trace_types.ATTR_GEN_AI_PROVIDER_NAME", "gen_ai.provider.name"),
    ("trace_types.ATTR_GEN_AI_REQUEST_MODEL", "gen_ai.request.model"),
    ("trace_types.ATTR_GEN_AI_REQUEST_STREAM", "gen_ai.request.stream"),
    ("trace_types.ATTR_GEN_AI_RESPONSE_ID", "gen_ai.response.id"),
    ("trace_types.ATTR_GEN_AI_RESPONSE_MODEL", "gen_ai.response.model"),
    ("trace_types.ATTR_GEN_AI_RESPONSE_FINISH_REASONS", "gen_ai.response.finish_reasons"),
    ("trace_types.ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK", "gen_ai.response.time_to_first_chunk"),
    ("trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS", "gen_ai.usage.input_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS", "gen_ai.usage.output_tokens"),
    (
        "trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS",
        "gen_ai.usage.cache_read.input_tokens",
    ),
    (
        "trace_types.ATTR_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS",
        "gen_ai.usage.cache_write.input_tokens",
    ),
    (
        "trace_types.ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS",
        "gen_ai.usage.reasoning.output_tokens",
    ),
    ("trace_types.ATTR_GEN_AI_USAGE_TEXT_INPUT_TOKENS", "gen_ai.usage.text.input_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_TEXT_OUTPUT_TOKENS", "gen_ai.usage.text.output_tokens"),
    (
        "trace_types.ATTR_GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS",
        "gen_ai.usage.text.cache_read.input_tokens",
    ),
    ("trace_types.ATTR_GEN_AI_USAGE_AUDIO_INPUT_TOKENS", "gen_ai.usage.audio.input_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS", "gen_ai.usage.audio.output_tokens"),
    (
        "trace_types.ATTR_GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS",
        "gen_ai.usage.audio.cache_read.input_tokens",
    ),
    ("trace_types.ATTR_GEN_AI_USAGE_IMAGE_INPUT_TOKENS", "gen_ai.usage.image.input_tokens"),
    (
        "trace_types.ATTR_GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS",
        "gen_ai.usage.image.cache_read.input_tokens",
    ),
    ("trace_types.ATTR_GEN_AI_TOKEN_TYPE", "gen_ai.token.type"),
    ("trace_types.ATTR_GEN_AI_CONVERSATION_ID", "gen_ai.conversation.id"),
    ("trace_types.ATTR_GEN_AI_AGENT_NAME", "gen_ai.agent.name"),
    ("trace_types.ATTR_GEN_AI_TOOL_NAME", "gen_ai.tool.name"),
    ("trace_types.ATTR_GEN_AI_TOOL_CALL_ID", "gen_ai.tool.call.id"),
    ("trace_types.ATTR_GEN_AI_TOOL_DESCRIPTION", "gen_ai.tool.description"),
    ("trace_types.ATTR_GEN_AI_TOOL_TYPE", "gen_ai.tool.type"),
    ("trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS", "gen_ai.tool.call.arguments"),
    ("trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT", "gen_ai.tool.call.result"),
    ("trace_types.ATTR_GEN_AI_TOOL_DEFINITIONS", "gen_ai.tool.definitions"),
    ("trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS", "gen_ai.system_instructions"),
    ("trace_types.ATTR_GEN_AI_INPUT_MESSAGES", "gen_ai.input.messages"),
    ("trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES", "gen_ai.output.messages"),
    ("trace_types.ATTR_GEN_AI_OUTPUT_TYPE", "gen_ai.output.type"),
    ("trace_types.ATTR_GEN_AI_RETRIEVAL_DOCUMENTS", "gen_ai.retrieval.documents"),
    ("trace_types.ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT", "gen_ai.retrieval.query.text"),
    ("trace_types.ATTR_GEN_AI_MEMORY_QUERY_TEXT", "gen_ai.memory.query.text"),
    ("trace_types.ATTR_GEN_AI_MEMORY_RECORDS", "gen_ai.memory.records"),
    ("trace_types.ATTR_GEN_AI_EVALUATION_NAME", "gen_ai.evaluation.name"),
    ("trace_types.ATTR_GEN_AI_EVALUATION_SCORE_LABEL", "gen_ai.evaluation.score.label"),
    ("trace_types.ATTR_GEN_AI_EVALUATION_EXPLANATION", "gen_ai.evaluation.explanation"),
    ("trace_types.ATTR_GEN_AI_PROMPT_VARIABLE", "gen_ai.prompt.variable"),
    ("trace_types.ATTR_GEN_AI_WORKFLOW_NAME", "gen_ai.workflow.name"),
    ("trace_types.ATTR_ERROR_TYPE", "error.type"),
    ("trace_types.ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS", "gen_ai.usage.input_text_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS", "gen_ai.usage.input_audio_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS", "gen_ai.usage.input_cached_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS", "gen_ai.usage.output_text_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS", "gen_ai.usage.output_audio_tokens"),
    ("trace_types.ATTR_GEN_AI_USAGE_REASONING_TOKENS", "gen_ai.usage.reasoning_tokens"),
    ("trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE", "gen_ai.system.message"),
    ("trace_types.EVENT_GEN_AI_USER_MESSAGE", "gen_ai.user.message"),
    ("trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE", "gen_ai.assistant.message"),
    ("trace_types.EVENT_GEN_AI_TOOL_MESSAGE", "gen_ai.tool.message"),
    ("trace_types.EVENT_GEN_AI_CHOICE", "gen_ai.choice"),
    (
        "trace_types.EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS",
        "gen_ai.client.inference.operation.details",
    ),
    ("trace_types.METRIC_GEN_AI_CLIENT_TOKEN_USAGE", "gen_ai.client.token.usage"),
    ("trace_types.METRIC_GEN_AI_CLIENT_OPERATION_DURATION", "gen_ai.client.operation.duration"),
    (
        "trace_types.METRIC_GEN_AI_CLIENT_TIME_TO_FIRST_CHUNK",
        "gen_ai.client.operation.time_to_first_chunk",
    ),
    ("trace_types.METRIC_GEN_AI_INVOKE_AGENT_DURATION", "gen_ai.invoke_agent.duration"),
    ("trace_types.METRIC_GEN_AI_EXECUTE_TOOL_DURATION", "gen_ai.execute_tool.duration"),
    ("trace_types.ATTR_EXCEPTION_TRACE", "exception.stacktrace"),
    ("trace_types.ATTR_EXCEPTION_TYPE", "exception.type"),
    ("trace_types.ATTR_EXCEPTION_MESSAGE", "exception.message"),
    (
        "trace_types.ATTR_LANGFUSE_COMPLETION_START_TIME",
        "langfuse.observation.completion_start_time",
    ),
    ("trace_types.ATTR_AMD_CATEGORY", "lk.amd.category"),
    ("trace_types.ATTR_AMD_REASON", "lk.amd.reason"),
    ("trace_types.ATTR_AMD_SPEECH_DURATION", "lk.amd.speech_duration"),
    ("trace_types.ATTR_AMD_DELAY", "lk.amd.delay"),
    ("trace_types.ATTR_AMD_TRANSCRIPT", "lk.pii.amd.transcript"),
    ("trace_types.ATTR_INTERRUPTION_SOURCE", "lk.interruption.source"),
    ("trace_types.ATTR_PLAYOUT_POSITION", "lk.playout.position"),
    ("trace_types.ATTR_PREVIOUS_AGENT_LABEL", "lk.previous_agent_label"),
    ("trace_types.ATTR_FALLBACK_LABEL", "lk.fallback.label"),
    ("trace_types.ATTR_FALLBACK_INDEX", "lk.fallback.index"),
    ("trace_types.ATTR_FALLBACK_RECOVERY", "lk.fallback.recovery"),
    ("trace_types.ATTR_IS_INTERRUPTION", "lk.is_interruption"),
    ("trace_types.ATTR_INTERRUPTION_PROBABILITY", "lk.interruption.probability"),
    ("trace_types.ATTR_INTERRUPTION_TOTAL_DURATION", "lk.interruption.total_duration"),
    ("trace_types.ATTR_INTERRUPTION_PREDICTION_DURATION", "lk.interruption.prediction_duration"),
    ("trace_types.ATTR_INTERRUPTION_DETECTION_DELAY", "lk.interruption.detection_delay"),
    ("trace_types.ATTR_BLOCKING_DURATION", "lk.blocking.duration"),
    ("trace_types.ATTR_BLOCKING_THRESHOLD", "lk.blocking.threshold"),
    ("trace_types.ATTR_BLOCKING_SEVERITY", "lk.blocking.severity"),
    ("trace_types.ATTR_BLOCKING_TASK", "lk.blocking.task"),
    ("trace_types.ATTR_BLOCKING_STACK", "lk.blocking.stack"),
    ("trace_types.ATTR_BLOCKING_GC_TIME", "lk.blocking.gc_time"),
    ("trace_types.ATTR_BLOCKING_CPU_TIME", "lk.blocking.cpu_time"),
    ("trace_types.ATTR_BLOCKING_IMPORT", "lk.blocking.import"),
    ("trace_types.ATTR_BLOCKING_SUPPRESSED", "lk.blocking.suppressed"),
    ("trace_types.ATTR_BLOCKING_COUNT", "lk.blocking.count"),
    ("trace_types.ATTR_BLOCKING_TOTAL_DURATION", "lk.blocking.total_duration"),
    ("trace_types.ATTR_BLOCKING_MAX_DURATION", "lk.blocking.max_duration"),
    ("otel_metrics._turn_e2e_latency", "lk.agents.turn.e2e_latency"),
    ("otel_metrics._turn_llm_ttft", "lk.agents.turn.llm_ttft"),
    ("otel_metrics._turn_tts_ttfb", "lk.agents.turn.tts_ttfb"),
    ("otel_metrics._turn_transcription_delay", "lk.agents.turn.transcription_delay"),
    ("otel_metrics._turn_end_of_turn_delay", "lk.agents.turn.end_of_turn_delay"),
    (
        "otel_metrics._turn_on_user_turn_completed_delay",
        "lk.agents.turn.on_user_turn_completed_delay",
    ),
    ("otel_metrics._llm_input_tokens", "lk.agents.usage.llm_input_tokens"),
    ("otel_metrics._llm_input_cached_tokens", "lk.agents.usage.llm_input_cached_tokens"),
    ("otel_metrics._llm_output_tokens", "lk.agents.usage.llm_output_tokens"),
    ("otel_metrics._llm_input_audio_tokens", "lk.agents.usage.llm_input_audio_tokens"),
    ("otel_metrics._llm_input_text_tokens", "lk.agents.usage.llm_input_text_tokens"),
    ("otel_metrics._llm_output_audio_tokens", "lk.agents.usage.llm_output_audio_tokens"),
    ("otel_metrics._llm_output_text_tokens", "lk.agents.usage.llm_output_text_tokens"),
    ("otel_metrics._llm_session_duration", "lk.agents.usage.llm_session_duration"),
    ("otel_metrics._tts_characters", "lk.agents.usage.tts_characters"),
    ("otel_metrics._tts_audio_duration", "lk.agents.usage.tts_audio_duration"),
    ("otel_metrics._stt_audio_duration", "lk.agents.usage.stt_audio_duration"),
    ("otel_metrics._interruption_num_requests", "lk.agents.usage.interruption_num_requests"),
    ("otel_metrics._connection_acquire_time", "lk.agents.connection.acquire_time"),
    ("otel_metrics._event_loop_blocked", "lk.agents.event_loop.blocked_duration"),
    ("types.ATTRIBUTE_SIMULATION_ENABLED", "lk.simulation.enabled"),
    ("types.ATTRIBUTE_SIMULATION_RUN_ID", "lk.simulation.run_id"),
    ("types.ATTRIBUTE_SIMULATION_JOB_ID", "lk.simulation.job_id"),
    ("types.ATTRIBUTE_REDACTION_ENABLED", "lk.redaction.enabled"),
    ("observability.TAG_SUCCESS", "lk.success"),
    ("observability.TAG_FAIL", "lk.fail"),
    # enum values the emitters pass through (`gen_ai.operation.name`, `gen_ai.output.type`,
    # `gen_ai.response.finish_reasons`, `lk.eou.outcome`), as they were before the change
    ("trace_types.GenAIOperationName.CHAT", "chat"),
    ("trace_types.GenAIOperationName.GENERATE_CONTENT", "generate_content"),
    ("trace_types.GenAIOperationName.TEXT_COMPLETION", "text_completion"),
    ("trace_types.GenAIOperationName.EMBEDDINGS", "embeddings"),
    ("trace_types.GenAIOperationName.RETRIEVAL", "retrieval"),
    ("trace_types.GenAIOperationName.FETCH_RESPONSE", "fetch_response"),
    ("trace_types.GenAIOperationName.CREATE_AGENT", "create_agent"),
    ("trace_types.GenAIOperationName.INVOKE_AGENT", "invoke_agent"),
    ("trace_types.GenAIOperationName.EXECUTE_TOOL", "execute_tool"),
    ("trace_types.GenAIOperationName.INVOKE_WORKFLOW", "invoke_workflow"),
    ("trace_types.GenAIOperationName.PLAN", "plan"),
    ("trace_types.GenAIOperationName.SEARCH_MEMORY", "search_memory"),
    ("trace_types.GenAIOperationName.CREATE_MEMORY", "create_memory"),
    ("trace_types.GenAIOperationName.UPDATE_MEMORY", "update_memory"),
    ("trace_types.GenAIOperationName.UPSERT_MEMORY", "upsert_memory"),
    ("trace_types.GenAIOperationName.DELETE_MEMORY", "delete_memory"),
    ("trace_types.GenAIOperationName.CREATE_MEMORY_STORE", "create_memory_store"),
    ("trace_types.GenAIOperationName.DELETE_MEMORY_STORE", "delete_memory_store"),
    ("trace_types.GenAIOutputType.TEXT", "text"),
    ("trace_types.GenAIOutputType.JSON", "json"),
    ("trace_types.GenAIOutputType.IMAGE", "image"),
    ("trace_types.GenAIOutputType.SPEECH", "speech"),
    ("trace_types.GenAIFinishReason.STOP", "stop"),
    ("trace_types.GenAIFinishReason.LENGTH", "length"),
    ("trace_types.GenAIFinishReason.CONTENT_FILTER", "content_filter"),
    ("trace_types.GenAIFinishReason.TOOL_CALL", "tool_call"),
    ("trace_types.GenAIFinishReason.COMPACTION", "compaction"),
    ("trace_types.GenAIFinishReason.ERROR", "error"),
    ("trace_types.EOU_OUTCOME_COMMITTED", "committed"),
    ("trace_types.EOU_OUTCOME_USER_RESUMED", "user_resumed"),
    ("trace_types.EOU_OUTCOME_DROPPED", "dropped"),
]

# (`otel_metrics` instrument, unit it was created with before the change; "" = none).
UNITS: list[tuple[str, str]] = [
    ("_turn_e2e_latency", "s"),
    ("_turn_llm_ttft", "s"),
    ("_turn_tts_ttfb", "s"),
    ("_turn_transcription_delay", "s"),
    ("_turn_end_of_turn_delay", "s"),
    ("_turn_on_user_turn_completed_delay", "s"),
    ("_llm_input_tokens", ""),
    ("_llm_input_cached_tokens", ""),
    ("_llm_output_tokens", ""),
    ("_llm_input_audio_tokens", ""),
    ("_llm_input_text_tokens", ""),
    ("_llm_output_audio_tokens", ""),
    ("_llm_output_text_tokens", ""),
    ("_llm_session_duration", "s"),
    ("_tts_characters", ""),
    ("_tts_audio_duration", "s"),
    ("_stt_audio_duration", "s"),
    ("_interruption_num_requests", ""),
    ("_connection_acquire_time", "s"),
    ("_event_loop_blocked", "s"),
    ("_genai_token_usage", "{token}"),
    ("_genai_operation_duration", "s"),
    ("_genai_time_to_first_chunk", "s"),
    ("_genai_execute_tool_duration", "s"),
    ("_genai_invoke_agent_duration", "s"),
]

# Metrics the registry cannot declare (weaver cannot `ref` an upstream metric): their unit is
# pinned above but has no `METRIC_*_UNIT` constant to agree with.
UNREGISTERED_METRICS = {
    "gen_ai.client.operation.duration",
    "gen_ai.execute_tool.duration",
    "gen_ai.invoke_agent.duration",
}

# Every span-event name the package passes to `add_event`, as it was before the change.
EVENTS = [
    "agent_state_changed",
    "answer_sent",
    "early_pc_created",
    "engine",
    "entrypoint_cancelled",
    "entrypoint_returned",
    "entrypoint_started",
    "event_loop_blocked",
    "exception",
    "first_frame",
    "first_media",
    "gen_ai.client.inference.operation.details",
    "generation",
    "job_accepted",
    "job_assigned",
    "job_received",
    "job_started_on_server",
    "join_recv",
    "offer_sent",
    "participant_linked",
    "pc_connected",
    "pc_created",
    "pre_connect_audio",
    "preemptive_generation_discarded",
    "process_assigned",
    "rearmed",
    "room_connected",
    "signal",
    "stop_response",
    "subscribed",
    "superseded",
    "track_subscribed",
    "user_resumed",
    "user_state_changed",
    "ws_open",
]

# tests/trace_schema.py's SPAN_PARENTS keys: every span name the package starts.
SPANS = [
    "agent_session",
    "agent_speaking",
    "agent_turn",
    "amd",
    "drain_agent_activity",
    "eou_detection",
    "eou_wait",
    "event_loop_blocked",
    "function_tool",
    "job_entrypoint",
    "job_shutdown",
    "judge_evaluation",
    "keyterm_detection",
    "llm_fallback_adapter",
    "llm_node",
    "llm_request",
    "llm_request_run",
    "on_enter",
    "on_exit",
    "on_session_end",
    "on_user_turn_completed",
    "pause_agent_activity",
    "publish_audio_output",
    "realtime_inference",
    "realtime_metrics",
    "resume_agent_activity",
    "room_connect",
    "room_disconnect",
    "rpc_call",
    "rpc_handler",
    "session_close",
    "session_end_upload",
    "session_start",
    "setup_toolsets",
    "shutdown_callback",
    "start_agent_activity",
    "tts_fallback_adapter",
    "tts_node",
    "tts_request",
    "tts_request_run",
    "tts_stream_adapter",
    "update_agent",
    "user_speaking",
    "user_turn",
    "wait_for_audio_track",
    "wait_for_participant",
    "wait_for_video_track",
]


def _current(path: str) -> str:
    module, name = path.split(".", 1)
    if module == "otel_metrics":
        # the instrument (or its proxy, before a meter provider is set) carries its registered name
        inst = getattr(otel_metrics, name)
        return getattr(inst, "name", None) or inst._name
    if module == "observability":
        return {"TAG_SUCCESS": observability.LK_SUCCESS, "TAG_FAIL": observability.LK_FAIL}[name]
    value: object = {"trace_types": trace_types, "types": types}[module]
    for attr in name.split("."):
        value = getattr(value, attr)
    return value  # type: ignore[return-value]


@pytest.mark.parametrize(("path", "old"), PINS, ids=[p for p, _ in PINS])
def test_every_pre_registry_literal_is_what_its_constant_says(path: str, old: str) -> None:
    assert _current(path) == old, f"the wire string changed: {path} was {old!r}"


@pytest.mark.parametrize("span", SPANS)
def test_every_span_name_is_registered(span: str) -> None:
    assert getattr(_semconv, "SPAN_" + span.upper()) == span


@pytest.mark.parametrize(("name", "old"), UNITS, ids=[n for n, _ in UNITS])
def test_every_instrument_keeps_the_unit_it_shipped_with(name: str, old: str) -> None:
    inst = getattr(otel_metrics, name)
    unit = getattr(inst, "unit", None)
    assert (unit if unit is not None else inst._unit) == old, f"the unit of {name} changed"
    metric = getattr(inst, "name", None) or inst._name
    if metric in UNREGISTERED_METRICS:
        return
    key = "METRIC_" + metric.upper().replace(".", "_")
    assert getattr(_semconv, key + "_UNIT") == old, "the registry declares a different unit"


@pytest.mark.parametrize("event", EVENTS)
def test_every_span_event_name_is_registered(event: str) -> None:
    assert getattr(trace_types, "CHECKPOINT_" + event.upper().replace(".", "_")) == event


# The values are the protobuf enum names `room_io` emits at runtime (`rtc.<Enum>.Name(..)`).
@pytest.mark.parametrize(
    ("names", "prefix"),
    [
        (rtc.TrackSource.keys(), "LK_TRACK_SOURCE_"),
        (rtc.DisconnectReason.keys(), "LK_DISCONNECT_REASON_"),
    ],
    ids=["lk.track_source", "lk.disconnect_reason"],
)
def test_every_runtime_enum_name_is_a_registered_value(names: list[str], prefix: str) -> None:
    unregistered = [n for n in names if getattr(_semconv, prefix + n, None) != n]
    assert not unregistered, f"emitted but not registered: {unregistered}"


PACKAGE = Path(trace_types.__file__).resolve().parents[1]

# `lk.*` string constants that are not telemetry (participant attributes, data-channel topics,
# chat message ids, tool-call extras) or are deliberately literal (the documented conflict list,
# a dynamic tag). Everything else must come from `trace_types` / `_semconv`. An f-string is listed
# with its placeholders as `{expr}`.
LITERAL_ALLOWLIST: dict[str, set[str]] = {
    "types.py": {"*"},  # participant attributes, topics, userdata keys, telemetry metadata
    "telemetry/trace_types.py": {"lk.sip."},  # the conflict list
    "observability.py": {"lk.judge.{name}:{judgment.verdict}"},  # a dynamic tag
    "voice/amd/detector.py": {"lk.amd:{result.category.value}"},  # a dynamic tag
    "delegation/tool.py": {
        "lk.delegation.task_id",
        "lk.delegation.endpoint",
        "lk.delegation.transport",
        "lk.delegation.profile",
    },
    "llm/tool_context.py": {"lk.approval"},
    "voice/avatar/_datastream_io.py": {
        "lk.clear_buffer",
        "lk.playback_finished",
        "lk.playback_started",
        "lk.audio_stream",
    },
    "voice/generation.py": {"lk.agent_task.instructions", "lk.expressive.instructions"},
    "voice/remote_session.py": {"lk.agent.session"},
    "voice/room_io/_pre_connect_audio.py": {"lk.agent.pre-connect-audio-buffer"},
}

# Calls whose first argument (or `name=`) is a span or span-event name.
_SPAN_STARTERS = {
    "start_span",
    "start_as_current_span",
    "detached_span",
    "session_span",
    "use_span",
    "add_event",
    "_add_session_event",
}

# A folded value: a set of possible strings, or a tuple / list of folded elements, or a dict of
# folded entries keyed by their literal string key. `None` means "not decidable from this file".
Folded = set[str] | tuple["Folded", ...] | dict[str, "Folded"]
Env = dict[str, Folded]


def _bind_pattern(
    target: ast.expr, value: Folded | None, prefix: str, out: dict[str, Folded]
) -> None:
    """Bind one assignment pattern to one folded value, recursing through tuple and list
    destructuring (`(A, (B, C)) = ("x", ("y", "z"))`). A pattern that does not line up with the
    value, or a value that did not fold, binds nothing."""
    if value is None:
        return
    if isinstance(target, ast.Name):
        out[prefix + target.id] = value
    elif isinstance(target, ast.Tuple | ast.List) and isinstance(value, tuple):
        if len(target.elts) == len(value):
            for t, v in zip(target.elts, value, strict=True):
                _bind_pattern(t, v, prefix, out)


def _env(tree: ast.Module) -> Env:
    """Same-file constants, resolved transitively to a fixed point with the one `_fold` the call
    sites use: module-level names and class attributes (as `Cls.ATTR`; a class-local reference
    `B = A` inside the class resolves to `Cls.A`), from plain, annotated, chained and (nested)
    tuple- or list-destructured assignments. A name bound more than once: within a pass the last
    binding in source order that folds wins, and a binding that does not fold is skipped (the
    `found` dict is overwritten in statement order; the union below only merges a pass with the
    previous pass). Function-local names, parameters and imports are not constants."""
    # (prefix the targets live under, prefix the right-hand side resolves names in, statement)
    stmts: list[tuple[str, str, ast.stmt]] = [("", "", n) for n in tree.body]
    stmts += [
        (f"{n.name}.", f"{n.name}.", inner)
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        for inner in n.body
    ]
    env: Env = {}
    while True:
        found: dict[str, Folded] = {}
        for prefix, scope, stmt in stmts:
            targets: list[ast.expr]
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                targets = [stmt.target]
            else:
                continue
            value = _fold(stmt.value, env, scope)
            for t in targets:
                _bind_pattern(t, value, prefix, found)
        merged: Env = dict(env)
        for name, value in found.items():
            old = env.get(name)
            if old is None:
                merged[name] = value
            elif isinstance(old, set) and isinstance(value, set):
                merged[name] = old | value
            else:
                merged[name] = value
        if merged == env:
            return env
        env = merged


def _lookup(name: str, env: Env, scope: str) -> Folded | None:
    """A name as seen from `scope` (`""` at module level, `"Cls."` inside a class body): the
    class-local binding first, then the module-level one."""
    if scope and (v := env.get(scope + name)) is not None:
        return v
    return env.get(name)


def _strings(v: Folded | None) -> set[str] | None:
    return v if isinstance(v, set) else None


def _fold(e: ast.AST | None, env: Env, scope: str = "") -> Folded | None:
    """THE evaluator, used for every constant binding and every call-site argument alike: the
    value `e` has when it is decidable from literals and same-file constants.

    Folds: str / bytes / int / float / bool / None constants (scalars become their `str()`, as an
    f-string would render them); literal tuples and lists (element-wise) and dicts with literal
    string keys; implicit and `+` concatenation of strings; `sep.join(seq)` where `sep` and every
    element of `seq` fold and `seq` is a literal or a folded sequence; f-strings whose parts all
    fold and carry no format spec or conversion; a `Name` or `Cls.ATTR` bound in `env`.
    Does not fold (returns None): conditionals, format specs, calls other than `join`, anything
    else."""
    if isinstance(e, ast.Constant):
        if isinstance(e.value, str):
            return {e.value}
        if isinstance(e.value, bytes):
            return {e.value.decode("latin-1")}
        if isinstance(e.value, bool | int | float) or e.value is None:
            return {str(e.value)}
        return None
    if (
        isinstance(e, ast.UnaryOp)
        and isinstance(e.op, ast.UAdd | ast.USub)
        and isinstance(e.operand, ast.Constant)
        and isinstance(e.operand.value, int | float)
        and not isinstance(e.operand.value, bool)
    ):
        n = e.operand.value if isinstance(e.op, ast.UAdd) else -e.operand.value
        return {str(n)}
    if isinstance(e, ast.Tuple | ast.List):
        elts = [_fold(x, env, scope) for x in e.elts]
        return tuple(elts) if all(x is not None for x in elts) else None  # type: ignore[arg-type]
    if isinstance(e, ast.Dict):
        out: dict[str, Folded] = {}
        for k, v in zip(e.keys, e.values, strict=True):
            fv = _fold(v, env, scope)
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)) or fv is None:
                return None
            out[k.value] = fv
        return out
    if isinstance(e, ast.JoinedStr):
        acc = {""}
        for v in e.values:
            if isinstance(v, ast.FormattedValue) and (v.format_spec or v.conversion != -1):
                return None  # ponytail: format specs are obfuscation, out of scope
            part = _strings(_fold(v.value if isinstance(v, ast.FormattedValue) else v, env, scope))
            if part is None:
                return None
            acc = {a + b for a in acc for b in part}
        return acc
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        left = _strings(_fold(e.left, env, scope))
        right = _strings(_fold(e.right, env, scope))
        return (
            {a + b for a in left for b in right} if left is not None and right is not None else None
        )
    if (
        isinstance(e, ast.Call)
        and isinstance(e.func, ast.Attribute)
        and e.func.attr == "join"
        and len(e.args) == 1
        and not e.keywords
    ):
        sep = _strings(_fold(e.func.value, env, scope))
        seq = _fold(e.args[0], env, scope)
        if sep is None or not isinstance(seq, tuple):
            return None
        parts = [_strings(x) for x in seq]
        if any(p is None for p in parts):
            return None
        acc = parts[0] if parts else {""}
        for p in parts[1:]:
            acc = {a + j + b for a in acc for j in sep for b in p}  # type: ignore[union-attr]
        return acc
    if isinstance(e, ast.Name):
        return _lookup(e.id, env, scope)
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
        return env.get(f"{e.value.id}.{e.attr}")
    return None


def _texts(e: ast.AST | None, env: Env, scope: str) -> set[str]:
    """What `_fold` decides as strings, or for an f-string with runtime parts its template with
    `{expr}` placeholders (so a dynamic tag can be allowlisted by shape, and so
    `start_span(f"x_{n}")` is reported as `x_{n}`)."""
    folded = _strings(_fold(e, env, scope))
    if folded is not None:
        return folded
    if isinstance(e, ast.JoinedStr):
        parts = []
        for v in e.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                parts.append("{" + ast.unparse(v.value) + "}")
        return {"".join(parts)}
    return set()


def _called_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def _span_name_texts(call: ast.Call, env: Env, scope: str) -> set[str]:
    """The span name a call passes: first positional, `name=`, or the `"name"` entry of a `**`
    argument that folds to a dict (a literal or a same-file constant dict)."""
    arg: ast.expr | None = call.args[0] if call.args else None
    names: set[str] = set()
    for kw in call.keywords:
        if kw.arg == "name":
            arg = kw.value
        elif kw.arg is None:
            d = _fold(kw.value, env, scope)
            if isinstance(d, dict) and (v := _strings(d.get("name"))) is not None:
                names |= v
    return names | (_texts(arg, env, scope) if arg is not None else set())


def _scopes(tree: ast.Module) -> dict[int, str]:
    """For every node, the class-body scope prefix it is evaluated in (`""` at module level and
    inside functions, `"Cls."` directly inside `class Cls`)."""
    out: dict[int, str] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                visit(child, "")
            else:
                out[id(child)] = scope
                visit(child, scope)

    visit(tree, "")
    return out


def _offenders(rel: str, tree: ast.Module, allowed: set[str]) -> list[str]:
    env = _env(tree)
    scopes = _scopes(tree)
    out: dict[str, None] = {}  # ordered set
    # the pieces of an f-string are judged as part of the whole, not on their own
    pieces = {
        id(d) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) for d in ast.walk(n)
    } - {id(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)}
    for node in ast.walk(tree):
        if id(node) in pieces or isinstance(node, ast.Name | ast.Attribute):
            continue
        scope = scopes.get(id(node), "")
        if isinstance(node, ast.expr):
            for text in sorted(_texts(node, env, scope)):
                if text.startswith("lk.") and "*" not in allowed and text not in allowed:
                    out[f"{rel}:{node.lineno}: {text!r}"] = None
        if isinstance(node, ast.Call) and _called_name(node) in _SPAN_STARTERS:
            for text in sorted(_span_name_texts(node, env, scope)):
                if text:  # an empty name is not a span anyone meant to start
                    out[f"{rel}:{node.lineno}: {_called_name(node)}({text!r})"] = None
        if isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in targets if isinstance(t, ast.Name)}
            if "_span_name" in names and node.value is not None:
                for text in sorted(_texts(node.value, env, scope)):
                    out[f"{rel}:{node.lineno}: _span_name = {text!r}"] = None
    return list(out)


def test_no_telemetry_name_is_hand_written_at_an_emitter_site() -> None:
    """Binds the emitter sites to the registry, on the AST rather than the source text. The scan
    exists to catch an accidental stray telemetry name, not deliberate obfuscation. Its contract
    is this exact list; no sentence here claims more than the list.

    FLAGGED: a string that starts with `lk.`; a span / span-event started by a string (first
    positional, `name=`, or `**` of an expression that folds to a dict with a `"name"` key); a
    `_span_name` assigned a string directly (`_span_name = …` / `_span_name: T = …`; a
    destructured target such as `(_span_name,) = (…)` is not checked). "A string" means an
    expression the evaluator folds.

    BINDING SITES collected (and nowhere else): assignments and annotated assignments at module
    level and directly inside a top-level `class` body, including chained targets and nested
    tuple / list destructuring. Class attributes are keyed `Cls.ATTR`. A name bound more than
    once: within each pass the LAST binding in source order that folds wins, and a binding that
    does not fold is skipped (it neither poisons nor resets the name), so `A = "lk"; x = A + …;
    A = "other"` reads `A` as `"other"` everywhere, and `A = "x"; A = runtime` reads `A` as
    `"x"`.
    REFERENCE FORMS that resolve: a bare name (module level, or the enclosing top-level class's
    attribute first when written inside that class body), and `Cls.ATTR` for a top-level class.
    Resolution is transitive to a fixed point over those sites.
    OPERATIONS that fold: str / bytes constants (bytes as latin-1 text, so `b"".join` folds like
    `"".join`); int / float / bool / None constants and `+n` /
    `-n` numeric constants, rendered as `str()`; literal tuples, lists and dicts with literal
    string keys, element-wise; implicit and `+` concatenation of two strings; `sep.join(seq)`
    where `sep` folds to a string and `seq` is a literal or folded tuple / list whose elements
    fold to strings; f-strings whose parts all fold to strings (no format spec, no conversion);
    f-strings nested in f-strings under the same rule.

    NOT CAUGHT, out of scope as obfuscation: aliasing the span-starter function (`start =
    tracer.start_span`), conditional expressions, format specs, conversions and character
    arithmetic (`f"{108:c}"`, `f"{x!r}"`), computed kwarg names (`**{"na" + "me": …}`),
    function-local shadowing.
    NOT CAUGHT, out of scope by rebinding: a name rebound after its use, so the earlier value is
    gone by the time the fixed point settles (`A = "lk"; x = A + ".scan_probe"; A = "other"`).
    NOT CAUGHT, out of scope by enumeration (the list above does not include them): nested
    classes (`Outer.Names.B`, or a span started inside a nested class body); sequence
    concatenation (`("a",) + ("b",)`); an f-string part that folds to a sequence
    (`f"span_{PARTS}"`); complex constants (`1j`); a destructured `_span_name` target.
    NOT CAUGHT, runtime: a parameter, a function result, `%` / `.format`, config, imports, code
    outside `livekit/agents` (plugins), comments. An empty join folds to `""` and reports
    nothing. An f-string with a runtime part is reported by its template
    (`start_span(f"span_{name}")` is flagged as `span_{name}`), not silently passed.
    The per-file allowlist above is the only other exemption."""
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel == "telemetry/_semconv.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders += _offenders(rel, tree, LITERAL_ALLOWLIST.get(rel, set()))
    assert not offenders, "\n".join(offenders)


# CAUGHT: every bypass from Codex rounds 2 to 4 that the contract covers, as must-reject fixtures.
@pytest.mark.parametrize(
    "probe",
    [
        "x = 'lk.scan_probe'",
        'x = "lk" ".scan_probe"',
        'x = "lk\\u002escan_probe"',
        'x = r"lk.scan_probe"',
        "x = f'lk.scan_{probe}'",
        "x = b'lk.scan_probe'",
        'x = rb"lk.scan_probe"',
        'x = "lk" + ".scan_probe"',
        'x = "".join(["lk", ".scan_probe"])',
        'x = f\'{"l" + "k."}scan_probe\'',
        "PREFIX = 'lk'\nx = PREFIX + '.scan_probe'",
        "tracer.start_span('scan_probe')",
        'tracer.start_span (\n    "scan_probe", context=ctx)',
        "tracer.start_span(r'scan_probe')",
        "tracer.start_span('scan_' 'probe')",
        "tracer.start_span(f'scan_{probe}')",
        "tracer.start_as_current_span(name='scan_probe')",
        "span.add_event('scan_probe')",
        "class A:\n    _span_name: ClassVar[str] = 'scan_probe'",
        "NAME = 'scan_probe'\ntracer.start_span(NAME)",
        "tracer.start_span('scan' + '_probe')",
        "tracer.start_span(**{'name': 'scan_probe'})",
        "tracer.start_span(b'scan_probe')",
        "tracer.start_span(f'span_{1}')",
        "class Names:\n    SPAN = 'scan_probe'\ntracer.start_span(Names.SPAN)",
        "NAME, = ('scan_probe',)\ntracer.start_span(NAME)",
        "A, B = ('other', 'scan_probe')\ntracer.start_span(B)",
        "kwargs = {'name': 'scan_probe'}\ntracer.start_span(**kwargs)",
        "NAME = 'scan_probe'\ntracer.start_span(NAME)\ndef f(NAME):\n    pass",
        "PARTS = ('scan', '_probe')\ntracer.start_span(''.join(PARTS))",
        "A = B\nB = C\nC = D\nD = 'scan_probe'\ntracer.start_span(A)",
        "tracer.start_span((('scan_probe')))",
        "N = 'scan_probe'\nclass C:\n    _span_name = N",
        # round 5: one evaluator for bindings and call sites, nested patterns, class scopes
        "PARTS = ('scan', '_probe')\nNAME = ''.join(PARTS)\ntracer.start_span(NAME)",
        "PARTS = ('scan', '_probe')\ntracer.start_span(''.join(PARTS) + '_extra')",
        "class Names:\n    A = 'scan_probe'\n    B = A\ntracer.start_span(Names.B)",
        "class Names:\n    A, B = ('scan', '_probe')\ntracer.start_span(Names.A + Names.B)",
        "(A, (B, C)) = ('scan', ('_', 'probe'))\ntracer.start_span(A + B + C)",
        "NAME = f'span_{1}'\ntracer.start_span(NAME)",
        "A = B\nB = ('scan', '_probe')\ntracer.start_span(''.join(A))",
        "class Names:\n    PARTS = ('scan', '_probe')\ntracer.start_span(''.join(Names.PARTS))",
        "class Names:\n    KWARGS = {'name': 'scan_probe'}\ntracer.start_span(**Names.KWARGS)",
        "A = B\nB = 'scan'\ntracer.start_span(f'{A}_probe')",
        "[A, B] = ['scan', '_probe']\ntracer.start_span(A + B)",
        "SEP = '_'\ntracer.start_span(SEP.join(['scan', 'probe']))",
        # own compositions
        "PARTS = ('scan', '_probe')\ntracer.start_span(f\"{''.join(PARTS)}_{2.5}\")",
        "class Names:\n    P = ('lk', '.scan_probe')\n    Q = ''.join(P)\nx = Names.Q",
        "x = 'lk.'.join(())",  # the separator is itself a literal `lk.` string; no crash
        # round 6: signed numeric constants, nested f-strings, nested literal sequences
        "N = -1\nNAME = f'span_{N}'\ntracer.start_span(NAME)",
        "N = +1\nNAME = f'span_{N}'\ntracer.start_span(NAME)",
        "N = 1\nNAME = f\"span_{f'{N}'}\"\ntracer.start_span(NAME)",
        "(A, [B, C]) = ('scan', ['_', 'probe'])\ntracer.start_span(''.join((A, B, C)))",
        "N = None\nNAME = f'span_{N}'\ntracer.start_span(NAME)",
        "PREFIX = 'lk'\nx = f'{PREFIX}.{-2.5}'",
        "PARTS = (b'scan', b'_probe')\nNAME = b''.join(PARTS)\ntracer.start_span(NAME)",
        # a later binding that does not fold is skipped, so the earlier literal still resolves
        "A = 'scan_probe'\nA = runtime\ntracer.start_span(A)",
    ],
)
def test_the_emitter_site_scan_is_not_fooled_by_spelling(probe: str) -> None:
    assert _offenders("probe.py", ast.parse(probe), set()), probe


# NOT CAUGHT, out of scope: obfuscation. Listed so the boundary is a fact the suite checks, not a
# claim. A change that starts catching one of these is welcome; a change that starts catching
# nothing from the list above is not.
@pytest.mark.parametrize(
    "probe",
    [
        "x = f'{108:c}{107:c}.scan_probe'",  # format spec / character arithmetic
        "x = f'{\"lk.scan_probe\"!r}'",  # conversion
        "start = tracer.start_span\nstart('scan_probe')",  # aliased span-starter
        "tracer.start_span(**{'na' + 'me': 'scan_probe'})",  # computed kwarg name
        "tracer.start_span('scan_probe' if True else 'other')",  # conditional expression
        "tracer.start_span('other' if c else 'scan_probe')",  # conditional expression
        "def f():\n    NAME = 'scan_probe'\n    tracer.start_span(NAME)",  # function-local shadowing
    ],
)
def test_the_emitter_site_scan_leaves_obfuscation_out_of_scope(probe: str) -> None:
    assert not _offenders("probe.py", ast.parse(probe), set()), probe


# NOT CAUGHT, out of scope by enumeration: forms the contract's exact list does not include. Each
# is a fact the suite checks; adding one to the evaluator means moving it to the list above.
@pytest.mark.parametrize(
    "probe",
    [
        # out of scope: nested classes are not a binding site
        "class Outer:\n    class Names:\n        A = 'scan'\n        B = A + '_probe'\ntracer.start_span(Outer.Names.B)",
        "class Outer:\n    class Names:\n        A = 'scan'\n        B = A + '_probe'\n        tracer.start_span(B)",
        "class Outer:\n    class Names:\n        A = 'scan'\n        B = A + '_probe'\n    tracer.start_span(Names.B)",
        "class Outer:\n    class Inner:\n        [A, [B, C]] = ['scan', ['_', 'probe']]\ntracer.start_span(Outer.Inner.A + Outer.Inner.B + Outer.Inner.C)",
        # out of scope: `+` folds strings only, not sequences
        "PARTS = ('scan',) + ('_probe',)\ntracer.start_span(''.join(PARTS))",
        # out of scope: an f-string part must fold to a string, not a sequence
        "PARTS = ('scan', 'probe')\nNAME = f'span_{PARTS}'\ntracer.start_span(NAME)",
        "PREFIX = 'lk'\nPARTS = ('scan', 'probe')\nx = f'{PREFIX}.{PARTS}'",
        # out of scope: complex constants
        "N = 1j\nNAME = f'span_{N}'\ntracer.start_span(NAME)",
        # out of scope: rebinding (the last foldable binding in source order wins)
        "A = 'lk'\nx = A + '.scan_probe'\nA = 'other'",
        "class Names:\n    A = 'lk'\n    A = 'other'\nx = Names.A + '.scan_probe'",
        # out of scope: only a direct `_span_name = …` assignment is checked
        "(_span_name,) = ('scan_probe',)",
    ],
)
def test_the_emitter_site_scan_leaves_unlisted_forms_out_of_scope(probe: str) -> None:
    assert not _offenders("probe.py", ast.parse(probe), set()), probe


# NOT CAUGHT, runtime: the name does not exist until the code runs.
@pytest.mark.parametrize(
    "probe",
    [
        "def f(name):\n    tracer.start_span(name)",
        "tracer.start_span(compute())",
        "tracer.start_span(config['span'])",
        "from elsewhere import NAME\ntracer.start_span(NAME)",
        "tracer.start_span('span_{}'.format(name))",
        "x = PREFIX + '.scan_probe'",  # PREFIX bound elsewhere
        "tracer.start_span(''.join([]))",  # the empty sequence folds to "", nothing to report
        "tracer.start_span(''.join(()))",
        "x = ''.join([])",
        "PARTS = ()\nx = ''.join(PARTS)",
    ],
)
def test_the_emitter_site_scan_documents_its_runtime_boundary(probe: str) -> None:
    assert not _offenders("probe.py", ast.parse(probe), set()), probe


def test_a_runtime_fstring_span_name_is_reported_by_its_template() -> None:
    assert _offenders("probe.py", ast.parse("tracer.start_span(f'span_{name}')"), set()) == [
        "probe.py:1: start_span('span_{name}')"
    ]
