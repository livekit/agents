"""Span attribute and event name constants for LiveKit Agents telemetry.

Attributes carrying conversational content, tool payloads, or other user data
must include a dot-delimited ``pii`` segment (``lk.pii.<name>``): PII-enabled
projects have these attributes stripped at the LiveKit Cloud collector, and the
segment is the only marker it honors. Attributes must never embed such content
in span names, event names, or log message bodies — those are not redactable.
"""

ATTR_PARTICIPANT_ID = "lk.participant_id"
ATTR_PARTICIPANT_IDENTITY = "lk.pii.participant_identity"
ATTR_PARTICIPANT_KIND = "lk.participant_kind"

# session start
ATTR_JOB_ID = "lk.job_id"
ATTR_AGENT_NAME = "lk.agent_name"
ATTR_CLOUD_AGENT_ID = "lk.cloud_agent_id"
ATTR_DEPLOYMENT_ID = "lk.deployment_id"
ATTR_ROOM_NAME = "lk.pii.room_name"

# join keys shared with the server, SIP, and client traces
ATTR_ROOM_SID = "lk.room_sid"
ATTR_DISPATCH_ID = "lk.dispatch_id"
ATTR_WORKER_ID = "lk.job.worker_id"
ATTR_JOB_AGENT_ID = "lk.job.agent_id"

# room connect / room io
ATTR_ROOM_AUTO_SUBSCRIBE = "lk.room.auto_subscribe"
ATTR_ROOM_E2EE = "lk.room.e2ee"
ATTR_ROOM_REMOTE_PARTICIPANT_COUNT = "lk.room.remote_participant_count"
ATTR_ROOM_IO_PARTICIPANT_FILTER = "lk.room_io.participant_filter"
"""Whether RoomIO waited for a specific participant identity (true) or the first eligible one."""

# rpc (``rpc.method`` from the OpenTelemetry RPC semantic conventions, plus lk.rpc.* details)
ATTR_RPC_METHOD = "rpc.method"
ATTR_RPC_REQUEST_ID = "lk.rpc.request_id"
ATTR_RPC_CALLER_IDENTITY = "lk.rpc.caller_identity"
ATTR_RPC_DESTINATION_IDENTITY = "lk.rpc.destination_identity"
ATTR_RPC_PAYLOAD = "lk.pii.rpc.payload"
"""Request payload, truncated to ``telemetry.traces.MAX_PAYLOAD_ATTR_LEN`` characters."""
ATTR_RPC_PAYLOAD_SIZE = "lk.rpc.payload_size"
ATTR_RPC_RESPONSE = "lk.pii.rpc.response"
"""Response payload, truncated like the request."""
ATTR_RPC_RESPONSE_SIZE = "lk.rpc.response_size"
ATTR_RPC_RESPONSE_TIMEOUT = "lk.rpc.response_timeout"
ATTR_RPC_ERROR_CODE = "lk.rpc.error_code"
ATTR_RPC_HANDLER_REGISTERED = "lk.rpc.handler_registered"
"""False when a caller invoked a method this participant never registered."""

ATTR_SHUTDOWN_REASON = "lk.shutdown.reason"
"""The string passed to ``JobContext.shutdown(reason=...)``; developer-authored, like a log line."""
ATTR_SHUTDOWN_USER_INITIATED = "lk.shutdown.user_initiated"
ATTR_CALLBACK_NAME = "lk.callback.name"


ATTR_FUNCTION_TOOL_NAME = "lk.function_tool.name"
ATTR_FUNCTION_TOOL_ARGS = "lk.pii.function_tool.arguments"
ATTR_FUNCTION_TOOL_IS_ERROR = "lk.function_tool.is_error"
ATTR_FUNCTION_TOOL_OUTPUT = "lk.pii.function_tool.output"

# OpenTelemetry GenAI semantic conventions, mirroring the attribute registry of
# https://github.com/open-telemetry/semantic-conventions-genai. Backends ingest these
# directly, so the names must stay byte-for-byte identical to the registry. The ones the
# spec flags as sensitive are redacted by name in `lk_agents_core::telemetry::policy`, since a
# standard name cannot carry the `lk.pii.` marker segment.

ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

ATTR_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
ATTR_GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
ATTR_GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
ATTR_GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"

ATTR_ERROR_TYPE = "error.type"


# Unofficial OpenTelemetry GenAI attributes, these are namespaces recognised by LangFuse
# https://langfuse.com/integrations/native/opentelemetry#usage
# but not in the official OpenTelemetry specification. Emitted alongside the official
# ``gen_ai.usage.*.{input,output}_tokens`` names above.
ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS = "gen_ai.usage.input_text_tokens"
ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS = "gen_ai.usage.input_cached_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS = "gen_ai.usage.output_text_tokens"

# Exception attributes
ATTR_EXCEPTION_TRACE = "exception.stacktrace"
ATTR_EXCEPTION_TYPE = "exception.type"
ATTR_EXCEPTION_MESSAGE = "exception.message"
