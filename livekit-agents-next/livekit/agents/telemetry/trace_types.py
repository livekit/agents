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

# job dispatch timeline (job_entrypoint and job_dispatch spans). The stage instants are
# timestamped events (job_received, job_accepted, job_assigned, process_assigned,
# entrypoint_started); these attributes are the seconds between adjacent stages, so the
# chain reads off the span without timestamp arithmetic. They sum to the dispatch latency.
ATTR_JOB_ACCEPT_LATENCY = "lk.job.accept_latency"
"""Seconds from the availability request to the worker's accept (the request handler)."""
ATTR_JOB_ASSIGNMENT_LATENCY = "lk.job.assignment_latency"
"""Seconds from the accept to the server's assignment (a server round trip)."""
ATTR_JOB_LAUNCH_LATENCY = "lk.job.launch_latency"
"""Seconds from the assignment to a process taking the job (pool acquisition)."""
ATTR_JOB_ENTRYPOINT_LATENCY = "lk.job.entrypoint_latency"
"""Seconds from the process taking the job to the user entrypoint running in it."""
ATTR_JOB_DISPATCH_LATENCY = "lk.job.dispatch_latency"
"""Seconds from the availability request to the entrypoint running: the whole chain."""


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


"""Times the endpointing wait restarted on a later trigger (late transcript, VAD)."""


# OpenTelemetry GenAI semantic conventions, mirroring the attribute registry of
# https://github.com/open-telemetry/semantic-conventions-genai. Backends ingest these
# directly, so the names must stay byte-for-byte identical to the registry. The ones the
# spec flags as sensitive are listed in `telemetry.traces.GEN_AI_PII_ATTRIBUTES`, since a
# standard name cannot carry the `lk.pii.` marker segment.

ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"


ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

ATTR_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"


ATTR_GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
ATTR_GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ATTR_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

ATTR_GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
ATTR_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
ATTR_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"

ATTR_GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"
ATTR_GEN_AI_MEMORY_QUERY_TEXT = "gen_ai.memory.query.text"
ATTR_GEN_AI_MEMORY_RECORDS = "gen_ai.memory.records"
ATTR_GEN_AI_EVALUATION_NAME = "gen_ai.evaluation.name"
ATTR_GEN_AI_EVALUATION_SCORE_LABEL = "gen_ai.evaluation.score.label"
ATTR_GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
ATTR_GEN_AI_PROMPT_VARIABLE = "gen_ai.prompt.variable"  # template: gen_ai.prompt.variable.<key>

ATTR_ERROR_TYPE = "error.type"


# Unofficial OpenTelemetry GenAI attributes, these are namespaces recognised by LangFuse
# https://langfuse.com/integrations/native/opentelemetry#usage
# but not in the official OpenTelemetry specification. Emitted alongside the official
# ``gen_ai.usage.*.{input,output}_tokens`` names above.
ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS = "gen_ai.usage.input_text_tokens"
ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS = "gen_ai.usage.input_cached_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS = "gen_ai.usage.output_text_tokens"

# OpenTelemetry GenAI event names (for structured logging)
EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"
EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS = "gen_ai.client.inference.operation.details"


# Exception attributes
ATTR_EXCEPTION_TRACE = "exception.stacktrace"
ATTR_EXCEPTION_TYPE = "exception.type"
ATTR_EXCEPTION_MESSAGE = "exception.message"


# Event loop blocking (telemetry/loop_monitor.py)
ATTR_BLOCKING_DURATION = "lk.blocking.duration"
"""Seconds the event loop was blocked, measured to within one heartbeat interval."""
ATTR_BLOCKING_THRESHOLD = "lk.blocking.threshold"
ATTR_BLOCKING_SEVERITY = "lk.blocking.severity"
"""``warning`` or ``error``, by which threshold the block crossed."""
ATTR_BLOCKING_TASK = "lk.blocking.task"
"""Name of the asyncio task that was running when the loop thread was sampled."""
ATTR_BLOCKING_STACK = "lk.blocking.stack"
"""Loop-thread stack sampled while blocked (source locations only, no values)."""
ATTR_BLOCKING_GC_TIME = "lk.blocking.gc_time"
"""Seconds of garbage collection that ran on the loop thread during the block."""
ATTR_BLOCKING_CPU_TIME = "lk.blocking.cpu_time"
"""CPU seconds consumed by the loop thread during the block; near zero means it was waiting."""
ATTR_BLOCKING_IMPORT = "lk.blocking.import"
"""The module a lazy import was loading when the loop thread was sampled: import it at warm-up."""
ATTR_BLOCKING_SUPPRESSED = "lk.blocking.suppressed"
"""Reports dropped by rate limiting since the previous emitted span."""
