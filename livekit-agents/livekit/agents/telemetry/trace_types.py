"""Span attribute and event name constants for LiveKit Agents telemetry.

Attributes carrying conversational content, tool payloads, or other user data
must include a dot-delimited ``pii`` segment (``lk.pii.<name>``): PII-enabled
projects have these attributes stripped at the LiveKit Cloud collector, and the
segment is the only marker it honors. Attributes must never embed such content
in span names, event names, or log message bodies — those are not redactable.
"""

from .. import _semconv

# The strings come from the protocol registry (`_semconv`, vendored by `lk-agents-codegen`);
# these are the names the package has always used for them. `tests/test_semconv_pins.py` holds
# the old literals. The few literals left below are keys the registry does not carry yet.


ATTR_SPEECH_ID = _semconv.LK_SPEECH_ID
ATTR_AGENT_LABEL = _semconv.LK_AGENT_LABEL
ATTR_RETRY_COUNT = _semconv.LK_RETRY_COUNT
ATTR_PROVIDER_REQUEST_IDS = _semconv.LK_PROVIDER_REQUEST_IDS
"""Provider-known correlation ids associated with this span (list[str]).

Populated by STT/TTS plugins when the id is either sent to the provider
(e.g. WS context_id) or returned by it (e.g. response request_id / session_id),
so it can be cross-referenced with the provider's logs for debugging."""


ATTR_PARTICIPANT_ID = _semconv.LK_PARTICIPANT_ID
ATTR_PARTICIPANT_IDENTITY = _semconv.LK_PII_PARTICIPANT_IDENTITY
ATTR_PARTICIPANT_KIND = _semconv.LK_PARTICIPANT_KIND

# session start
ATTR_JOB_ID = _semconv.LK_JOB_ID
ATTR_AGENT_NAME = _semconv.LK_AGENT_NAME
ATTR_CLOUD_AGENT_ID = _semconv.LK_CLOUD_AGENT_ID
ATTR_DEPLOYMENT_ID = _semconv.LK_DEPLOYMENT_ID
ATTR_ROOM_NAME = _semconv.LK_PII_ROOM_NAME

# join keys shared with the server, SIP, and client traces
ATTR_ROOM_SID = _semconv.LK_ROOM_SID
ATTR_DISPATCH_ID = _semconv.LK_DISPATCH_ID
ATTR_WORKER_ID = _semconv.LK_JOB_WORKER_ID
ATTR_JOB_AGENT_ID = _semconv.LK_JOB_AGENT_ID
ATTR_SIP_PREFIX = "lk.sip."  # ponytail: not in the registry, see the conflict list
"""Prefix under which a linked SIP participant's ``sip.*`` attributes are copied (call id,
trunk id and number, rule id, hostname, status, headers)."""
ATTR_SIP_PHONE_NUMBER = _semconv.LK_PII_SIP_PHONE_NUMBER
"""The end user's phone number (``sip.phoneNumber``), the one SIP attribute that is PII."""

# job dispatch timeline (job_entrypoint and job_dispatch spans). The stage instants are
# timestamped events (job_received, job_accepted, job_assigned, process_assigned,
# entrypoint_started); these attributes are the seconds between adjacent stages, so the
# chain reads off the span without timestamp arithmetic. They sum to the dispatch latency.
ATTR_JOB_ACCEPT_LATENCY = _semconv.LK_JOB_ACCEPT_LATENCY
"""Seconds from the availability request to the worker's accept (the request handler)."""
ATTR_JOB_ASSIGNMENT_LATENCY = _semconv.LK_JOB_ASSIGNMENT_LATENCY
"""Seconds from the accept to the server's assignment (a server round trip)."""
ATTR_JOB_LAUNCH_LATENCY = _semconv.LK_JOB_LAUNCH_LATENCY
"""Seconds from the assignment to a process taking the job (pool acquisition)."""
ATTR_JOB_ENTRYPOINT_LATENCY = _semconv.LK_JOB_ENTRYPOINT_LATENCY
"""Seconds from the process taking the job to the user entrypoint running in it."""
ATTR_JOB_DISPATCH_LATENCY = _semconv.LK_JOB_DISPATCH_LATENCY
"""Seconds from the availability request to the entrypoint running: the whole chain."""

# keyterm detection (keyterm_detection span): counts only, the terms themselves are the
# customer's vocabulary and travel as lk.pii.keyterms in the session report
ATTR_KEYTERMS_COUNT = _semconv.LK_KEYTERMS_COUNT
"""Keyterms in effect after the pass (static + confirmed)."""
ATTR_KEYTERMS_ADDED = _semconv.LK_KEYTERMS_ADDED
ATTR_KEYTERMS_REMOVED = _semconv.LK_KEYTERMS_REMOVED

# room connect / room io
ATTR_ROOM_AUTO_SUBSCRIBE = _semconv.LK_ROOM_AUTO_SUBSCRIBE
ATTR_ROOM_E2EE = _semconv.LK_ROOM_E2EE
ATTR_ROOM_REMOTE_PARTICIPANT_COUNT = _semconv.LK_ROOM_REMOTE_PARTICIPANT_COUNT
ATTR_ROOM_IO_PARTICIPANT_FILTER = _semconv.LK_ROOM_IO_PARTICIPANT_FILTER
"""Whether RoomIO waited for a specific participant identity (true) or the first eligible one."""
ATTR_TRACK_SID = _semconv.LK_TRACK_SID
ATTR_TRACK_SOURCE = _semconv.LK_TRACK_SOURCE
ATTR_FIRST_FRAME_DELAY = _semconv.LK_FIRST_FRAME_DELAY
"""Seconds from linking the participant to the first media frame received from them."""
ATTR_PRE_CONNECT_AUDIO_DURATION = _semconv.LK_PRE_CONNECT_AUDIO_DURATION
ATTR_CONNECTION_STATE = _semconv.LK_CONNECTION_STATE
ATTR_DISCONNECT_REASON = _semconv.LK_DISCONNECT_REASON
ATTR_OLD_STATE = _semconv.LK_OLD_STATE
ATTR_NEW_STATE = _semconv.LK_NEW_STATE

# rpc (``rpc.method`` from the OpenTelemetry RPC semantic conventions, plus lk.rpc.* details)
ATTR_RPC_METHOD = _semconv.RPC_METHOD
ATTR_RPC_REQUEST_ID = _semconv.LK_RPC_REQUEST_ID
ATTR_RPC_CALLER_IDENTITY = _semconv.LK_RPC_CALLER_IDENTITY
ATTR_RPC_DESTINATION_IDENTITY = _semconv.LK_RPC_DESTINATION_IDENTITY
ATTR_RPC_PAYLOAD = _semconv.LK_PII_RPC_PAYLOAD
"""Request payload, truncated to ``telemetry.rpc.MAX_PAYLOAD_ATTR_LEN`` characters."""
ATTR_RPC_PAYLOAD_SIZE = _semconv.LK_RPC_PAYLOAD_SIZE
ATTR_RPC_RESPONSE = _semconv.LK_PII_RPC_RESPONSE
"""Response payload, truncated like the request."""
ATTR_RPC_RESPONSE_SIZE = _semconv.LK_RPC_RESPONSE_SIZE
ATTR_RPC_RESPONSE_TIMEOUT = _semconv.LK_RPC_RESPONSE_TIMEOUT
ATTR_RPC_ERROR_CODE = _semconv.LK_RPC_ERROR_CODE
ATTR_RPC_HANDLER_REGISTERED = _semconv.LK_RPC_HANDLER_REGISTERED
"""False when a caller invoked a method this participant never registered."""

# session close / job shutdown
ATTR_CLOSE_REASON = _semconv.LK_CLOSE_REASON
ATTR_CLOSE_DRAIN = _semconv.LK_CLOSE_DRAIN
ATTR_SHUTDOWN_REASON = _semconv.LK_SHUTDOWN_REASON
"""The string passed to ``JobContext.shutdown(reason=...)``; developer-authored, like a log line."""
ATTR_SHUTDOWN_USER_INITIATED = _semconv.LK_SHUTDOWN_USER_INITIATED
ATTR_CALLBACK_NAME = _semconv.LK_CALLBACK_NAME

# agent turn
ATTR_AGENT_TURN_ID = _semconv.LK_GENERATION_ID
"""On ``agent_turn``: the latest generation (LLM step) of the speech; each step is also a
``generation`` event carrying its own id."""
ATTR_AGENT_PARENT_TURN_ID = _semconv.LK_PARENT_GENERATION_ID
ATTR_GENERATION_COUNT = _semconv.LK_GENERATION_COUNT
"""On ``agent_turn``: how many generations (LLM steps) the speech took; more than one means
tool calls were executed before the final reply."""
ATTR_USER_INPUT = _semconv.LK_PII_USER_INPUT
ATTR_INSTRUCTIONS = _semconv.LK_PII_INSTRUCTIONS
ATTR_SPEECH_INTERRUPTED = _semconv.LK_INTERRUPTED

# llm node
ATTR_CHAT_CTX = _semconv.LK_PII_CHAT_CTX
ATTR_FUNCTION_TOOLS = _semconv.LK_FUNCTION_TOOLS
ATTR_PROVIDER_TOOLS = _semconv.LK_PROVIDER_TOOLS
ATTR_TOOL_SETS = _semconv.LK_TOOL_SETS
ATTR_RESPONSE_TEXT = _semconv.LK_PII_RESPONSE_TEXT
ATTR_RESPONSE_FUNCTION_CALLS = _semconv.LK_PII_RESPONSE_FUNCTION_CALLS
ATTR_RESPONSE_TTFT = _semconv.LK_RESPONSE_TTFT

# input delta (RecordingOptions.input_delta)
ATTR_INPUT_DELTA = _semconv.LK_INPUT_DELTA
"""The span's record (lk.pii.chat_ctx on llm_node, gen_ai.input.messages on llm_request)
continues the one on its parent, ``lk.input.base_span_id``. See ``telemetry/input_delta.py``
for the procedure and the rebuild rule."""
ATTR_INPUT_BASE_SPAN_ID = _semconv.LK_INPUT_BASE_SPAN_ID
ATTR_INPUT_DROPPED_FROM_BASE = _semconv.LK_INPUT_DROPPED_FROM_BASE
"""Rebuild: the parent's full record without its last N entries, followed by this span's.
Without gen_ai.system_instructions on an llm_request span, the parent's apply."""

# function tool
ATTR_FUNCTION_TOOL_ID = _semconv.LK_FUNCTION_TOOL_ID
ATTR_FUNCTION_TOOL_NAME = _semconv.LK_FUNCTION_TOOL_NAME
ATTR_FUNCTION_TOOL_ARGS = _semconv.LK_PII_FUNCTION_TOOL_ARGUMENTS
ATTR_FUNCTION_TOOL_IS_ERROR = _semconv.LK_FUNCTION_TOOL_IS_ERROR
ATTR_FUNCTION_TOOL_OUTPUT = _semconv.LK_PII_FUNCTION_TOOL_OUTPUT

# tts node
ATTR_TTS_INPUT_TEXT = _semconv.LK_PII_INPUT_TEXT
ATTR_TTS_STREAMING = _semconv.LK_TTS_STREAMING
ATTR_TTS_LABEL = _semconv.LK_TTS_LABEL
ATTR_RESPONSE_TTFB = _semconv.LK_RESPONSE_TTFB

# eou detection
ATTR_EOU_PROBABILITY = _semconv.LK_EOU_PROBABILITY
ATTR_EOU_UNLIKELY_THRESHOLD = _semconv.LK_EOU_UNLIKELY_THRESHOLD
ATTR_EOU_DELAY = _semconv.LK_EOU_ENDPOINTING_DELAY
ATTR_EOU_LANGUAGE = _semconv.LK_EOU_LANGUAGE
ATTR_USER_TRANSCRIPT = _semconv.LK_PII_USER_TRANSCRIPT
ATTR_STT_EVENTS = _semconv.LK_STT_EVENTS
"""JSON list of transcript events on user_turn, in arrival order. Each entry has
received_at (SpeechEvent.created_at, Unix seconds), type, and transcript_length
(characters in the first alternative)."""
ATTR_TRANSCRIPT_CONFIDENCE = _semconv.LK_TRANSCRIPT_CONFIDENCE
ATTR_TRANSCRIPTION_DELAY = _semconv.LK_TRANSCRIPTION_DELAY
ATTR_END_OF_TURN_DELAY = _semconv.LK_END_OF_TURN_DELAY
ATTR_EOU_SOURCE = _semconv.LK_EOU_SOURCE
ATTR_EOU_DETECTION_DELAY = _semconv.LK_EOU_DETECTION_DELAY
ATTR_EOU_FROM_CACHE = _semconv.LK_EOU_FROM_CACHE
# eou_wait span: from the user's last speech to the turn decision
ATTR_EOU_OUTCOME = _semconv.LK_EOU_OUTCOME
"""How the wait ended: ``committed``, ``user_resumed``, or ``dropped``."""
ATTR_EOU_WAIT_DURATION = _semconv.LK_EOU_WAIT_DURATION
"""Seconds from the end of the user's speech to the turn decision."""
ATTR_EOU_REARM_COUNT = _semconv.LK_EOU_REARM_COUNT
ATTR_EOU_NOT_COMMITTED_COUNT = _semconv.LK_EOU_NOT_COMMITTED_COUNT
"""Turn decisions the wait rejected (the detector said the user was not done) before it ended."""
ATTR_EOU_RESUME_COUNT = _semconv.LK_EOU_RESUME_COUNT
"""On user_turn: endpointing waits the user cut short by speaking again."""
ATTR_ON_USER_TURN_COMPLETED_DELAY = _semconv.LK_ON_USER_TURN_COMPLETED_DELAY
"""Seconds the on_user_turn_completed hook took; on the reply's agent_turn with the other stages."""
"""Times the endpointing wait restarted on a later trigger (late transcript, VAD)."""

# speech scheduling
ATTR_SPEECH_QUEUE_WAIT = _semconv.LK_SPEECH_QUEUE_WAIT
"""Seconds a speech handle waited in the queue before generation was authorized."""

# metrics
ATTR_LLM_METRICS = _semconv.LK_LLM_METRICS
ATTR_TTS_METRICS = _semconv.LK_TTS_METRICS
ATTR_REALTIME_MODEL_METRICS = _semconv.LK_REALTIME_MODEL_METRICS

# latency span attributes
ATTR_E2E_LATENCY = _semconv.LK_E2E_LATENCY

# OpenTelemetry GenAI semantic conventions, mirroring the attribute registry of
# https://github.com/open-telemetry/semantic-conventions-genai. Backends ingest these
# directly, so the names must stay byte-for-byte identical to the registry. The ones the
# spec flags as sensitive are listed in `telemetry.pii.GEN_AI_PII_ATTRIBUTES`, since a
# standard name cannot carry the `lk.pii.` marker segment.

ATTR_GEN_AI_OPERATION_NAME = _semconv.GEN_AI_OPERATION_NAME
ATTR_GEN_AI_PROVIDER_NAME = _semconv.GEN_AI_PROVIDER_NAME

ATTR_GEN_AI_REQUEST_MODEL = _semconv.GEN_AI_REQUEST_MODEL
ATTR_GEN_AI_REQUEST_STREAM = _semconv.GEN_AI_REQUEST_STREAM

ATTR_GEN_AI_RESPONSE_ID = _semconv.GEN_AI_RESPONSE_ID
ATTR_GEN_AI_RESPONSE_MODEL = _semconv.GEN_AI_RESPONSE_MODEL
ATTR_GEN_AI_RESPONSE_FINISH_REASONS = _semconv.GEN_AI_RESPONSE_FINISH_REASONS
ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = _semconv.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK

ATTR_GEN_AI_USAGE_INPUT_TOKENS = _semconv.GEN_AI_USAGE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = _semconv.GEN_AI_USAGE_OUTPUT_TOKENS
ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = _semconv.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
ATTR_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = _semconv.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = _semconv.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
ATTR_GEN_AI_USAGE_TEXT_INPUT_TOKENS = _semconv.GEN_AI_USAGE_TEXT_INPUT_TOKENS
ATTR_GEN_AI_USAGE_TEXT_OUTPUT_TOKENS = _semconv.GEN_AI_USAGE_TEXT_OUTPUT_TOKENS
ATTR_GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS = _semconv.GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS
ATTR_GEN_AI_USAGE_AUDIO_INPUT_TOKENS = _semconv.GEN_AI_USAGE_AUDIO_INPUT_TOKENS
ATTR_GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS = _semconv.GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS
ATTR_GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS = (
    _semconv.GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS
)
ATTR_GEN_AI_USAGE_IMAGE_INPUT_TOKENS = _semconv.GEN_AI_USAGE_IMAGE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS = (
    _semconv.GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS
)
ATTR_GEN_AI_TOKEN_TYPE = _semconv.GEN_AI_TOKEN_TYPE

ATTR_GEN_AI_CONVERSATION_ID = _semconv.GEN_AI_CONVERSATION_ID

ATTR_GEN_AI_AGENT_NAME = _semconv.GEN_AI_AGENT_NAME

ATTR_GEN_AI_TOOL_NAME = _semconv.GEN_AI_TOOL_NAME
ATTR_GEN_AI_TOOL_CALL_ID = _semconv.GEN_AI_TOOL_CALL_ID
ATTR_GEN_AI_TOOL_DESCRIPTION = _semconv.GEN_AI_TOOL_DESCRIPTION
ATTR_GEN_AI_TOOL_TYPE = _semconv.GEN_AI_TOOL_TYPE
ATTR_GEN_AI_TOOL_CALL_ARGUMENTS = _semconv.GEN_AI_TOOL_CALL_ARGUMENTS
ATTR_GEN_AI_TOOL_CALL_RESULT = _semconv.GEN_AI_TOOL_CALL_RESULT
ATTR_GEN_AI_TOOL_DEFINITIONS = _semconv.GEN_AI_TOOL_DEFINITIONS

ATTR_GEN_AI_SYSTEM_INSTRUCTIONS = _semconv.GEN_AI_SYSTEM_INSTRUCTIONS
ATTR_GEN_AI_INPUT_MESSAGES = _semconv.GEN_AI_INPUT_MESSAGES
ATTR_GEN_AI_OUTPUT_MESSAGES = _semconv.GEN_AI_OUTPUT_MESSAGES
ATTR_GEN_AI_OUTPUT_TYPE = _semconv.GEN_AI_OUTPUT_TYPE

ATTR_GEN_AI_RETRIEVAL_DOCUMENTS = _semconv.GEN_AI_RETRIEVAL_DOCUMENTS
ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT = _semconv.GEN_AI_RETRIEVAL_QUERY_TEXT
ATTR_GEN_AI_MEMORY_QUERY_TEXT = _semconv.GEN_AI_MEMORY_QUERY_TEXT
ATTR_GEN_AI_MEMORY_RECORDS = _semconv.GEN_AI_MEMORY_RECORDS
ATTR_GEN_AI_EVALUATION_NAME = _semconv.GEN_AI_EVALUATION_NAME
ATTR_GEN_AI_EVALUATION_SCORE_LABEL = _semconv.GEN_AI_EVALUATION_SCORE_LABEL
ATTR_GEN_AI_EVALUATION_EXPLANATION = _semconv.GEN_AI_EVALUATION_EXPLANATION
ATTR_GEN_AI_PROMPT_VARIABLE = (
    _semconv.GEN_AI_PROMPT_VARIABLE
)  # template: gen_ai.prompt.variable.<key>
ATTR_GEN_AI_WORKFLOW_NAME = _semconv.GEN_AI_WORKFLOW_NAME

ATTR_ERROR_TYPE = _semconv.ERROR_TYPE


class GenAIOperationName:
    """Well-known ``gen_ai.operation.name`` values."""

    CHAT = _semconv.GEN_AI_OPERATION_NAME_CHAT
    GENERATE_CONTENT = _semconv.GEN_AI_OPERATION_NAME_GENERATE_CONTENT
    TEXT_COMPLETION = _semconv.GEN_AI_OPERATION_NAME_TEXT_COMPLETION
    EMBEDDINGS = _semconv.GEN_AI_OPERATION_NAME_EMBEDDINGS
    RETRIEVAL = _semconv.GEN_AI_OPERATION_NAME_RETRIEVAL
    FETCH_RESPONSE = _semconv.GEN_AI_OPERATION_NAME_FETCH_RESPONSE
    CREATE_AGENT = _semconv.GEN_AI_OPERATION_NAME_CREATE_AGENT
    INVOKE_AGENT = _semconv.GEN_AI_OPERATION_NAME_INVOKE_AGENT
    EXECUTE_TOOL = _semconv.GEN_AI_OPERATION_NAME_EXECUTE_TOOL
    INVOKE_WORKFLOW = _semconv.GEN_AI_OPERATION_NAME_INVOKE_WORKFLOW
    PLAN = _semconv.GEN_AI_OPERATION_NAME_PLAN
    SEARCH_MEMORY = _semconv.GEN_AI_OPERATION_NAME_SEARCH_MEMORY
    CREATE_MEMORY = _semconv.GEN_AI_OPERATION_NAME_CREATE_MEMORY
    UPDATE_MEMORY = _semconv.GEN_AI_OPERATION_NAME_UPDATE_MEMORY
    UPSERT_MEMORY = _semconv.GEN_AI_OPERATION_NAME_UPSERT_MEMORY
    DELETE_MEMORY = _semconv.GEN_AI_OPERATION_NAME_DELETE_MEMORY
    CREATE_MEMORY_STORE = _semconv.GEN_AI_OPERATION_NAME_CREATE_MEMORY_STORE
    DELETE_MEMORY_STORE = _semconv.GEN_AI_OPERATION_NAME_DELETE_MEMORY_STORE


class GenAIOutputType:
    """Well-known ``gen_ai.output.type`` values."""

    TEXT = _semconv.GEN_AI_OUTPUT_TYPE_TEXT
    JSON = _semconv.GEN_AI_OUTPUT_TYPE_JSON
    IMAGE = _semconv.GEN_AI_OUTPUT_TYPE_IMAGE
    SPEECH = _semconv.GEN_AI_OUTPUT_TYPE_SPEECH


class GenAIFinishReason:
    """Well-known ``gen_ai.response.finish_reasons`` values."""

    STOP = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_STOP
    LENGTH = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_LENGTH
    CONTENT_FILTER = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_CONTENT_FILTER
    TOOL_CALL = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_TOOL_CALL
    COMPACTION = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_COMPACTION
    ERROR = _semconv.GEN_AI_RESPONSE_FINISH_REASONS_ERROR


GEN_AI_PROVIDER_NAMES: frozenset[str] = frozenset(
    {
        "openai",
        "gcp.gen_ai",
        "gcp.vertex_ai",
        "gcp.gemini",
        "anthropic",
        "cohere",
        "azure.ai.inference",
        "azure.ai.openai",
        "ibm.watsonx.ai",
        "aws.bedrock",
        "perplexity",
        "x_ai",
        "deepseek",
        "groq",
        _semconv.GEN_AI_PROVIDER_NAME_MISTRAL_AI,
        "moonshot_ai",
    }
)
"""The ``gen_ai.provider.name`` values the registry enumerates.

For a provider on this list the convention says the registry spelling MUST be used, since
backends treat the attribute as the discriminator for provider-specific parsing. A provider
that is not on it MAY report a custom value, so those pass through untouched.
"""

# Plugins report `provider` either as a display name ("AWS Bedrock", "MistralAI") or, for the
# OpenAI-compatible clients, as the base URL's host ("api.openai.com"). Both are mapped here:
# by host first, then by the display name reduced to lowercase alphanumerics, so
# "AWS Bedrock" / "aws_bedrock" / "awsbedrock" all resolve alike.
_PROVIDER_BY_HOST: dict[str, str] = {
    "api.anthropic.com": "anthropic",
    "api.cohere.ai": "cohere",
    "api.cohere.com": "cohere",
    "api.deepseek.com": "deepseek",
    "api.groq.com": "groq",
    "api.mistral.ai": _semconv.GEN_AI_PROVIDER_NAME_MISTRAL_AI,
    "api.moonshot.ai": "moonshot_ai",
    "api.moonshot.cn": "moonshot_ai",
    "api.openai.com": "openai",
    "api.perplexity.ai": "perplexity",
    "api.x.ai": "x_ai",
    "generativelanguage.googleapis.com": "gcp.gemini",
}

_PROVIDER_BY_HOST_SUFFIX: tuple[tuple[str, str], ...] = (
    (".openai.azure.com", "azure.ai.openai"),
    (".services.ai.azure.com", "azure.ai.inference"),
    (".aiplatform.googleapis.com", "gcp.vertex_ai"),
)

_PROVIDER_BY_NAME: dict[str, str] = {
    "amazon": "aws.bedrock",
    "amazonbedrock": "aws.bedrock",
    "anthropic": "anthropic",
    "awsbedrock": "aws.bedrock",
    "azureaiinference": "azure.ai.inference",
    "azureopenai": "azure.ai.openai",
    "bedrock": "aws.bedrock",
    "cohere": "cohere",
    "deepseek": "deepseek",
    "gemini": "gcp.gemini",
    "google": "gcp.gen_ai",
    "googlecloudplatform": "gcp.gen_ai",
    "googlegenai": "gcp.gen_ai",
    "groq": "groq",
    "ibmwatsonxai": "ibm.watsonx.ai",
    "mistral": _semconv.GEN_AI_PROVIDER_NAME_MISTRAL_AI,
    "mistralai": _semconv.GEN_AI_PROVIDER_NAME_MISTRAL_AI,
    "moonshot": "moonshot_ai",
    "moonshotai": "moonshot_ai",
    "openai": "openai",
    "perplexity": "perplexity",
    "vertexai": "gcp.vertex_ai",
    "vertexaimodelgarden": "gcp.vertex_ai",
    "watsonx": "ibm.watsonx.ai",
    "xai": "x_ai",
}


def gen_ai_provider_name(provider: str | None) -> str | None:
    """Normalize a LiveKit plugin's ``provider`` to its GenAI registry spelling."""
    if not provider or not (value := provider.strip()):
        return None

    host = value.lower()
    if (mapped := _PROVIDER_BY_HOST.get(host)) is not None:
        return mapped
    for suffix, mapped in _PROVIDER_BY_HOST_SUFFIX:
        if host.endswith(suffix):
            return mapped
    # only the Bedrock endpoints, not every AWS service that shares the domain
    if host.startswith("bedrock") and host.endswith(".amazonaws.com"):
        return "aws.bedrock"

    canonical = "".join(c for c in host if c.isalnum())
    # a provider outside the registry keeps its own id, which the convention allows
    return _PROVIDER_BY_NAME.get(canonical, value)


# Unofficial OpenTelemetry GenAI attributes, these are namespaces recognised by LangFuse
# https://langfuse.com/integrations/native/opentelemetry#usage
# but not in the official OpenTelemetry specification. Emitted alongside the official
# ``gen_ai.usage.*.{input,output}_tokens`` names above.
ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS = _semconv.GEN_AI_USAGE_INPUT_TEXT_TOKENS
ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS = _semconv.GEN_AI_USAGE_INPUT_AUDIO_TOKENS
ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS = _semconv.GEN_AI_USAGE_INPUT_CACHED_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS = _semconv.GEN_AI_USAGE_OUTPUT_TEXT_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS = _semconv.GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS
ATTR_GEN_AI_USAGE_REASONING_TOKENS = _semconv.GEN_AI_USAGE_REASONING_TOKENS

# OpenTelemetry GenAI event names (for structured logging)
EVENT_GEN_AI_SYSTEM_MESSAGE = _semconv.EVENT_GEN_AI_SYSTEM_MESSAGE
EVENT_GEN_AI_USER_MESSAGE = _semconv.EVENT_GEN_AI_USER_MESSAGE
EVENT_GEN_AI_ASSISTANT_MESSAGE = _semconv.EVENT_GEN_AI_ASSISTANT_MESSAGE
EVENT_GEN_AI_TOOL_MESSAGE = _semconv.EVENT_GEN_AI_TOOL_MESSAGE
EVENT_GEN_AI_CHOICE = _semconv.EVENT_GEN_AI_CHOICE
EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS = (
    _semconv.CHECKPOINT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS
)

# OpenTelemetry GenAI metric names
METRIC_GEN_AI_CLIENT_TOKEN_USAGE = _semconv.METRIC_GEN_AI_CLIENT_TOKEN_USAGE
METRIC_GEN_AI_CLIENT_OPERATION_DURATION = _semconv.METRIC_GEN_AI_CLIENT_OPERATION_DURATION
METRIC_GEN_AI_CLIENT_TIME_TO_FIRST_CHUNK = (
    _semconv.METRIC_GEN_AI_CLIENT_OPERATION_TIME_TO_FIRST_CHUNK
)
METRIC_GEN_AI_INVOKE_AGENT_DURATION = _semconv.METRIC_GEN_AI_INVOKE_AGENT_DURATION
METRIC_GEN_AI_EXECUTE_TOOL_DURATION = _semconv.METRIC_GEN_AI_EXECUTE_TOOL_DURATION

# Exception attributes
ATTR_EXCEPTION_TRACE = _semconv.EXCEPTION_STACKTRACE
ATTR_EXCEPTION_TYPE = _semconv.EXCEPTION_TYPE
ATTR_EXCEPTION_MESSAGE = _semconv.EXCEPTION_MESSAGE

# Platform-specific attributes
ATTR_LANGFUSE_COMPLETION_START_TIME = _semconv.LANGFUSE_OBSERVATION_COMPLETION_START_TIME

# AMD (Answering Machine Detection) attributes
ATTR_AMD_CATEGORY = _semconv.LK_AMD_CATEGORY
ATTR_AMD_REASON = _semconv.LK_AMD_REASON
ATTR_AMD_SPEECH_DURATION = _semconv.LK_AMD_SPEECH_DURATION
ATTR_AMD_DELAY = _semconv.LK_AMD_DELAY
ATTR_AMD_TRANSCRIPT = _semconv.LK_PII_AMD_TRANSCRIPT

# Interruptions (agent_turn)
ATTR_INTERRUPTION_SOURCE = _semconv.LK_INTERRUPTION_SOURCE
"""What interrupted the speech: ``audio_activity`` (barge-in), ``user_turn`` (a committed
turn preempting the reply), or ``programmatic`` (session.interrupt(), a tool, teardown)."""
ATTR_PLAYOUT_POSITION = _semconv.LK_PLAYOUT_POSITION
"""Seconds of audio that had actually played when the speech was interrupted."""

# Agent handoff (update_agent span)
ATTR_PREVIOUS_AGENT_LABEL = _semconv.LK_PREVIOUS_AGENT_LABEL

# Fallback adapters (the adapter's request span)
ATTR_FALLBACK_LABEL = _semconv.LK_FALLBACK_LABEL
"""Label of the provider that served the request."""
ATTR_FALLBACK_INDEX = _semconv.LK_FALLBACK_INDEX
ATTR_FALLBACK_RECOVERY = _semconv.LK_FALLBACK_RECOVERY
"""True on a background check that a failed provider is back; its response is discarded."""

# Adaptive Interruption attributes
ATTR_IS_INTERRUPTION = _semconv.LK_IS_INTERRUPTION
ATTR_INTERRUPTION_PROBABILITY = _semconv.LK_INTERRUPTION_PROBABILITY
ATTR_INTERRUPTION_TOTAL_DURATION = _semconv.LK_INTERRUPTION_TOTAL_DURATION
ATTR_INTERRUPTION_PREDICTION_DURATION = _semconv.LK_INTERRUPTION_PREDICTION_DURATION
ATTR_INTERRUPTION_DETECTION_DELAY = _semconv.LK_INTERRUPTION_DETECTION_DELAY

# Event loop blocking (telemetry/loop_monitor.py)
ATTR_BLOCKING_DURATION = _semconv.LK_BLOCKING_DURATION
"""Seconds the event loop was blocked, measured to within one heartbeat interval."""
ATTR_BLOCKING_THRESHOLD = _semconv.LK_BLOCKING_THRESHOLD
ATTR_BLOCKING_SEVERITY = _semconv.LK_BLOCKING_SEVERITY
"""``warning`` or ``error``, by which threshold the block crossed."""
ATTR_BLOCKING_TASK = _semconv.LK_BLOCKING_TASK
"""Name of the asyncio task that was running when the loop thread was sampled."""
ATTR_BLOCKING_STACK = _semconv.LK_BLOCKING_STACK
"""Loop-thread stack sampled while blocked (source locations only, no values)."""
ATTR_BLOCKING_GC_TIME = _semconv.LK_BLOCKING_GC_TIME
"""Seconds of garbage collection that ran on the loop thread during the block."""
ATTR_BLOCKING_CPU_TIME = _semconv.LK_BLOCKING_CPU_TIME
"""CPU seconds consumed by the loop thread during the block; near zero means it was waiting."""
ATTR_BLOCKING_IMPORT = _semconv.LK_BLOCKING_IMPORT
"""The module a lazy import was loading when the loop thread was sampled: import it at warm-up."""
ATTR_BLOCKING_SUPPRESSED = _semconv.LK_BLOCKING_SUPPRESSED
"""Reports dropped by rate limiting since the previous emitted span."""
# summary on agent_session, so a session with stalls can be found without opening the trace
ATTR_BLOCKING_COUNT = _semconv.LK_BLOCKING_COUNT
ATTR_BLOCKING_TOTAL_DURATION = _semconv.LK_BLOCKING_TOTAL_DURATION
ATTR_BLOCKING_MAX_DURATION = _semconv.LK_BLOCKING_MAX_DURATION


# --- Registry re-exports: span, span-event and enum-value names the producers emit -----------
# Span names, one per registered span (tests/trace_schema.py SPAN_PARENTS + the Rust Node binding).
SPAN_AGENT_SESSION = _semconv.SPAN_AGENT_SESSION
SPAN_AGENT_SPEAKING = _semconv.SPAN_AGENT_SPEAKING
SPAN_AGENT_TURN = _semconv.SPAN_AGENT_TURN
SPAN_AMD = _semconv.SPAN_AMD
SPAN_DRAIN_AGENT_ACTIVITY = _semconv.SPAN_DRAIN_AGENT_ACTIVITY
SPAN_EOU_DETECTION = _semconv.SPAN_EOU_DETECTION
SPAN_EOU_WAIT = _semconv.SPAN_EOU_WAIT
SPAN_EVENT_LOOP_BLOCKED = _semconv.SPAN_EVENT_LOOP_BLOCKED
SPAN_FUNCTION_TOOL = _semconv.SPAN_FUNCTION_TOOL
SPAN_JOB_ENTRYPOINT = _semconv.SPAN_JOB_ENTRYPOINT
SPAN_JOB_SHUTDOWN = _semconv.SPAN_JOB_SHUTDOWN
SPAN_JUDGE_EVALUATION = _semconv.SPAN_JUDGE_EVALUATION
SPAN_KEYTERM_DETECTION = _semconv.SPAN_KEYTERM_DETECTION
SPAN_LK_CONNECT = _semconv.SPAN_LK_CONNECT
SPAN_LK_PUBLISH = _semconv.SPAN_LK_PUBLISH
SPAN_LK_RECONNECT = _semconv.SPAN_LK_RECONNECT
SPAN_LK_SUBSCRIBE = _semconv.SPAN_LK_SUBSCRIBE
SPAN_LLM_FALLBACK_ADAPTER = _semconv.SPAN_LLM_FALLBACK_ADAPTER
SPAN_LLM_NODE = _semconv.SPAN_LLM_NODE
SPAN_LLM_REQUEST = _semconv.SPAN_LLM_REQUEST
SPAN_LLM_REQUEST_RUN = _semconv.SPAN_LLM_REQUEST_RUN
SPAN_METRICS = _semconv.SPAN_METRICS
SPAN_ON_ENTER = _semconv.SPAN_ON_ENTER
SPAN_ON_EXIT = _semconv.SPAN_ON_EXIT
SPAN_ON_SESSION_END = _semconv.SPAN_ON_SESSION_END
SPAN_ON_USER_TURN_COMPLETED = _semconv.SPAN_ON_USER_TURN_COMPLETED
SPAN_PAUSE_AGENT_ACTIVITY = _semconv.SPAN_PAUSE_AGENT_ACTIVITY
SPAN_PUBLISH_AUDIO_OUTPUT = _semconv.SPAN_PUBLISH_AUDIO_OUTPUT
SPAN_REALTIME_INFERENCE = _semconv.SPAN_REALTIME_INFERENCE
SPAN_REALTIME_METRICS = _semconv.SPAN_REALTIME_METRICS
SPAN_RESUME_AGENT_ACTIVITY = _semconv.SPAN_RESUME_AGENT_ACTIVITY
SPAN_ROOM_CONNECT = _semconv.SPAN_ROOM_CONNECT
SPAN_ROOM_DISCONNECT = _semconv.SPAN_ROOM_DISCONNECT
SPAN_RPC_CALL = _semconv.SPAN_RPC_CALL
SPAN_RPC_HANDLER = _semconv.SPAN_RPC_HANDLER
SPAN_SESSION_CLOSE = _semconv.SPAN_SESSION_CLOSE
SPAN_SESSION_END_UPLOAD = _semconv.SPAN_SESSION_END_UPLOAD
SPAN_SESSION_START = _semconv.SPAN_SESSION_START
SPAN_SETUP_TOOLSETS = _semconv.SPAN_SETUP_TOOLSETS
SPAN_SHUTDOWN_CALLBACK = _semconv.SPAN_SHUTDOWN_CALLBACK
SPAN_START_AGENT_ACTIVITY = _semconv.SPAN_START_AGENT_ACTIVITY
SPAN_TTS_FALLBACK_ADAPTER = _semconv.SPAN_TTS_FALLBACK_ADAPTER
SPAN_TTS_NODE = _semconv.SPAN_TTS_NODE
SPAN_TTS_REQUEST = _semconv.SPAN_TTS_REQUEST
SPAN_TTS_REQUEST_RUN = _semconv.SPAN_TTS_REQUEST_RUN
SPAN_TTS_STREAM = _semconv.SPAN_TTS_STREAM
SPAN_TTS_STREAM_ADAPTER = _semconv.SPAN_TTS_STREAM_ADAPTER
SPAN_TTS_SYNTHESIZE = _semconv.SPAN_TTS_SYNTHESIZE
SPAN_UPDATE_AGENT = _semconv.SPAN_UPDATE_AGENT
SPAN_USER_SPEAKING = _semconv.SPAN_USER_SPEAKING
SPAN_USER_TURN = _semconv.SPAN_USER_TURN
SPAN_WAIT_FOR_AUDIO_TRACK = _semconv.SPAN_WAIT_FOR_AUDIO_TRACK
SPAN_WAIT_FOR_PARTICIPANT = _semconv.SPAN_WAIT_FOR_PARTICIPANT
SPAN_WAIT_FOR_VIDEO_TRACK = _semconv.SPAN_WAIT_FOR_VIDEO_TRACK

# Span-event (checkpoint) names.
CHECKPOINT_AGENT_STATE_CHANGED = _semconv.CHECKPOINT_AGENT_STATE_CHANGED
CHECKPOINT_ANSWER_SENT = _semconv.CHECKPOINT_ANSWER_SENT
CHECKPOINT_EARLY_PC_CREATED = _semconv.CHECKPOINT_EARLY_PC_CREATED
CHECKPOINT_ENGINE = _semconv.CHECKPOINT_ENGINE
CHECKPOINT_ENTRYPOINT_CANCELLED = _semconv.CHECKPOINT_ENTRYPOINT_CANCELLED
CHECKPOINT_ENTRYPOINT_RETURNED = _semconv.CHECKPOINT_ENTRYPOINT_RETURNED
CHECKPOINT_ENTRYPOINT_STARTED = _semconv.CHECKPOINT_ENTRYPOINT_STARTED
CHECKPOINT_EVENT_LOOP_BLOCKED = _semconv.CHECKPOINT_EVENT_LOOP_BLOCKED
CHECKPOINT_EXCEPTION = _semconv.CHECKPOINT_EXCEPTION
CHECKPOINT_FIRST_FRAME = _semconv.CHECKPOINT_FIRST_FRAME
CHECKPOINT_FIRST_MEDIA = _semconv.CHECKPOINT_FIRST_MEDIA
CHECKPOINT_GENERATION = _semconv.CHECKPOINT_GENERATION
CHECKPOINT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS = (
    _semconv.CHECKPOINT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS
)
CHECKPOINT_JOB_ACCEPTED = _semconv.CHECKPOINT_JOB_ACCEPTED
CHECKPOINT_JOB_ASSIGNED = _semconv.CHECKPOINT_JOB_ASSIGNED
CHECKPOINT_JOB_RECEIVED = _semconv.CHECKPOINT_JOB_RECEIVED
CHECKPOINT_JOB_STARTED_ON_SERVER = _semconv.CHECKPOINT_JOB_STARTED_ON_SERVER
CHECKPOINT_JOIN_RECV = _semconv.CHECKPOINT_JOIN_RECV
CHECKPOINT_OFFER_SENT = _semconv.CHECKPOINT_OFFER_SENT
CHECKPOINT_PARTICIPANT_LINKED = _semconv.CHECKPOINT_PARTICIPANT_LINKED
CHECKPOINT_PC_CONNECTED = _semconv.CHECKPOINT_PC_CONNECTED
CHECKPOINT_PC_CREATED = _semconv.CHECKPOINT_PC_CREATED
CHECKPOINT_PREEMPTIVE_GENERATION_DISCARDED = _semconv.CHECKPOINT_PREEMPTIVE_GENERATION_DISCARDED
CHECKPOINT_PRE_CONNECT_AUDIO = _semconv.CHECKPOINT_PRE_CONNECT_AUDIO
CHECKPOINT_PROCESS_ASSIGNED = _semconv.CHECKPOINT_PROCESS_ASSIGNED
CHECKPOINT_REARMED = _semconv.CHECKPOINT_REARMED
CHECKPOINT_ROOM_CONNECTED = _semconv.CHECKPOINT_ROOM_CONNECTED
CHECKPOINT_SIGNAL = _semconv.CHECKPOINT_SIGNAL
CHECKPOINT_STOP_RESPONSE = _semconv.CHECKPOINT_STOP_RESPONSE
CHECKPOINT_SUBSCRIBED = _semconv.CHECKPOINT_SUBSCRIBED
CHECKPOINT_SUPERSEDED = _semconv.CHECKPOINT_SUPERSEDED
CHECKPOINT_TRACK_SUBSCRIBED = _semconv.CHECKPOINT_TRACK_SUBSCRIBED
CHECKPOINT_USER_RESUMED = _semconv.CHECKPOINT_USER_RESUMED
CHECKPOINT_USER_STATE_CHANGED = _semconv.CHECKPOINT_USER_STATE_CHANGED
CHECKPOINT_WS_OPEN = _semconv.CHECKPOINT_WS_OPEN

# `lk.eou.outcome` values.
EOU_OUTCOME_COMMITTED = _semconv.LK_EOU_OUTCOME_COMMITTED
EOU_OUTCOME_DROPPED = _semconv.LK_EOU_OUTCOME_DROPPED
EOU_OUTCOME_USER_RESUMED = _semconv.LK_EOU_OUTCOME_USER_RESUMED

# PII keys the log records carry (`extra={...}`).
ATTR_INTERIM_TRANSCRIPT = _semconv.LK_PII_INTERIM_TRANSCRIPT
ATTR_PII_EVENT = _semconv.LK_PII_EVENT
ATTR_PII_TRANSCRIPT = _semconv.LK_PII_TRANSCRIPT
ATTR_PII_TEXT = _semconv.LK_PII_TEXT
ATTR_PII_ARGUMENTS = _semconv.LK_PII_ARGUMENTS
ATTR_PII_OUTPUT = _semconv.LK_PII_OUTPUT
ATTR_PII_ERROR = _semconv.LK_PII_ERROR
ATTR_PII_RAW_ARGUMENTS = _semconv.LK_PII_RAW_ARGUMENTS
ATTR_PII_REPAIRED = _semconv.LK_PII_REPAIRED
ATTR_PII_PROGRESS_MESSAGE = _semconv.LK_PII_PROGRESS_MESSAGE
ATTR_PII_DTMF_INPUTS = _semconv.LK_PII_DTMF_INPUTS
ATTR_PII_KEYTERMS = _semconv.LK_PII_KEYTERMS
