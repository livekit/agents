"""Span attribute and event name constants for LiveKit Agents telemetry.

Attributes carrying conversational content, tool payloads, or other user data
must include a dot-delimited ``pii`` segment (``lk.pii.<name>``): PII-enabled
projects have these attributes stripped at the LiveKit Cloud collector, and the
segment is the only marker it honors. Attributes must never embed such content
in span names, event names, or log message bodies — those are not redactable.
"""

ATTR_SPEECH_ID = "lk.speech_id"
ATTR_AGENT_LABEL = "lk.agent_label"
ATTR_START_TIME = "lk.start_time"
ATTR_END_TIME = "lk.end_time"
ATTR_RETRY_COUNT = "lk.retry_count"
ATTR_PROVIDER_REQUEST_IDS = "lk.provider_request_ids"
"""Provider-known correlation ids associated with this span (list[str]).

Populated by STT/TTS plugins when the id is either sent to the provider
(e.g. WS context_id) or returned by it (e.g. response request_id / session_id),
so it can be cross-referenced with the provider's logs for debugging."""


ATTR_PARTICIPANT_ID = "lk.participant_id"
ATTR_PARTICIPANT_IDENTITY = "lk.pii.participant_identity"
ATTR_PARTICIPANT_KIND = "lk.participant_kind"

# session start
ATTR_JOB_ID = "lk.job_id"
ATTR_AGENT_NAME = "lk.agent_name"
ATTR_CLOUD_AGENT_ID = "lk.cloud_agent_id"
ATTR_DEPLOYMENT_ID = "lk.deployment_id"
ATTR_ROOM_NAME = "lk.pii.room_name"
ATTR_SESSION_OPTIONS = "lk.session_options"

# agent turn
ATTR_AGENT_TURN_ID = "lk.generation_id"
ATTR_AGENT_PARENT_TURN_ID = "lk.parent_generation_id"
ATTR_USER_INPUT = "lk.pii.user_input"
ATTR_INSTRUCTIONS = "lk.pii.instructions"
ATTR_SPEECH_INTERRUPTED = "lk.interrupted"

# llm node
ATTR_CHAT_CTX = "lk.pii.chat_ctx"
ATTR_FUNCTION_TOOLS = "lk.function_tools"
ATTR_PROVIDER_TOOLS = "lk.provider_tools"
ATTR_TOOL_SETS = "lk.tool_sets"
ATTR_RESPONSE_TEXT = "lk.pii.response.text"
ATTR_RESPONSE_FUNCTION_CALLS = "lk.pii.response.function_calls"
ATTR_RESPONSE_TTFT = "lk.response.ttft"

# function tool
ATTR_FUNCTION_TOOL_ID = "lk.function_tool.id"
ATTR_FUNCTION_TOOL_NAME = "lk.function_tool.name"
ATTR_FUNCTION_TOOL_ARGS = "lk.pii.function_tool.arguments"
ATTR_FUNCTION_TOOL_IS_ERROR = "lk.function_tool.is_error"
ATTR_FUNCTION_TOOL_OUTPUT = "lk.pii.function_tool.output"

# tts node
ATTR_TTS_INPUT_TEXT = "lk.pii.input_text"
ATTR_TTS_STREAMING = "lk.tts.streaming"
ATTR_TTS_LABEL = "lk.tts.label"
ATTR_RESPONSE_TTFB = "lk.response.ttfb"

# eou detection
ATTR_EOU_PROBABILITY = "lk.eou.probability"
ATTR_EOU_UNLIKELY_THRESHOLD = "lk.eou.unlikely_threshold"
ATTR_EOU_DELAY = "lk.eou.endpointing_delay"
ATTR_EOU_LANGUAGE = "lk.eou.language"
ATTR_USER_TRANSCRIPT = "lk.pii.user_transcript"
ATTR_TRANSCRIPT_CONFIDENCE = "lk.transcript_confidence"
ATTR_TRANSCRIPTION_DELAY = "lk.transcription_delay"
ATTR_END_OF_TURN_DELAY = "lk.end_of_turn_delay"
ATTR_EOU_SOURCE = "lk.eou.source"
ATTR_EOU_DETECTION_DELAY = "lk.eou.detection_delay"
ATTR_EOU_FROM_CACHE = "lk.eou.from_cache"

# metrics
ATTR_LLM_METRICS = "lk.llm_metrics"
ATTR_TTS_METRICS = "lk.tts_metrics"
ATTR_REALTIME_MODEL_METRICS = "lk.realtime_model_metrics"

# latency span attributes
ATTR_E2E_LATENCY = "lk.e2e_latency"

# OpenTelemetry GenAI semantic conventions, mirroring the attribute registry of
# https://github.com/open-telemetry/semantic-conventions-genai. Backends ingest these
# directly, so the names must stay byte-for-byte identical to the registry. The ones the
# spec flags as sensitive are listed in `telemetry.pii.GEN_AI_PII_ATTRIBUTES`, since a
# standard name cannot carry the `lk.pii.` marker segment.

ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"

ATTR_GEN_AI_RESPONSE_ID = "gen_ai.response.id"
ATTR_GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
ATTR_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS = "gen_ai.usage.cache_write.input_tokens"
ATTR_GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
ATTR_GEN_AI_USAGE_TEXT_INPUT_TOKENS = "gen_ai.usage.text.input_tokens"
ATTR_GEN_AI_USAGE_TEXT_OUTPUT_TOKENS = "gen_ai.usage.text.output_tokens"
ATTR_GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.text.cache_read.input_tokens"
ATTR_GEN_AI_USAGE_AUDIO_INPUT_TOKENS = "gen_ai.usage.audio.input_tokens"
ATTR_GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS = "gen_ai.usage.audio.output_tokens"
ATTR_GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.audio.cache_read.input_tokens"
ATTR_GEN_AI_USAGE_IMAGE_INPUT_TOKENS = "gen_ai.usage.image.input_tokens"
ATTR_GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.image.cache_read.input_tokens"
ATTR_GEN_AI_TOKEN_TYPE = "gen_ai.token.type"

ATTR_GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

ATTR_GEN_AI_AGENT_NAME = "gen_ai.agent.name"

ATTR_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
ATTR_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"
ATTR_GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
ATTR_GEN_AI_TOOL_CALL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_GEN_AI_TOOL_CALL_RESULT = "gen_ai.tool.call.result"
ATTR_GEN_AI_TOOL_DEFINITIONS = "gen_ai.tool.definitions"

ATTR_GEN_AI_SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
ATTR_GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"
ATTR_GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"
ATTR_GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"

ATTR_GEN_AI_RETRIEVAL_DOCUMENTS = "gen_ai.retrieval.documents"
ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT = "gen_ai.retrieval.query.text"
ATTR_GEN_AI_MEMORY_QUERY_TEXT = "gen_ai.memory.query.text"
ATTR_GEN_AI_MEMORY_RECORDS = "gen_ai.memory.records"
ATTR_GEN_AI_EVALUATION_EXPLANATION = "gen_ai.evaluation.explanation"
ATTR_GEN_AI_PROMPT_VARIABLE = "gen_ai.prompt.variable"  # template: gen_ai.prompt.variable.<key>
ATTR_GEN_AI_WORKFLOW_NAME = "gen_ai.workflow.name"

ATTR_ERROR_TYPE = "error.type"


class GenAIOperationName:
    """Well-known ``gen_ai.operation.name`` values."""

    CHAT = "chat"
    GENERATE_CONTENT = "generate_content"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    RETRIEVAL = "retrieval"
    FETCH_RESPONSE = "fetch_response"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    EXECUTE_TOOL = "execute_tool"
    INVOKE_WORKFLOW = "invoke_workflow"
    PLAN = "plan"
    SEARCH_MEMORY = "search_memory"
    CREATE_MEMORY = "create_memory"
    UPDATE_MEMORY = "update_memory"
    UPSERT_MEMORY = "upsert_memory"
    DELETE_MEMORY = "delete_memory"
    CREATE_MEMORY_STORE = "create_memory_store"
    DELETE_MEMORY_STORE = "delete_memory_store"


class GenAIOutputType:
    """Well-known ``gen_ai.output.type`` values."""

    TEXT = "text"
    JSON = "json"
    IMAGE = "image"
    SPEECH = "speech"


class GenAIFinishReason:
    """Well-known ``gen_ai.response.finish_reasons`` values."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALL = "tool_call"
    COMPACTION = "compaction"
    ERROR = "error"


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
        "mistral_ai",
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
    "api.mistral.ai": "mistral_ai",
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
    "mistral": "mistral_ai",
    "mistralai": "mistral_ai",
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
ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS = "gen_ai.usage.input_text_tokens"
ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS = "gen_ai.usage.input_audio_tokens"
ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS = "gen_ai.usage.input_cached_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS = "gen_ai.usage.output_text_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS = "gen_ai.usage.output_audio_tokens"
ATTR_GEN_AI_USAGE_REASONING_TOKENS = "gen_ai.usage.reasoning_tokens"

# OpenTelemetry GenAI event names (for structured logging)
EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"
EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS = "gen_ai.client.inference.operation.details"

# OpenTelemetry GenAI metric names
METRIC_GEN_AI_CLIENT_TOKEN_USAGE = "gen_ai.client.token.usage"
METRIC_GEN_AI_CLIENT_OPERATION_DURATION = "gen_ai.client.operation.duration"
METRIC_GEN_AI_CLIENT_TIME_TO_FIRST_CHUNK = "gen_ai.client.operation.time_to_first_chunk"
METRIC_GEN_AI_INVOKE_AGENT_DURATION = "gen_ai.invoke_agent.duration"
METRIC_GEN_AI_EXECUTE_TOOL_DURATION = "gen_ai.execute_tool.duration"

# Exception attributes
ATTR_EXCEPTION_TRACE = "exception.stacktrace"
ATTR_EXCEPTION_TYPE = "exception.type"
ATTR_EXCEPTION_MESSAGE = "exception.message"

# Platform-specific attributes
ATTR_LANGFUSE_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"

# AMD (Answering Machine Detection) attributes
ATTR_AMD_CATEGORY = "lk.amd.category"
ATTR_AMD_REASON = "lk.amd.reason"
ATTR_AMD_SPEECH_DURATION = "lk.amd.speech_duration"
ATTR_AMD_DELAY = "lk.amd.delay"
ATTR_AMD_TRANSCRIPT = "lk.pii.amd.transcript"

# Adaptive Interruption attributes
ATTR_IS_INTERRUPTION = "lk.is_interruption"
ATTR_INTERRUPTION_PROBABILITY = "lk.interruption.probability"
ATTR_INTERRUPTION_TOTAL_DURATION = "lk.interruption.total_duration"
ATTR_INTERRUPTION_PREDICTION_DURATION = "lk.interruption.prediction_duration"
ATTR_INTERRUPTION_DETECTION_DELAY = "lk.interruption.detection_delay"
