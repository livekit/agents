ATTR_SPEECH_ID = "lk.speech_id"
ATTR_AGENT_LABEL = "lk.agent_label"
ATTR_START_TIME = "lk.start_time"
ATTR_END_TIME = "lk.end_time"
ATTR_RETRY_COUNT = "lk.retry_count"


ATTR_PARTICIPANT_ID = "lk.participant_id"
ATTR_PARTICIPANT_IDENTITY = "lk.participant_identity"
ATTR_PARTICIPANT_KIND = "lk.participant_kind"

# session start
ATTR_JOB_ID = "lk.job_id"
ATTR_AGENT_NAME = "lk.agent_name"
ATTR_ROOM_NAME = "lk.room_name"
ATTR_SESSION_OPTIONS = "lk.session_options"

# agent turn
ATTR_AGENT_TURN_ID = "lk.generation_id"
ATTR_AGENT_PARENT_TURN_ID = "lk.parent_generation_id"
ATTR_USER_INPUT = "lk.user_input"
ATTR_INSTRUCTIONS = "lk.instructions"
ATTR_SPEECH_INTERRUPTED = "lk.interrupted"

# llm node
ATTR_CHAT_CTX = "lk.chat_ctx"
ATTR_FUNCTION_TOOLS = "lk.function_tools"
ATTR_RESPONSE_TEXT = "lk.response.text"
ATTR_RESPONSE_FUNCTION_CALLS = "lk.response.function_calls"

# function tool
ATTR_FUNCTION_TOOL_ID = "lk.function_tool.id"
ATTR_FUNCTION_TOOL_NAME = "lk.function_tool.name"
ATTR_FUNCTION_TOOL_ARGS = "lk.function_tool.arguments"
ATTR_FUNCTION_TOOL_IS_ERROR = "lk.function_tool.is_error"
ATTR_FUNCTION_TOOL_OUTPUT = "lk.function_tool.output"

# tts node
ATTR_TTS_INPUT_TEXT = "lk.input_text"
ATTR_TTS_STREAMING = "lk.tts.streaming"
ATTR_TTS_LABEL = "lk.tts.label"

# eou detection
ATTR_EOU_PROBABILITY = "lk.eou.probability"
ATTR_EOU_UNLIKELY_THRESHOLD = "lk.eou.unlikely_threshold"
ATTR_EOU_DELAY = "lk.eou.endpointing_delay"
ATTR_EOU_LANGUAGE = "lk.eou.language"
ATTR_USER_TRANSCRIPT = "lk.user_transcript"
ATTR_TRANSCRIPT_CONFIDENCE = "lk.transcript_confidence"
ATTR_TRANSCRIPTION_DELAY = "lk.transcription_delay"
ATTR_END_OF_TURN_DELAY = "lk.end_of_turn_delay"

# metrics
ATTR_LLM_METRICS = "lk.llm_metrics"
ATTR_TTS_METRICS = "lk.tts_metrics"
ATTR_REALTIME_MODEL_METRICS = "lk.realtime_model_metrics"

# OpenTelemetry GenAI attributes
# OpenTelemetry specification: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

# Unofficial OpenTelemetry GenAI attributes, these are namespaces recognised by LangFuse
# https://langfuse.com/integrations/native/opentelemetry#usage
# but not yet in the official OpenTelemetry specification.
ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS = "gen_ai.usage.input_text_tokens"
ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS = "gen_ai.usage.input_audio_tokens"
ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS = "gen_ai.usage.input_cached_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS = "gen_ai.usage.output_text_tokens"
ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS = "gen_ai.usage.output_audio_tokens"

# OpenTelemetry GenAI event names (for structured logging)
EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"

# Exception attributes
ATTR_EXCEPTION_TRACE = "exception.stacktrace"
ATTR_EXCEPTION_TYPE = "exception.type"
ATTR_EXCEPTION_MESSAGE = "exception.message"

# Platform-specific attributes
ATTR_LANGFUSE_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"
