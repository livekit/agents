# Import official OpenTelemetry semantic conventions
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.attributes import error_attributes

ATTR_SPEECH_ID = "lk.speech_id"
ATTR_AGENT_LABEL = "lk.agent_label"
ATTR_START_TIME = "lk.start_time"
ATTR_END_TIME = "lk.end_time"

# session start
ATTR_JOB_ID = "lk.job_id"
ATTR_AGENT_NAME = "lk.agent_name"
ATTR_ROOM_NAME = "lk.room_name"
ATTR_SESSION_OPTIONS = "lk.session_options"

# assistant turn
ATTR_USER_INPUT = "lk.user_input"
ATTR_INSTRUCTIONS = "lk.instructions"
ATTR_SPEECH_INTERRUPTED = "lk.interrupted"

# llm node
ATTR_CHAT_CTX = "lk.chat_ctx"
ATTR_FUNCTION_TOOLS = "lk.function_tools"
ATTR_RESPONSE_TEXT = "lk.response.text"
ATTR_RESPONSE_FUNCTION_CALLS = "lk.response.function_calls"

# function tool
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
ATTR_END_OF_UTTERANCE_DELAY = "lk.end_of_utterance_delay"

# metrics
ATTR_LLM_METRICS = "lk.llm_metrics"
ATTR_TTS_METRICS = "lk.tts_metrics"

# OpenTelemetry GenAI attributes - use only for direct LLM model interactions
ATTR_GEN_AI_REQUEST_MODEL = gen_ai_attributes.GEN_AI_REQUEST_MODEL
ATTR_GEN_AI_SYSTEM = gen_ai_attributes.GEN_AI_SYSTEM
ATTR_GEN_AI_OPERATION_NAME = gen_ai_attributes.GEN_AI_OPERATION_NAME
ATTR_GEN_AI_RESPONSE_ID = gen_ai_attributes.GEN_AI_RESPONSE_ID
ATTR_GEN_AI_RESPONSE_MODEL = gen_ai_attributes.GEN_AI_RESPONSE_MODEL
ATTR_GEN_AI_USAGE_INPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS
ATTR_GEN_AI_USAGE_OUTPUT_TOKENS = gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS

# OpenLLMetry standard attributes
ATTR_LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
ATTR_LLM_IS_STREAMING = "llm.is_streaming"

# OpenLLMetry standard attributes - function/tool requests
ATTR_LLM_REQUEST_FUNCTIONS_NAME = "llm.request.functions.{}.name"
ATTR_LLM_REQUEST_FUNCTIONS_DESCRIPTION = "llm.request.functions.{}.description"
ATTR_LLM_REQUEST_FUNCTIONS_PARAMETERS = "llm.request.functions.{}.parameters"

# OpenTelemetry GenAI event names (for structured logging)
EVENT_GEN_AI_SYSTEM_MESSAGE = "gen_ai.system.message"
EVENT_GEN_AI_USER_MESSAGE = "gen_ai.user.message"
EVENT_GEN_AI_ASSISTANT_MESSAGE = "gen_ai.assistant.message"
EVENT_GEN_AI_TOOL_MESSAGE = "gen_ai.tool.message"
EVENT_GEN_AI_CHOICE = "gen_ai.choice"

# Platform-specific attributes
ATTR_LANGFUSE_COMPLETION_START_TIME = "langfuse.observation.completion_start_time"
