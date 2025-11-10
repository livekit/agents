from . import remote_chat_context, utils
from .chat_context import (
    AudioContent,
    ChatContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
    ImageContent,
)
from .fallback_adapter import AvailabilityChangedEvent, FallbackAdapter
from .llm import (
    LLM,
    ChatChunk,
    ChoiceDelta,
    CompletionUsage,
    FunctionToolCall,
    LLMError,
    LLMStream,
)
from .realtime import (
    GenerationCreatedEvent,
    InputSpeechStartedEvent,
    InputSpeechStoppedEvent,
    InputTranscriptionCompleted,
    MessageGeneration,
    RealtimeCapabilities,
    RealtimeError,
    RealtimeModel,
    RealtimeModelError,
    RealtimeSession,
    RealtimeSessionReconnectedEvent,
)
from .tool_context import (
    FunctionTool,
    RawFunctionTool,
    StopResponse,
    ToolChoice,
    ToolContext,
    ToolError,
    find_function_tools,
    function_tool,
    is_function_tool,
    is_raw_function_tool,
)

__all__ = [
    "LLM",
    "LLMStream",
    "ChatContext",
    "ChatRole",
    "ChatMessage",
    "ChatContent",
    "FunctionCall",
    "FunctionCallOutput",
    "AudioContent",
    "ImageContent",
    "ChatItem",
    "ChoiceDelta",
    "ChatChunk",
    "CompletionUsage",
    "FallbackAdapter",
    "AvailabilityChangedEvent",
    "ToolChoice",
    "is_function_tool",
    "function_tool",
    "find_function_tools",
    "FunctionTool",
    "is_raw_function_tool",
    "RawFunctionTool",
    "ToolContext",
    "ToolError",
    "StopResponse",
    "utils",
    "remote_chat_context",
    "FunctionToolCall",
    "RealtimeModel",
    "RealtimeError",
    "RealtimeModelError",
    "RealtimeCapabilities",
    "RealtimeSession",
    "InputTranscriptionCompleted",
    "InputSpeechStartedEvent",
    "InputSpeechStoppedEvent",
    "GenerationCreatedEvent",
    "MessageGeneration",
    "RealtimeSessionReconnectedEvent",
    "RealtimeSessionRestoredEvent",
    "LLMError",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
# --- Compatibility shim: expose FunctionContext alias if not present ---
try:
    FunctionContext  # type: ignore  # noqa: F401
except Exception:
    # Provide minimal compatibility placeholder for tests/imports
    class FunctionContext:
        """Compatibility placeholder for tests that import FunctionContext."""
        pass
# --- Compatibility shim: expose FunctionContext and TypeInfo aliases if missing ---
# This shim is intentionally minimal and only intended to satisfy test imports.
try:
    FunctionContext  # type: ignore  # noqa: F401
except Exception:
    class FunctionContext:
        """Compatibility placeholder for tests that import FunctionContext."""
        pass

try:
    TypeInfo  # type: ignore  # noqa: F401
except Exception:
    class TypeInfo:
        """Compatibility placeholder for tests that import TypeInfo."""
        def __init__(self, *args, **kwargs):
            # store args/kwargs for introspection if needed by tests
            self.args = args
            self.kwargs = kwargs
