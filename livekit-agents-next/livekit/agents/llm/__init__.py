"""Model contracts: the generic LLM, chat context, tools, realtime and duplex models, and their fallback adapters."""

from __future__ import annotations

import asyncio

from .._generated import facades_llm as _facades, public_llm as _public
from .._generated.public_llm import *  # noqa: F403
from . import utils
from .chat_context import (
    AgentConfigUpdate,
    AgentHandoff,
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
from .duplex import (
    AudioGate,
    DuplexAudioFrame,
    DuplexCapabilities,
    DuplexInputGap,
    DuplexInputMute,
    DuplexModel,
    DuplexOutputTranscriptDelta,
    DuplexRealtimeAdapter,
    DuplexSession,
    FixedGate,
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
    RealtimeModelFallbackAdapter,
    RealtimeSession,
    RealtimeSessionReconnectedEvent,
    RemoteItemAddedEvent,
)
from .tool_context import (
    AsyncToolset,
    Completed,
    DuplicateScope,
    Failed,
    FunctionTool,
    InputRequired,
    ModelRetry,
    ProviderTool,
    RawFunctionTool,
    Rejected,
    StopResponse,
    Tool,
    ToolChoice,
    ToolContext,
    ToolError,
    ToolFlag,
    ToolResult,
    Toolset,
    Working,
    function_tool,
    is_function_tool,
    is_raw_function_tool,
)

__all__ = [
    "ChatContext",
    "ChatRole",
    "ChatMessage",
    "ChatContent",
    "FunctionCall",
    "FunctionCallOutput",
    "AudioContent",
    "ImageContent",
    "AgentConfigUpdate",
    "AgentHandoff",
    "ChatItem",
    "FallbackAdapter",
    "RealtimeModelFallbackAdapter",
    "ToolChoice",
    "Tool",
    "Toolset",
    "AsyncToolset",
    "is_function_tool",
    "function_tool",
    "FunctionTool",
    "is_raw_function_tool",
    "RawFunctionTool",
    "ProviderTool",
    "ToolContext",
    "ToolError",
    "ToolResult",
    "ToolFlag",
    "DuplicateScope",
    "StopResponse",
    "ModelRetry",
    "Completed",
    "Failed",
    "Rejected",
    "InputRequired",
    "Working",
    "utils",
    "DuplexModel",
    "DuplexSession",
    "DuplexCapabilities",
    "DuplexAudioFrame",
    "DuplexInputGap",
    "DuplexInputMute",
    "DuplexOutputTranscriptDelta",
    "DuplexRealtimeAdapter",
    "AudioGate",
    "FixedGate",
    "RealtimeModel",
    "RealtimeError",
    "RealtimeCapabilities",
    "RealtimeSession",
    "InputTranscriptionCompleted",
    "InputSpeechStartedEvent",
    "InputSpeechStoppedEvent",
    "GenerationCreatedEvent",
    "MessageGeneration",
    "RealtimeSessionReconnectedEvent",
    "RemoteItemAddedEvent",
]
__all__ += _public.__all__


class FallbackAdapter(_facades.FallbackAdapter):
    """Several LLMs behind one: a failed one is skipped until it recovers. The chain is the
    core's (``_core.FallbackLlm``)."""

    def prewarm(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._members[0].prewarm(loop=loop)
