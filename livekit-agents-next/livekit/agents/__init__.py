# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing

from . import cli, http, inference, llm, metrics, stt, tokenize, tts, utils, voice
from ._bindings import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    StreamStateError,
    UnexpectedModelBehavior,
    create_api_error_from_http,
    redacted_cause,
)
from ._generated import public as _public
from ._generated.public import *  # noqa: F403
from .evals import SimulationContext, SimulationVerdict
from .job import (
    JobContext,
    get_job_context,
)
from .llm.chat_context import (
    AgentHandoff,
    ChatContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
)
from .llm.tool_context import (
    AsyncToolOptions,
    AsyncToolset,
    Completed,
    Failed,
    InputRequired,
    ModelRetry,
    Rejected,
    StopResponse,
    ToolError,
    ToolHandlingOptions,
    ToolResult,
    Toolset,
    UpdatePromptArgs,
    Working,
    function_tool,
)
from .plugin import Plugin
from .storage.context import load_context, persisted, save_context
from .text_session import TextSession, TextSessionRoutingError, TextSessionState, TextSessionTurn
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from .version import __version__ as __version__
from .voice import (
    AMD,
    Agent,
    AgentRetries,
    AgentStateChangedEvent,
    AMDCategory,
    AMDPredictionEvent,
    AudioConfig,
    BackgroundAudioPlayer,
    BuiltinAudioClip,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    DetectionOptions,
    ErrorEvent,
    FunctionToolsExecutedEvent,
    ModelSettings,
    PlayHandle,
    RunContext,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    VoiceSession,
    VoiceTask,
    VoiceTaskFailed,
    io,
    room_io,
)
from .voice.io import EndpointingOptions, InterruptionOptions, TurnHandlingOptions
from .voice.run_result import (
    ChatMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ProgressEvent,
    RunCompleted,
    RunEvent,
    RunInputRequired,
    RunNoOutput,
    RunResult,
    RunStream,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    mock_tools,
)
from .worker import (
    AgentHandler,
    AgentServer,
    rtc_session,
)

if typing.TYPE_CHECKING:
    from .a2a import a2a_session  # noqa: F401
    from .llm import mcp  # noqa: F401


def __getattr__(name: str) -> object:
    if name == "mcp":
        from .llm import mcp

        return mcp

    if name == "a2a_session":
        from .a2a import a2a_session

        return a2a_session

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TextSession",
    "TextSessionState",
    "TextSessionTurn",
    "TextSessionRoutingError",
    "persisted",
    "load_context",
    "save_context",
    "AgentHandler",
    "AgentServer",
    "rtc_session",
    "a2a_session",
    "JobContext",
    "get_job_context",
    "function_tool",
    "Toolset",
    "AsyncToolset",
    "AsyncToolOptions",
    "ToolHandlingOptions",
    "UpdatePromptArgs",
    "ChatContext",
    "ChatItem",
    "room_io",
    "ChatMessage",
    "ChatRole",
    "ChatContent",
    "CloseReason",
    "ErrorEvent",
    "CloseEvent",
    "ConversationItemAddedEvent",
    "AgentStateChangedEvent",
    "UserInputTranscribedEvent",
    "UserStateChangedEvent",
    "SpeechCreatedEvent",
    "FunctionToolsExecutedEvent",
    "FunctionCall",
    "FunctionCallOutput",
    "AgentHandoff",
    "StopResponse",
    "ToolError",
    "ToolResult",
    "ModelRetry",
    "Completed",
    "Failed",
    "Rejected",
    "InputRequired",
    "Working",
    "RunContext",
    "Plugin",
    "SimulationContext",
    "SimulationVerdict",
    "VoiceSession",
    "VoiceTask",
    "VoiceTaskFailed",
    "AMD",
    "AMDCategory",
    "AMDPredictionEvent",
    "DetectionOptions",
    "BackgroundAudioPlayer",
    "AudioConfig",
    "BuiltinAudioClip",
    "PlayHandle",
    "Agent",
    "AgentRetries",
    "ModelSettings",
    "UnexpectedModelBehavior",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "StreamStateError",
    "create_api_error_from_http",
    "redacted_cause",
    "APIConnectOptions",
    "NotGiven",
    "NOT_GIVEN",
    "NotGivenOr",
    "DEFAULT_API_CONNECT_OPTIONS",
    "io",
    "cli",
    "http",
    "llm",
    "metrics",
    "stt",
    "inference",
    "tokenize",
    "tts",
    "utils",
    "voice",
    "mock_tools",
    "RunResult",
    "RunCompleted",
    "RunInputRequired",
    "RunNoOutput",
    "RunStream",
    "StreamEvent",
    "TextDeltaEvent",
    "TextEndEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "ProgressEvent",
    "RunEvent",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "TurnHandlingOptions",
    "EndpointingOptions",
    "InterruptionOptions",
]
__all__ += _public.__all__
