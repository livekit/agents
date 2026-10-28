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

"""LiveKit Agents for Python

See [https://docs.livekit.io/agents/](https://docs.livekit.io/agents/) for quickstarts,
documentation, and examples.
"""

import typing

from . import cli, inference, ipc, llm, metrics, stt, tokenize, tts, utils, voice
from ._core import Tagger
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AssignmentTimeoutError,
    StreamStateError,
    UnexpectedModelBehavior,
    create_api_error_from_http,
)
from ._generated.language import LanguageCode
from ._generated.vad import VAD, VADCapabilities, VADEvent, VADEventType, VADStream
from .job import (
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    JobProcess,
    JobRequest,
    get_job_context,
)
from .llm.chat_context import (
    AgentConfigUpdate,
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
    Completed,
    Failed,
    FunctionTool,
    InputRequired,
    ModelRetry,
    ProviderTool,
    Rejected,
    StopResponse,
    ToolContext,
    ToolError,
    ToolResult,
    Working,
    function_tool,
)
from .simulation import (
    Scenario,
    ScenarioGroup,
    ScenarioUserdata,
    SimulationContext,
    SimulationDispatch,
    SimulationRun,
    SimulationVerdict,
)
from .storage.context_store import load_context, persisted, save_context
from .text_session import TextSession, TextSessionRoutingError, TextSessionState, TextSessionTurn
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    FlushSentinel,
    NotGiven,
    NotGivenOr,
)
from .version import __version__
from .voice import (
    Agent,
    AgentEvent,
    AgentFalseInterruptionEvent,
    AgentStateChangedEvent,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    ExpressiveOptions,
    FunctionToolsExecutedEvent,
    ModelSettings,
    NonverbalOptions,
    RecordingOptions,
    RunContext,
    RunOutputOptions,
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    SpeechSteeringOptions,
    TextTransforms,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    VoiceSession,
    VoiceTask,
    io,
    room_io,
)
from .voice.agent_session import KeytermDetectionOptions, STTContextOptions
from .voice.amd import AMD, AMDCategory, AMDPredictionEvent
from .voice.background_audio import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip, PlayHandle
from .voice.run_result import (
    AgentHandoffEvent,
    ChatMessageEvent,
    EventAssert,
    EventRangeAssert,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunAssert,
    RunCompleted,
    RunEvent,
    RunInputRequired,
    RunResult,
    RunUsage,
    mock_tools,
)
from .voice.run_stream import (
    ProgressEvent,
    RunStream,
    StreamEvent,
    TextDeltaEvent,
    TextEndEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .voice.turn import (
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
    TurnHandlingOptions,
    UserTurnLimitOptions,
)
from .worker import (
    AgentHandler,
    AgentServer,
    Plugin,
    RtcDoor,
    ServerType,
    WorkerPermissions,
    rtc_session,
)

if typing.TYPE_CHECKING:
    from .a2a.endpoint import a2a_session  # noqa: F401
    from .llm import mcp  # noqa: F401


def __getattr__(name: str) -> typing.Any:
    if name == "mcp":
        from .llm import mcp

        return mcp

    if name == "a2a_session":
        # the text door, loaded on first use like the rest of the A2A stack
        from .a2a.endpoint import a2a_session

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
    "__version__",
    "AgentHandler",
    "AgentServer",
    "RtcDoor",
    "rtc_session",
    "a2a_session",
    "ServerType",
    "WorkerPermissions",
    "JobProcess",
    "JobContext",
    "JobRequest",
    "get_job_context",
    "JobExecutorType",
    "AutoSubscribe",
    "FunctionTool",
    "function_tool",
    "ProviderTool",
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
    "AgentFalseInterruptionEvent",
    "UserInputTranscribedEvent",
    "UserStateChangedEvent",
    "UserTranscriptionTimeoutEvent",
    "SpeechCreatedEvent",
    "ToolExecutionUpdatedEvent",
    "ToolCallStarted",
    "ToolCallUpdated",
    "ToolCallEnded",
    "ToolReplyUpdated",
    "SessionUsageUpdatedEvent",
    "FunctionToolsExecutedEvent",
    "FunctionCall",
    "FunctionCallOutput",
    "AgentConfigUpdate",
    "AgentHandoff",
    "StopResponse",
    "ToolContext",
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
    "Scenario",
    "ScenarioGroup",
    "ScenarioUserdata",
    "SimulationContext",
    "SimulationDispatch",
    "SimulationRun",
    "SimulationVerdict",
    "VoiceSession",
    "VoiceTask",
    "ExpressiveOptions",
    "NonverbalOptions",
    "SpeechSteeringOptions",
    "RecordingOptions",
    "RunOutputOptions",
    "TextTransforms",
    "AgentEvent",
    "ModelSettings",
    "Agent",
    "AssignmentTimeoutError",
    "UnexpectedModelBehavior",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "StreamStateError",
    "create_api_error_from_http",
    "APIConnectOptions",
    "NotGiven",
    "NOT_GIVEN",
    "NotGivenOr",
    "DEFAULT_API_CONNECT_OPTIONS",
    "BackgroundAudioPlayer",
    "BuiltinAudioClip",
    "AudioConfig",
    "PlayHandle",
    "FlushSentinel",
    "LanguageCode",
    "Tagger",
    "io",
    "cli",
    "ipc",
    "llm",
    "metrics",
    "stt",
    "inference",
    "tokenize",
    "tts",
    "utils",
    "voice",
    "VAD",
    "VADCapabilities",
    "VADEvent",
    "VADEventType",
    "VADStream",
    # run_result
    "mock_tools",
    "EventAssert",
    "EventRangeAssert",
    "RunAssert",
    "RunResult",
    "RunCompleted",
    "RunInputRequired",
    "RunUsage",
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
    "AgentHandoffEvent",
    "TurnHandlingOptions",
    "EndpointingOptions",
    "InterruptionOptions",
    "PreemptiveGenerationOptions",
    "UserTurnLimitOptions",
    "KeytermDetectionOptions",
    "STTContextOptions",
    "UserTurnExceededEvent",
    "AMD",
    "AMDCategory",
    "AMDPredictionEvent",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
