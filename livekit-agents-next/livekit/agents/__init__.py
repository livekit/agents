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

from . import cli, inference, ipc, llm, metrics, stt, tokenize, tts, utils, vad, voice
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AssignmentTimeoutError,
    UnexpectedModelBehavior,
    create_api_error_from_http,
)
from .job import (
    AutoSubscribe,
    JobContext,
    JobExecutorType,
    JobProcess,
    JobRequest,
    get_job_context,
)
from .language import LanguageCode
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
    Transition,
    Working,
    function_tool,
)
from .plugin import Plugin
from .simulation import (
    Scenario,
    ScenarioGroup,
    ScenarioUserdata,
    SimulationContext,
    SimulationDispatch,
    SimulationRun,
    SimulationVerdict,
)
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
    AgentTask,
    AudioRecognition,
    CloseEvent,
    CloseReason,
    ConversationItemAddedEvent,
    ErrorEvent,
    ExpressiveOptions,
    FunctionToolsExecutedEvent,
    LiveSession,
    MetricsCollectedEvent,
    ModelSettings,
    NonverbalOptions,
    RecordingOptions,
    RunContext,
    RunOutputOptions,
    SessionUsageUpdatedEvent,
    SpeechCreatedEvent,
    SpeechSteeringOptions,
    ToolCallEnded,
    ToolCallStarted,
    ToolCallUpdated,
    ToolExecutionUpdatedEvent,
    ToolReplyUpdated,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    UserTranscriptionTimeoutEvent,
    UserTurnExceededEvent,
    avatar,
    io,
    room_io,
    text_transforms,
)
from .voice.amd import (
    AMD,
    AMDCategory,
    AMDPredictionEvent,
)
from .voice.background_audio import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip, PlayHandle
from .voice.keyterm_detection import (
    KeytermDetectionOptions,
    KeytermsOptions,
    STTContextOptions,
)
from .voice.room_io import RoomInputOptions, RoomIO, RoomOutputOptions
from .voice.run_result import (
    AgentHandoffEvent,
    ChatMessageEvent,
    EventAssert,
    EventRangeAssert,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    RunAssert,
    RunEvent,
    RunResult,
    RunUsage,
    mock_tools,
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
    RtcDoor,
    WorkerOptions,
    WorkerPermissions,
    WorkerType,
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

    if name == "AgentSession":
        # deprecated alias, kept one release
        from .voice import agent_session

        return agent_session.AgentSession

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "AgentHandler",
    "AgentServer",
    "RtcDoor",
    "rtc_session",
    "a2a_session",
    "WorkerOptions",
    "WorkerType",
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
    "RoomIO",
    "RoomInputOptions",
    "RoomOutputOptions",
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
    "MetricsCollectedEvent",
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
    "Transition",
    "RunContext",
    "Plugin",
    "Scenario",
    "ScenarioGroup",
    "ScenarioUserdata",
    "SimulationContext",
    "SimulationDispatch",
    "SimulationRun",
    "SimulationVerdict",
    "LiveSession",
    "AgentSession",
    "AudioRecognition",
    "ExpressiveOptions",
    "NonverbalOptions",
    "SpeechSteeringOptions",
    "RecordingOptions",
    "RunOutputOptions",
    "text_transforms",
    "AgentEvent",
    "ModelSettings",
    "Agent",
    "AgentTask",
    "AssignmentTimeoutError",
    "UnexpectedModelBehavior",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
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
    "io",
    "avatar",
    "cli",
    "ipc",
    "llm",
    "metrics",
    "stt",
    "inference",
    "tokenize",
    "tts",
    "utils",
    "vad",
    "voice",
    # run_result
    "mock_tools",
    "EventAssert",
    "EventRangeAssert",
    "RunAssert",
    "RunResult",
    "RunUsage",
    "RunEvent",
    "ChatMessageEvent",
    "FunctionCallEvent",
    "FunctionCallOutputEvent",
    "AgentHandoffEvent",
    "AMD",
    "AMDCategory",
    "AMDPredictionEvent",
    "TurnHandlingOptions",
    "EndpointingOptions",
    "InterruptionOptions",
    "PreemptiveGenerationOptions",
    "UserTurnLimitOptions",
    "KeytermsOptions",
    "KeytermDetectionOptions",
    "STTContextOptions",
    "UserTurnExceededEvent",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
