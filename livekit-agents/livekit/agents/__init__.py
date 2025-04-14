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

from . import cli, ipc, llm, metrics, stt, tokenize, tts, utils, vad  # noqa: F401
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AssignmentTimeoutError,
)
from .job import AutoSubscribe, JobContext, JobExecutorType, JobProcess, JobRequest, get_job_context
from .llm.chat_context import (
    ChatContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
)
from .llm.tool_context import FunctionTool, StopResponse, ToolError, function_tool
from .plugin import Plugin
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from .version import __version__
from .voice import (
    Agent,
    AgentEvent,
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    MetricsCollectedEvent,
    ModelSettings,
    RunContext,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    io,
)
from .voice.background_audio import AudioConfig, BackgroundAudioPlayer, BuiltinAudioClip
from .voice.room_io import RoomInputOptions, RoomIO, RoomOutputOptions
from .worker import SimulateJobInfo, Worker, WorkerOptions, WorkerPermissions, WorkerType

__all__ = [
    "__version__",
    "Worker",
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
    "ChatContext",
    "ChatItem",
    "RoomIO",
    "RoomInputOptions",
    "RoomOutputOptions",
    "ChatMessage",
    "ChatRole",
    "ChatContent",
    "ErrorEvent",
    "CloseEvent",
    "ConversationItemAddedEvent",
    "AgentStateChangedEvent",
    "UserInputTranscribedEvent",
    "UserStateChangedEvent",
    "SpeechCreatedEvent",
    "MetricsCollectedEvent",
    "io",
    "FunctionCall",
    "FunctionCallOutput",
    "StopResponse",
    "ToolError",
    "RunContext",
    "Plugin",
    "AgentSession",
    "AgentEvent",
    "ModelSettings",
    "Agent",
    "AssignmentTimeoutError",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "APIConnectOptions",
    "NotGiven",
    "NOT_GIVEN",
    "NotGivenOr",
    "DEFAULT_API_CONNECT_OPTIONS",
    "BackgroundAudioPlayer",
    "BuiltinAudioClip",
    "AudioConfig",
    "SimulateJobInfo",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
