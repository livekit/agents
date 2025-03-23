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

from . import cli, ipc, llm, metrics, stt, tokenize, tts, utils, vad, voice  # noqa: F401
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AssignmentTimeoutError,
)
from .job import AutoSubscribe, JobContext, JobExecutorType, JobProcess, JobRequest
from .llm.chat_context import (
    ChatContent,
    ChatContext,
    ChatItem,
    ChatMessage,
    ChatRole,
    FunctionCall,
    FunctionCallOutput,
)
from .llm.tool_context import function_tool
from .plugin import Plugin
from .types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    AgentState,
    APIConnectOptions,
    NotGiven,
    NotGivenOr,
)
from .version import __version__
from .voice import Agent, AgentEvent, AgentSession, RunContext, io
from .voice.room_io import RoomInputOptions, RoomIO, RoomOutputOptions
from .voice.background_audio import BackgroundAudio
from .worker import Worker, WorkerOptions, WorkerPermissions, WorkerType

__all__ = [
    "__version__",
    "Worker",
    "WorkerOptions",
    "WorkerType",
    "WorkerPermissions",
    "JobProcess",
    "JobContext",
    "JobRequest",
    "JobExecutorType",
    "AutoSubscribe",
    "AgentState",
    "function_tool",
    "ChatContext",
    "ChatItem",
    "RoomIO",
    "RoomInputOptions",
    "RoomOutputOptions",
    "ChatMessage",
    "ChatRole",
    "ChatContent",
    "io",
    "FunctionCall",
    "FunctionCallOutput",
    "RunContext",
    "Plugin",
    "AgentSession",
    "AgentEvent",
    "Agent",
    "cli",
    "AssignmentTimeoutError",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "APIConnectOptions",
    "AgentState",
    "NotGiven",
    "NOT_GIVEN",
    "NotGivenOr",
    "DEFAULT_API_CONNECT_OPTIONS",
    "BackgroundAudio",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
