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

from . import (
    cli,
    ipc,
    llm,
    metrics,
    multimodal,
    pipeline,
    stt,
    tokenize,
    transcription,
    tts,
    utils,
    vad,
    voice_assistant,
)
from ._constants import ATTRIBUTE_AGENT_STATE
from ._exceptions import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AssignmentTimeoutError,
)
from ._types import AgentState
from .job import AutoSubscribe, JobContext, JobExecutorType, JobProcess, JobRequest
from .plugin import Plugin
from .version import __version__
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
    "Plugin",
    "ipc",
    "stt",
    "vad",
    "utils",
    "tts",
    "tokenize",
    "llm",
    "metrics",
    "transcription",
    "pipeline",
    "multimodal",
    "voice_assistant",
    "cli",
    "AssignmentTimeoutError",
    "APIConnectionError",
    "APIError",
    "APIStatusError",
    "APITimeoutError",
    "ATTRIBUTE_AGENT_STATE",
    "AgentState",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
