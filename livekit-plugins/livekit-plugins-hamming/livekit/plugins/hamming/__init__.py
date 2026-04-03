# Copyright 2026 Hamming, Inc.
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

"""Hamming plugin for LiveKit Agents.

Exports final post-call monitoring artifacts to Hamming.
"""

from livekit.agents import Plugin

from ._setup import (
    DoctorReport,
    attach_session,
    configure_hamming,
    doctor,
    doctor_json,
)
from .log import logger
from .version import __version__

__all__ = [
    "configure_hamming",
    "doctor",
    "doctor_json",
    "DoctorReport",
    "attach_session",
    "__version__",
]


class HammingPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(HammingPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__: dict[str, bool] = {}
for n in NOT_IN_ALL:
    __pdoc__[n] = False
