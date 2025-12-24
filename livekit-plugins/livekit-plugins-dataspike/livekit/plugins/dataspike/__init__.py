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

"""Dataspike deepfake detection plugin for LiveKit Agents."

This module provides a realtime deepfake detector that attaches to a LiveKit room,
samples frames from subscribed remote video tracks, and streams them to the
Dataspike WebSocket API for analysis. Results can be published back into the room
or handled via a custom callback.

Typical usage:
    >>> detector = DataspikeDetector()
    >>> await detector.start(agent_session, room)

Public API:
- `InputTrack`: Internal helper representing a sampled video track.
- `DataspikeDetector`: The main detector that manages sampling and the WS pipeline.
"""

from .detector import DataspikeDetector
from .log import logger
from .version import __version__

__all__ = [
    "DataspikeDetector",
    "InputTrack",
    "logger",
    "__version__",
]

from livekit.agents import Plugin


class DataspikePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DataspikePlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
