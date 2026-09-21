# Copyright 2025 LiveKit, Inc.
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

"""TypeSafe reviewer for LiveKit Agents.

Judges an agent's replies against its own prompt and tool catalog using
TypeSafe's System One model (Jev), and nudges the agent back on course when it
drifts. See :class:`Reviewer`.
"""

from livekit.agents import Plugin

from ._client import SystemOneClient
from .checks import CALIBRATED_FOR, Check, TurnState, default_checks
from .log import logger
from .reviewer import Reviewer, Verdict
from .version import __version__


class TypeSafePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(TypeSafePlugin())

__all__ = [
    "CALIBRATED_FOR",
    "Check",
    "Reviewer",
    "SystemOneClient",
    "TurnState",
    "TypeSafePlugin",
    "Verdict",
    "default_checks",
    "__version__",
]
