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

"""Speechmatics STT plugin for LiveKit Agents

See https://docs.livekit.io/agents/integrations/stt/speechmatics/ for more information.
"""

from typing import Any

from speechmatics.agent_stt import (
    AdditionalVocabEntry,
    AudioEncoding,
    Model,
    SpeakerIdentifier,
)

from .stt import STT, SpeechStream, TurnDetectionMode
from .tts import TTS
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "TurnDetectionMode",
    "SpeechStream",
    "AdditionalVocabEntry",
    "AudioEncoding",
    "Model",
    "SpeakerIdentifier",
    "logger",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class SpeechmaticsPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(SpeechmaticsPlugin())


# Names this plugin exported before Agent STT, served from the voice SDK that defined them so
# they are the same objects callers already hold. Resolved lazily, which both keeps the import
# off the hot path and lets the lookup warn — binding them at module level would skip
# `__getattr__` entirely. Remove these, and the `speechmatics-voice` dependency, after 2026-10-05.
_warned_deprecated: set[str] = set()


def _warn_deprecated(name: str, guidance: str) -> None:
    # A single `from ... import` looks the name up twice.
    if name not in _warned_deprecated:
        _warned_deprecated.add(name)
        logger.warning(f"`{name}` is deprecated and will be removed after 2026-10-05; {guidance}")


def __getattr__(name: str) -> Any:
    if name == "OperatingPoint":
        from speechmatics.voice import OperatingPoint

        _warn_deprecated(name, "use `model` instead (Agent STT accepts only `linden-1`)")
        return OperatingPoint

    if name == "SpeakerFocusMode":
        from speechmatics.voice import SpeakerFocusMode

        _warn_deprecated(
            name,
            "speaker focus is not supported by Agent STT and `focus_mode` is ignored; it is "
            "expected to be reintroduced in a future release",
        )
        return SpeakerFocusMode

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
