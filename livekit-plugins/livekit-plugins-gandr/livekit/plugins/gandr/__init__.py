"""Gandr TTS plugin for LiveKit Agents."""

from __future__ import annotations

from livekit.agents import Plugin

from .tts import TTS, ChunkedStream
from .version import __version__

GandrTTS = TTS  # the name the standalone file used

__all__ = ["TTS", "GandrTTS", "ChunkedStream", "__version__"]


class GandrPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)


Plugin.register_plugin(GandrPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
