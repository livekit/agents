"""Hamsa plugin for LiveKit Agents: Arabic dialect text-to-speech.

See https://docs.tryhamsa.com for the API behind it.
"""

from .models import TTSDialect, TTSSampleRate
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "TTSDialect",
    "TTSSampleRate",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class HamsaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(HamsaPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
