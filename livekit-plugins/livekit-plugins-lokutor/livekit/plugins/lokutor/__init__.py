"""Lokutor plugin for LiveKit Agents: speech-to-text and text-to-speech.

See https://docs.lokutor.com for the API behind it.
"""

from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "TTS", "ChunkedStream", "SynthesizeStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class LokutorPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(LokutorPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
