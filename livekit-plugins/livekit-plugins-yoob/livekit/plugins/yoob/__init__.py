"""Yoob plugin for LiveKit Agents

Expressive speech (Overtone) and live transcription (Undertone) from https://yoob.com.
"""

from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "STT", "SpeechStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class YoobPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(YoobPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
