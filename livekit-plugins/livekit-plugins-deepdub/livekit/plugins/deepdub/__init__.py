"""Deepdub plugin for LiveKit Agents: streaming text-to-speech and speech-to-text.

See https://docs.deepdub.ai for the API behind it.
"""

from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "TTS", "ChunkedStream", "SynthesizeStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class DeepdubPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DeepdubPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
