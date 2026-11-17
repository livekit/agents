"""HUMAIN Voice plugin for LiveKit Agents: Arabic-first text-to-speech.

See https://docs.voice.humain.com for the API behind it.
"""

from .models import TTSModels, TTSVoices
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "TTSModels", "TTSVoices", "__version__"]

from livekit.agents import Plugin

from .log import logger


class HumainPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(HumainPlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
