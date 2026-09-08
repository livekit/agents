"""Maya Research voice models for LiveKit Agents.

See https://www.mayaresearch.ai/llm.txt for the current service contract and
https://github.com/MayaResearch/maya-cookbook for runnable examples.
"""

from .models import TTSLanguages, TTSModels
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "TTSLanguages", "TTSModels", "__version__"]

from livekit.agents import Plugin

from .log import logger


class MayaPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(MayaPlugin())

__pdoc__ = {name: False for name in dir() if name not in __all__}
