"""Deepdub plugin for LiveKit Agents"""

from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "SynthesizeStream", "__version__"]

from livekit.agents import Plugin


class DeepdubPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)


Plugin.register_plugin(DeepdubPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
