from .tts import TTS, VoiceSettings
from .version import __version__

__all__ = ["TTS", "VoiceSettings", "__version__"]


from livekit.agents import Plugin

from .log import logger


class EdgeTTSPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, str(__package__ or ""), logger)


Plugin.register_plugin(EdgeTTSPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False