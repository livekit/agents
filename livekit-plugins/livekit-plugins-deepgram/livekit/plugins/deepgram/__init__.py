from .stt import STT, AudioEnergyFilter, SpeechStream
from .tts import TTS
from .version import __version__

__all__ = ["STT", "SpeechStream", "AudioEnergyFilter", "__version__", "TTS"]


from livekit.agents import Plugin

from .log import logger


class DeepgramPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DeepgramPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
