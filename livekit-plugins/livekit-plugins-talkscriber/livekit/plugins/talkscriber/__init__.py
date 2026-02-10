from livekit.agents import Plugin

from .log import logger
from .stt import STT, SpeechStream
from .tts import TTS as TTSClass, ChunkedStream, SynthesizeStream, configure_for_server
from .version import __version__

__all__ = [
    "STT",
    "SpeechStream",
    "TTS",
    "ChunkedStream",
    "SynthesizeStream",
    "configure_for_server",
    "__version__",
]

# Re-export TTS with the expected name
TTS = TTSClass


class TalkscriberPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(TalkscriberPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
