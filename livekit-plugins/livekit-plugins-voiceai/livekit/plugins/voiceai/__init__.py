"""Voice.ai TTS plugin for LiveKit Agents

Custom TTS plugin connecting to Voice.ai API for voice cloning and synthesis.
"""

from .models import TTSEncoding, TTSLanguages, TTSModels
from .tts import TTS, Voice
from .version import __version__

__all__ = [
    "TTS",
    "TTSEncoding",
    "TTSLanguages",
    "TTSModels",
    "Voice",
    "__version__",
]

from livekit.agents import Plugin

from .log import logger


class VoiceAIPlugin(Plugin):
    """Voice.ai plugin for LiveKit Agents framework."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(VoiceAIPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
