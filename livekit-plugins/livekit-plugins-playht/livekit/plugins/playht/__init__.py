from .models import TTSEngines
from .tts import DEFAULT_VOICE, TTS, Voice
from .version import __version__

__all__ = [
    "TTS",
    "Voice",
    "DEFAULT_VOICE",
    "TTSEngines",
    "__version__",
]

from livekit.agents import Plugin


class PlayHTPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)

    def download_files(self) -> None:
        self.download_files(self)


Plugin.register_plugin(PlayHTPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
