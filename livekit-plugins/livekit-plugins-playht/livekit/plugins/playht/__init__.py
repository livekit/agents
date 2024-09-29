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
