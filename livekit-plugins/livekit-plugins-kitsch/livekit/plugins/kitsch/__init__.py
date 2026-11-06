"""Kitsch TTS and STT integration for LiveKit Agents."""

from livekit.agents import Plugin

from .stt import STT
from .tts import TTS
from .version import __version__

__all__ = ["STT", "TTS", "__version__"]


class KitschPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, "livekit.plugins.kitsch")


Plugin.register_plugin(KitschPlugin())
