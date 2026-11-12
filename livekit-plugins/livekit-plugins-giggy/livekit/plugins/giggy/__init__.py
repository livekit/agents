"""Giggy text-to-speech and utterance speech recognition for LiveKit Agents."""

from livekit.agents import Plugin

from .log import logger
from .stt import STT
from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["STT", "TTS", "ChunkedStream", "__version__"]


class GiggyPlugin(Plugin):
    """Register the Giggy provider with LiveKit."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(GiggyPlugin())
