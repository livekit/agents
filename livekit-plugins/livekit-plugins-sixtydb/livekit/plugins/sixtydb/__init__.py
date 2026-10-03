"""60db HTTP text-to-speech plugin for LiveKit Agents."""

from livekit.agents import Plugin

from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "__version__"]


class SixtyDBPlugin(Plugin):
    """Register the 60db speech plugin."""

    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__)


Plugin.register_plugin(SixtyDBPlugin())
