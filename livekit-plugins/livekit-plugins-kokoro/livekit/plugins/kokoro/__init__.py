from livekit.agents import Plugin

from .log import logger
from .tts import TTS, ChunkedStream
from .version import __version__

__all__ = ["TTS", "ChunkedStream", "__version__"]


class KokoroPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__)
        self.download_model()

    def download_model(self) -> None:
        try:
            TTS.loadkokoro()
            logger.info("Kokoro TTS model downloaded ")
        except Exception as e:
            logger.error(f"Failed to download Kokoro TTS model: {str(e)}")


# Register the plugin
Plugin.register_plugin(KokoroPlugin())
