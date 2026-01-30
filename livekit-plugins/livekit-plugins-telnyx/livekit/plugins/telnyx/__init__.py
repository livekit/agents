from .stt import STT
from .tts import TTS
from .version import __version__

__all__ = ["STT", "TTS", "__version__"]


from livekit.agents import Plugin

from .log import logger


class TelnyxPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(TelnyxPlugin())
