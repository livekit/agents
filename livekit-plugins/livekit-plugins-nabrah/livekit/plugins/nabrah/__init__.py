from .stt import STT, NabrahRecognitionModel, SpeechStream

from livekit.agents import Plugin
from .version import __version__
from .log import logger


class NabrahPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(NabrahPlugin())

__all__ = ["STT", "NabrahRecognitionModel", "SpeechStream", "__version__"]
