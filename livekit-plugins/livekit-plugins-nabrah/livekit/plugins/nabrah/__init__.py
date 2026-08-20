from livekit.agents import Plugin

from .log import logger
from .stt import STT, NabrahRecognitionModel, SpeechStream
from .version import __version__


class NabrahPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(NabrahPlugin())

__all__ = ["STT", "NabrahRecognitionModel", "SpeechStream", "__version__"]
