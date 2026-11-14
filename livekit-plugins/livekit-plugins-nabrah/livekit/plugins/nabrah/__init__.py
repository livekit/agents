"""Nabrah STT: original recognizers and conversational STT over the backend WebSocket."""

from livekit.agents import Plugin

from ._factory import STT, NabrahRecognitionModel
from .backchannel import BackchannelController
from .log import logger
from .stt import SpeechStream
from .stt_conv import SpeechStreamConv, STTConv
from .version import __version__


class NabrahPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(NabrahPlugin())

__all__ = [
    "STT",
    "NabrahRecognitionModel",
    "SpeechStream",
    "STTConv",
    "SpeechStreamConv",
    "BackchannelController",
    "__version__",
]
