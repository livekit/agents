from .stt import STT, SpeechStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "__version__"]


from livekit.agents import Plugin

from .log import logger


class DeepgramPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DeepgramPlugin())
