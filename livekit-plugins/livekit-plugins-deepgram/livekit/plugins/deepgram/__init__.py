from livekit.agents import Plugin
from .version import __version__


class DeepgramPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__)

    def setup(self):
        pass


Plugin.register_plugin(DeepgramPlugin())

from .stt import STT, StreamOptions, RecognizeOptions, SpeechStream

__all__ = [
    "STT",
    "StreamOptions",
    "RecognizeOptions",
    "SpeechStream",
    "__version__",
]
