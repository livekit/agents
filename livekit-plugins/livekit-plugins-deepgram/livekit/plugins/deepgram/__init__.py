from .stt import STT, StreamOptions, RecognizeOptions, SpeechStream
from .version import __version__

__all__ = [
    "STT",
    "StreamOptions",
    "RecognizeOptions",
    "SpeechStream",
    "__version__",
]


from livekit.agents import Plugin


class DeepgramPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__)

    def download_files(self):
        pass


Plugin.register_plugin(DeepgramPlugin())
