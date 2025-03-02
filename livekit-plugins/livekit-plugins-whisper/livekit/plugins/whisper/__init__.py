from livekit.agents import Plugin

from .stt import WhisperSTT, WhisperSTTOptions, WhisperModel
from .log import logger
from .version import __version__

__all__ = ["WhisperSTT", "WhisperSTTOptions", "WhisperModel", "__version__"]


class WhisperPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, __package__, logger)

    def download_files(self) -> None:
        WhisperSTT.load()
        logger.info("Whisper model downloaded")


Plugin.register_plugin(WhisperPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
