from typing import override
from faster_whisper import download_model
from livekit.agents import Plugin

from .stt import WhisperSTT, WhisperSTTOptions, WhisperModel
from .log import logger
from .version import __version__

__all__ = ["WhisperSTT", "WhisperSTTOptions", "WhisperModel", "__version__"]


class WhisperPlugin(Plugin):
    def __init__(self):
        super().__init__(__name__, __version__, str(__package__), logger)

    @override
    def download_files(self) -> None:
        opts = WhisperSTTOptions()
        model_path = download_model(
            opts.model.value,
            local_files_only=False,
            cache_dir=None,
        )
        logger.info(f"Whisper model downloaded to {model_path}")


Plugin.register_plugin(WhisperPlugin())

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
