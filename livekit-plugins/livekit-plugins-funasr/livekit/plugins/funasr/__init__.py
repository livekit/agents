"""FunASR plugin for LiveKit Agents — self-hosted speech-to-text (SenseVoice / Paraformer / Fun-ASR-Nano)."""
from livekit.agents import Plugin
from .log import logger
from .stt import STT
from .version import __version__

__all__ = ["STT", "__version__"]


class FunASRPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(FunASRPlugin())
