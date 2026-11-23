"""LiveKit Agents plugin for StepFun (阶跃星辰) ASR, TTS and chat models."""

from __future__ import annotations

from livekit.agents import Plugin

from .log import logger
from .models import (
    Cluster,
    Language,
    STTBatchModels,
    STTModels,
    TTSModels,
    TTSVoices,
)
from .stt import STT, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

__all__ = [
    "STT",
    "TTS",
    "ChunkedStream",
    "Cluster",
    "Language",
    "STTBatchModels",
    "STTModels",
    "SpeechStream",
    "SynthesizeStream",
    "TTSModels",
    "TTSVoices",
    "__version__",
]


class StepFunPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(StepFunPlugin())

# Hide symbols the doc generator would otherwise pick up
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}
for _n in NOT_IN_ALL:
    __pdoc__[_n] = False
