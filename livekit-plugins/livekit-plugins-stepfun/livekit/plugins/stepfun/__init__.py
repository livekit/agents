"""LiveKit Agents plugin for StepFun (阶跃星辰) ASR, TTS and chat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from livekit.agents import Plugin

from .log import logger
from .models import (
    Cluster,
    Language,
    LLMModels,
    ReasoningEffort,
    STTBatchModels,
    STTModels,
    TTSModels,
    TTSVoices,
)
from .stt import STT, RecognizeStream, SpeechStream
from .tts import TTS, ChunkedStream, SynthesizeStream
from .version import __version__

# Register the optional OpenAI plugin on the importing thread, before jobs start.
try:
    from .llm import LLM

    _LLM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _LLM_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from .llm import LLM

__all__ = [
    "LLM",
    "STT",
    "TTS",
    "ChunkedStream",
    "Cluster",
    "Language",
    "LLMModels",
    "ReasoningEffort",
    "RecognizeStream",
    "STTBatchModels",
    "STTModels",
    "SpeechStream",
    "SynthesizeStream",
    "TTSModels",
    "TTSVoices",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Explain a missing `llm` extra at access time.

    Only reached when the eager import above failed, since a successful one
    binds LLM as a module global and module __getattr__ is never consulted for
    names that already exist.
    """
    if name == "LLM":
        assert _LLM_IMPORT_ERROR is not None
        raise _LLM_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
