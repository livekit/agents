from typing import TYPE_CHECKING, Any

from .eot import TurnDetector, TurnDetectorModels, TurnDetectorVersions
from .interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDataFrameType,
    InterruptionDetectionError,
    OverlappingSpeechEvent,
)
from .llm import LLM, LLMModels, LLMStream
from .stt import STT, STTModels
from .tts import TTS, TTSModels
from .vad import VAD, VADModels

if TYPE_CHECKING:
    from .avatar import AvatarSession, LemonSliceOptions


# AvatarSession subclasses voice.avatar.AvatarSession. Because this package is
# imported *during* voice package initialization (voice.agent imports
# inference), importing .avatar eagerly here would form a circular import. Load
# it lazily on first attribute access, by which point voice is fully
# initialized. See PEP 562.
def __getattr__(name: str) -> Any:
    if name in ("AvatarSession", "LemonSliceOptions"):
        from . import avatar

        return getattr(avatar, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "STT",
    "TTS",
    "LLM",
    "VAD",
    "AvatarSession",
    "LemonSliceOptions",
    "LLMStream",
    "STTModels",
    "TTSModels",
    "LLMModels",
    "VADModels",
    "AdaptiveInterruptionDetector",
    "InterruptionDetectionError",
    "OverlappingSpeechEvent",
    "InterruptionDataFrameType",
    "TurnDetector",
    "TurnDetectorModels",
    "TurnDetectorVersions",
]
