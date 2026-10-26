from typing import TYPE_CHECKING, Any

from ._utils import InferenceClass
from .eot import TurnDetector, TurnDetectorModels, TurnDetectorVersions
from .interruption import (
    AdaptiveInterruptionDetector,
    InterruptionDetectionError,
    OverlappingSpeechEvent,
)
from .llm import (
    LLM,
    LLMModels,
    LLMStream,
    OpenAIRealtimeModels,
    RealtimeModels,
    XAIRealtimeModels,
    is_realtime_model,
    llm_from_model_string,
)
from .stt import STT, STTModels
from .tts import TTS, TTSModels
from .vad import VAD, VADModels

if TYPE_CHECKING:
    from .avatar import AvatarSession, LemonSliceOptions
    from .realtime import RealtimeModel, RealtimeSession


def __getattr__(name: str) -> Any:
    # lazy: avatar subclasses voice's AvatarSession and realtime the OpenAI realtime model,
    # and voice imports this package while it initializes
    if name in ("AvatarSession", "LemonSliceOptions"):
        from . import avatar

        return getattr(avatar, name)
    if name in ("RealtimeModel", "RealtimeSession"):
        from . import realtime

        return getattr(realtime, name)
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
    "InferenceClass",
    "VADModels",
    "AdaptiveInterruptionDetector",
    "InterruptionDetectionError",
    "OverlappingSpeechEvent",
    "TurnDetector",
    "TurnDetectorModels",
    "TurnDetectorVersions",
    "RealtimeModel",
    "RealtimeSession",
    "RealtimeModels",
    "OpenAIRealtimeModels",
    "XAIRealtimeModels",
    "is_realtime_model",
    "llm_from_model_string",
]
