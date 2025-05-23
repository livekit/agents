from .fallback_adapter import AvailabilityChangedEvent, FallbackAdapter
from .stream_adapter import StreamAdapter, StreamAdapterWrapper
from .stt import (
    STT,
    RecognitionUsage,
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
    SpeechStream,
    STTCapabilities,
    STTError,
)

__all__ = [
    "SpeechEventType",
    "SpeechEvent",
    "SpeechData",
    "RecognizeStream",
    "SpeechStream",
    "STT",
    "STTCapabilities",
    "StreamAdapter",
    "StreamAdapterWrapper",
    "RecognitionUsage",
    "FallbackAdapter",
    "AvailabilityChangedEvent",
    "STTError",
]

# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
