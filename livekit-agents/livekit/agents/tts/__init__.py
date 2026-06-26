from .fallback_adapter import (
    AvailabilityChangedEvent,
    FallbackAdapter,
    FallbackChunkedStream,
    FallbackSynthesizeStream,
)
from .stream_adapter import StreamAdapter, StreamAdapterWrapper
from .stream_pacer import SentenceStreamPacer
from .tts import (
    TTS,
    AudioEmitter,
    ChunkedStream,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
    TTSError,
)

__all__ = [
    "TTS",
    "SynthesizedAudio",
    "SynthesizeStream",
    "TTSCapabilities",
    "StreamAdapterWrapper",
    "StreamAdapter",
    "ChunkedStream",
    "AvailabilityChangedEvent",
    "FallbackAdapter",
    "FallbackChunkedStream",
    "FallbackSynthesizeStream",
    "AudioEmitter",
    "TTSError",
    "SentenceStreamPacer",
]


# Cleanup docs of unexported modules
_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]

__pdoc__ = {}

for n in NOT_IN_ALL:
    __pdoc__[n] = False
