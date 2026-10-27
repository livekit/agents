from __future__ import annotations

from .. import _core
from .._generated.tts import (
    TTS,
    ChunkedStream,
    MarkupInfo,
    SynthesizedAudio,
    SynthesizeStream,
    TTSCapabilities,
    TTSError,
)

__all__ = [
    "TTS",
    "AudioEmitter",
    "ChunkedStream",
    "MarkupInfo",
    "SynthesizedAudio",
    "SynthesizeStream",
    "TTSCapabilities",
    "TTSError",
]

# The core's class under its public name: `audio/pcm` is framed as it is, any other container
# decoded (ffmpeg) as it arrives.
AudioEmitter = _core.AudioEmitter
