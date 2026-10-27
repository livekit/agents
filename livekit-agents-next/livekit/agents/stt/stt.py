from __future__ import annotations

from typing import Protocol, runtime_checkable

from .._generated.stt import (
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
    "STT",
    "STTCapabilities",
    "STTError",
    "RecognitionUsage",
    "RecognizeStream",
    "SpeakerContext",
    "SpeechData",
    "SpeechEvent",
    "SpeechEventType",
    "SpeechStream",
]


# A structural type, which has no core counterpart to be generated from.
@runtime_checkable
class SpeakerContext(Protocol):
    """A speaker model an STT context provides, which the expressive system turns into LLM
    instructions."""

    def to_instructions(self) -> str: ...
