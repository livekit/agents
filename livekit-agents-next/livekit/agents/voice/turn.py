from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from livekit import rtc

from .._generated.language import LanguageCode
from .._generated.turn import (
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
    TurnHandlingOptions,
    UserTurnLimitOptions,
)
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions


@dataclass
class TurnDetectionEvent:
    type: Literal["eot_prediction"]
    end_of_turn_probability: float
    last_speaking_time: float
    detection_delay: float | None = None
    """Latest input audio creation time -> prediction receive time."""
    inference_duration: float | None = None
    """Server-side model inference time."""
    backchannel_probability: float | None = None
    """How appropriate it is for the agent to backchannel at this pause.
    ``None`` when the detector does not produce one (e.g. the local mini model)."""


@runtime_checkable
class _StreamingTurnDetectorStream(Protocol):
    """I/O stream for the streaming turn detector."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...
    @property
    def is_fallback(self) -> bool: ...
    @property
    def prediction_timeout(self) -> float: ...

    async def unlikely_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def backchannel_threshold(self, language: LanguageCode | None) -> float | None: ...
    async def supports_language(self, language: LanguageCode | None) -> bool: ...

    def predict(self) -> asyncio.Future[TurnDetectionEvent]: ...
    def cancel_inference(self, *, timed_out: bool = False) -> None: ...
    def flush(self, reason: str | None = None) -> None: ...
    def push_audio(self, frame: rtc.AudioFrame) -> None: ...
    def end_input(self) -> None: ...
    async def aclose(self) -> None: ...


@runtime_checkable
class _StreamingTurnDetector(Protocol):
    """Turn detector that processes streaming data."""

    @property
    def model(self) -> str: ...
    @property
    def provider(self) -> str: ...

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> _StreamingTurnDetectorStream: ...


TurnDetectionMode = Literal["stt", "vad", "realtime_llm", "manual"] | _StreamingTurnDetector
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- a turn detector (``inference.TurnDetector`` by default): the default mode, the end of the
  user's turn predicted on their audio

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""

__all__ = [
    "EndpointingOptions",
    "InterruptionOptions",
    "PreemptiveGenerationOptions",
    "TurnDetectionEvent",
    "TurnDetectionMode",
    "TurnHandlingOptions",
    "UserTurnLimitOptions",
]
