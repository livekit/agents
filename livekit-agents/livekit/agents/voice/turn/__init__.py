"""Turn handling: endpointing/interruption/preemptive-generation/user-turn-limit
configuration (``.base``), and the audio EOT detector + stream state machine
(``.audio``).

``TurnDetectionMode`` is composed here because the union spans both halves.
"""

from __future__ import annotations

from typing import Literal

from .audio import (
    DEFAULT_SAMPLE_RATE,
    MIN_SILENCE_DURATION_MS,
    TurnDetectorOptions,
    _AudioTurnDetectionTransport,
    _AudioTurnDetector,
    _AudioTurnDetectorStream,
    _Status,
)
from .base import (
    _ENDPOINTING_DEFAULTS,
    _INTERRUPTION_DEFAULTS,
    _PREEMPTIVE_GENERATION_DEFAULTS,
    _USER_TURN_LIMIT_DEFAULTS,
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions,
    TurnDetectionEvent,
    TurnHandlingOptions,
    UserTurnLimitOptions,
    _migrate_turn_handling,
    _resolve_endpointing,
    _resolve_interruption,
    _resolve_preemptive_generation,
    _resolve_user_turn_limit,
    _TurnDetector,
)

TurnDetectionMode = (
    Literal["stt", "vad", "realtime_llm", "manual"] | _TurnDetector | _AudioTurnDetector
)
"""
The mode of turn detection to use.

- "stt": use speech-to-text result to detect the end of the user's turn
- "vad": use VAD to detect the start and end of the user's turn
- "realtime_llm": use server-side turn detection provided by the realtime LLM
- "manual": manually manage the turn detection
- _TurnDetector: use the default mode with the provided turn detector

(default) If not provided, automatically choose the best mode based on
    available models (realtime_llm -> vad -> stt -> manual)
If the needed model (VAD, STT, or RealtimeModel) is not provided, fallback to the default mode.
"""

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "MIN_SILENCE_DURATION_MS",
    "EndpointingOptions",
    "InterruptionOptions",
    "PreemptiveGenerationOptions",
    "TurnDetectionEvent",
    "TurnDetectionMode",
    "TurnDetectorOptions",
    "TurnHandlingOptions",
    "UserTurnLimitOptions",
    "_AudioTurnDetectionTransport",
    "_AudioTurnDetector",
    "_AudioTurnDetectorStream",
    "_ENDPOINTING_DEFAULTS",
    "_INTERRUPTION_DEFAULTS",
    "_PREEMPTIVE_GENERATION_DEFAULTS",
    "_Status",
    "_TurnDetector",
    "_USER_TURN_LIMIT_DEFAULTS",
    "_migrate_turn_handling",
    "_resolve_endpointing",
    "_resolve_interruption",
    "_resolve_preemptive_generation",
    "_resolve_user_turn_limit",
]
