# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/vad/events.rs
#     rust/crates/lk-agents-abi/src/vad/vad.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique

from livekit import rtc


@unique
class VADEventType(str, Enum):
    START_OF_SPEECH = "start_of_speech"
    INFERENCE_DONE = "inference_done"
    END_OF_SPEECH = "end_of_speech"


@dataclass
class VADEvent:
    """
    Represents an event detected by the Voice Activity Detector (VAD).
    """

    type: VADEventType
    """Type of the VAD event (e.g., start of speech, end of speech, inference done)."""
    samples_index: int
    """Index of the audio sample where the event occurred, relative to the inference sample rate."""
    timestamp: float
    """Timestamp (in seconds) when the event was fired."""
    speech_duration: float
    """Duration of the speech segment in seconds."""
    silence_duration: float
    """Duration of the silence segment in seconds."""
    frames: list[rtc.AudioFrame] = field(default_factory=list)
    """
    List of audio frames associated with the speech.

    - For `start_of_speech` events, this contains the audio chunks that triggered the detection.
    - For `inference_done` events, this contains the audio chunks that were processed.
    - For `end_of_speech` events, this contains the complete user speech.
    """
    probability: float = 0.0
    """Probability that speech is present (only for `INFERENCE_DONE` events)."""
    inference_duration: float = 0.0
    """Time taken to perform the inference, in seconds (only for `INFERENCE_DONE` events)."""
    speaking: bool = False
    """Indicates whether speech was detected in the frames."""
    raw_accumulated_silence: float = 0.0
    """Threshold used to detect silence."""
    raw_accumulated_speech: float = 0.0
    """Threshold used to detect speech."""


@dataclass
class VADCapabilities:
    update_interval: float


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
VADEventType.__module__ = "livekit.agents.vad"
VADEvent.__module__ = "livekit.agents.vad"
VADCapabilities.__module__ = "livekit.agents.vad"
