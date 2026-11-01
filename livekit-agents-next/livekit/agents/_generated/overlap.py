# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from typing import Literal

from numpy import float32, int16
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer


class OverlappingSpeechEvent(BaseModel):
    """`OverlappingSpeechEvent` (`inference/interruption.py:120`), the detector's only public verdict."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["overlapping_speech"] = "overlapping_speech"
    created_at: float = Field(default_factory=time.time)
    """Wall clock seconds at construction."""
    detected_at: float = Field(default_factory=time.time)
    """Wall clock seconds at which the overlap was resolved."""
    is_interruption: bool = False
    """The verdict. `false` is only a *backchannel* verdict when `agent_ended` is also false."""
    agent_ended: bool = False
    """The overlap ended because the agent stopped talking; `is_interruption == false` is then
    inconclusive.
    """
    total_duration: float = 0.0
    prediction_duration: float = 0.0
    detection_delay: float = 0.0
    overlap_started_at: float | None = None
    """Wall clock seconds at which the overlap began."""
    speech_input: NDArray[int16] | None = None
    probabilities: NDArray[float32] | None = None
    probability: float = 0.0
    num_requests: int = 0
    """Requests sent for this overlap. Reset to zero as it is read."""

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> object:
        from .. import _bindings

        return _bindings.overlapping_speech_serialize(self, handler)


OverlappingSpeechEvent.__module__ = "livekit.agents.inference.interruption"
