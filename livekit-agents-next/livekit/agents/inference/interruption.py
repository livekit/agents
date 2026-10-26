"""Adaptive interruption detection: the detector is the core's (``_core.InterruptionDetector``);
its event and error stay here, where the generated session events import them."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, SerializerFunctionWrapHandler, model_serializer

from .. import _core

__all__ = ["AdaptiveInterruptionDetector", "InterruptionDetectionError", "OverlappingSpeechEvent"]

AdaptiveInterruptionDetector = _core.InterruptionDetector


class InterruptionDetectionError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["interruption_detection_error"] = "interruption_detection_error"
    timestamp: float = Field(default_factory=time.time)
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


class OverlappingSpeechEvent(BaseModel):
    """Overlapping speech detected while the agent speaks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["overlapping_speech"] = "overlapping_speech"
    created_at: float = Field(default_factory=time.time)
    """When the event was emitted, in seconds."""
    detected_at: float = Field(default_factory=time.time)
    """When the overlap was detected, in seconds."""
    is_interruption: bool = False
    """Whether the overlap is an interruption."""
    agent_ended: bool = False
    """The overlap ended because the agent finished speaking; ``is_interruption`` is then
    inconclusive, not a backchannel verdict."""
    total_duration: float = 0.0
    """Round trip time of the inference, in seconds."""
    prediction_duration: float = 0.0
    """Model time of the inference, in seconds."""
    detection_delay: float = 0.0
    """From the onset of the speech to the final prediction, in seconds."""
    overlap_started_at: float | None = None
    """When the overlapping speech started, in seconds."""
    speech_input: npt.NDArray[np.int16] | None = None
    """The audio the inference ran on."""
    probabilities: npt.NDArray[np.float32] | None = None
    """The raw interruption probabilities."""
    probability: float = 0.0
    """The conservative estimate of the interruption probability."""
    num_requests: int = 0
    """Requests sent for this event."""

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Any:
        # the numpy arrays stay out of the dump
        return handler(self.model_copy(update={"speech_input": None, "probabilities": None}))
