from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

from ..types import NOT_GIVEN, NotGivenOr
from ..utils import is_given
from .audio_recognition import TurnDetectionMode


class EndpointingConfig(TypedDict, total=False):
    """Configuration for endpointing.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).
    """

    min_delay: float
    """Minimum time (s) since last detected speech before declaring the
    user's turn complete. Defaults to ``0.5``."""
    max_delay: float
    """Maximum time (s) the agent waits before terminating the turn.
    Defaults to ``3.0``."""


_ENDPOINTING_DEFAULTS: EndpointingConfig = {
    "min_delay": 0.5,
    "max_delay": 3.0,
}


class InterruptionConfig(TypedDict, total=False):
    """Configuration for interruption handling.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).

    ``mode`` absent means the session picks the best available strategy.
    """

    enabled: bool
    """Whether interruptions are enabled. Defaults to ``True``."""
    mode: Literal["adaptive", "vad"]
    """Interruption handling strategy. ``"adaptive"`` for ML-based
    detection, ``"vad"`` for simple voice-activity detection.
    Absent means auto-detect."""
    discard_audio_if_uninterruptible: bool
    """Drop buffered audio while the agent speaks and cannot be
    interrupted. Defaults to ``True``."""
    min_duration: float
    """Minimum speech length (s) to register as an interruption.
    Defaults to ``0.5``."""
    min_words: int
    """Minimum word count to consider an interruption (STT only).
    Defaults to ``0``."""
    resume_false_interruption: bool
    """Resume the agent's speech after a false interruption.
    Defaults to ``True``."""
    false_interruption_timeout: float | None
    """Seconds of silence after an interruption before it is
    classified as false. ``None`` disables. Defaults to ``2.0``."""


_INTERRUPTION_DEFAULTS: InterruptionConfig = {
    "enabled": True,
    "discard_audio_if_uninterruptible": True,
    "min_duration": 0.5,
    "min_words": 0,
    "resume_false_interruption": True,
    "false_interruption_timeout": 2.0,
}


class TurnHandlingConfig(TypedDict, total=False):
    """Configuration for the turn handling system.

    Can be passed as a plain dict::

        AgentSession(
            turn_handling={
                "endpointing": {"min_delay": 0.3},
                "interruption": {"enabled": False},
            },
        )

    All keys are optional and default to sensible values.
    """

    turn_detection: TurnDetectionMode | None
    """Strategy for deciding when the user has finished speaking.
    Absent means the session auto-selects."""
    endpointing: EndpointingConfig
    """Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``."""
    interruption: InterruptionConfig
    """Interruption handling configuration. Use ``{"enabled": False}`` to disable."""


def _resolve_endpointing(config: EndpointingConfig | None = None) -> EndpointingConfig:
    """Fill in defaults for missing keys."""
    if config is None:
        return EndpointingConfig(**_ENDPOINTING_DEFAULTS)
    return EndpointingConfig(**{**_ENDPOINTING_DEFAULTS, **config})


def _resolve_interruption(
    config: InterruptionConfig | None = None,
) -> InterruptionConfig:
    """Fill in defaults for missing keys (``mode`` stays absent if not provided)."""
    if config is None:
        return InterruptionConfig(**_INTERRUPTION_DEFAULTS)
    return InterruptionConfig(**{**_INTERRUPTION_DEFAULTS, **config})


def _migrate_turn_handling(
    min_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    max_endpointing_delay: NotGivenOr[float] = NOT_GIVEN,
    false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
    turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
    discard_audio_if_uninterruptible: NotGivenOr[bool] = NOT_GIVEN,
    min_interruption_duration: NotGivenOr[float] = NOT_GIVEN,
    min_interruption_words: NotGivenOr[int] = NOT_GIVEN,
    allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    resume_false_interruption: NotGivenOr[bool] = NOT_GIVEN,
    agent_false_interruption_timeout: NotGivenOr[float | None] = NOT_GIVEN,
) -> TurnHandlingConfig:
    """Build a TurnHandlingConfig from deprecated keyword arguments."""
    if is_given(agent_false_interruption_timeout):
        false_interruption_timeout = agent_false_interruption_timeout

    result: TurnHandlingConfig = {}

    # endpointing — only include keys that were explicitly provided
    endpointing: EndpointingConfig = {}
    if is_given(min_endpointing_delay):
        endpointing["min_delay"] = min_endpointing_delay
    if is_given(max_endpointing_delay):
        endpointing["max_delay"] = max_endpointing_delay
    if endpointing:
        result["endpointing"] = endpointing

    # interruption — only include keys that were explicitly provided
    interruption: InterruptionConfig = {}
    if allow_interruptions is False:
        interruption["enabled"] = False
    if is_given(discard_audio_if_uninterruptible):
        interruption["discard_audio_if_uninterruptible"] = discard_audio_if_uninterruptible
    if is_given(min_interruption_duration):
        interruption["min_duration"] = min_interruption_duration
    if is_given(min_interruption_words):
        interruption["min_words"] = min_interruption_words
    if is_given(false_interruption_timeout):
        interruption["false_interruption_timeout"] = false_interruption_timeout
    if is_given(resume_false_interruption):
        interruption["resume_false_interruption"] = resume_false_interruption
    if interruption:
        result["interruption"] = interruption

    if is_given(turn_detection):
        result["turn_detection"] = turn_detection

    return result
