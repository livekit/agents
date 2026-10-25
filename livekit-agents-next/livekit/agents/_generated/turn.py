# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/session_options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from ..voice.turn import TurnDetectionMode


class EndpointingOptions(TypedDict, total=False):
    """Configuration for endpointing.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).
    """

    mode: Literal["fixed", "dynamic"]
    """Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``."""
    min_delay: float
    """Minimum time (s) since last detected speech before declaring the
    user's turn complete. Defaults to ``0.5``.
    """
    max_delay: float
    """Maximum time (s) the agent waits before terminating the turn.
    Defaults to ``3.0``.
    """
    alpha: float
    """Exponential moving average coefficient for dynamic endpointing.
    The higher the value, the more weight is given to the history.
    Defaults to ``0.9``. Only applies when mode is ``dynamic``.
    """


class InterruptionOptions(TypedDict, total=False):
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
    Absent means auto-detect.
    """
    discard_audio_if_uninterruptible: bool
    """Drop buffered audio while the agent speaks and cannot be
    interrupted. Defaults to ``True``.
    """
    min_duration: float
    """Minimum speech length (s) to register as an interruption.
    Defaults to ``0.5``.
    """
    min_words: int
    """Minimum word count to consider an interruption (STT only).
    Defaults to ``0``.
    """
    resume_false_interruption: bool
    """Resume the agent's speech after a false interruption.
    Defaults to ``True``.
    """
    false_interruption_timeout: float | None
    """Seconds of silence after an interruption before it is
    classified as false. ``None`` disables. Defaults to ``2.0``.
    """
    backchannel_boundary: float | tuple[float, float] | None
    """Seconds near the start/end of each agent turn during which overlapping
    speech classified as a backchannel by the adaptive detector is suppressed
    (events flagged as interruptions still pass through). Use a tuple to apply
    different values for start and end separately. ``None`` disables. Defaults
    to ``(1.0, 1.0)``. End value accounts for STT transcript timestamp
    inaccuracy.
    """


class PreemptiveGenerationOptions(TypedDict, total=False):
    """Configuration for preemptive generation."""

    enabled: bool
    """Whether preemptive generation is enabled. Defaults to ``True``."""
    preemptive_tts: bool
    """Whether to also run TTS preemptively before the turn is confirmed.
    When ``False`` (default), only LLM runs preemptively; TTS starts once the
    turn is confirmed and the speech is scheduled.
    """
    max_speech_duration: float
    """Maximum user speech duration (s) for which preemptive generation
    is attempted. Beyond this threshold, preemptive generation is skipped
    since long utterances are more likely to change and users may expect
    slower responses. Defaults to ``10.0``.
    """
    max_retries: int
    """Maximum number of preemptive generation attempts per user turn.
    The counter resets when the turn completes. Defaults to ``3``.
    """


class UserTurnLimitOptions(TypedDict, total=False):
    """Configuration for detecting when a user has been speaking too long
    without the agent successfully responding.

    The framework tracks accumulated word count and wall-clock duration
    across consecutive user turns. Counters only reset when the agent
    transitions to ``speaking`` state (i.e., produces audio output).

    Both thresholds default to ``None`` (disabled). Set at least one to
    enable the feature.
    """

    max_words: int | None
    """Maximum accumulated word count before triggering. Uses the
    framework's WordTokenizer for counting. ``None`` disables word-based
    limiting. Defaults to ``None``.
    """
    max_duration: float | None
    """Maximum wall-clock duration (seconds) since the user first started
    speaking in the current accumulation window. ``None`` disables
    duration-based limiting. Defaults to ``None``.
    """


class TurnHandlingOptions(TypedDict, total=False):
    """Configuration for the turn handling system.

    Can be passed as a plain dict::

        AgentSession(
            turn_handling={
                "endpointing": {"min_delay": 0.3},
                "interruption": {"enabled": False},
                "preemptive_generation": {"preemptive_tts": True},
            },
        )

    All keys are optional and default to sensible values.
    """

    turn_detection: TurnDetectionMode | None
    """Strategy for deciding when the user has finished speaking.
    Absent means the session auto-selects.
    """
    endpointing: EndpointingOptions
    """Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``."""
    interruption: InterruptionOptions
    """Interruption handling configuration. Use ``{"enabled": False}`` to disable."""
    preemptive_generation: PreemptiveGenerationOptions
    """Preemptive generation configuration. Use ``{"enabled": False}`` to disable."""
    user_turn_limit: UserTurnLimitOptions
    """User turn limit configuration. Use ``{"max_words": 50}`` to enable."""


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
EndpointingOptions.__module__ = "livekit.agents.voice.turn"
InterruptionOptions.__module__ = "livekit.agents.voice.turn"
PreemptiveGenerationOptions.__module__ = "livekit.agents.voice.turn"
UserTurnLimitOptions.__module__ = "livekit.agents.voice.turn"
TurnHandlingOptions.__module__ = "livekit.agents.voice.turn"
