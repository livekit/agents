# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/voice/session_options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypeAlias

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from livekit.agents.voice.turn import TurnDetectionMode

EndpointingMode: TypeAlias = Literal["fixed", "dynamic"]
"""``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay."""


InterruptionMode: TypeAlias = Literal["adaptive", "vad"]
"""``"adaptive"`` for ML-based detection, ``"vad"`` for simple voice-activity detection."""


class EndpointingOptions(TypedDict):
    """Configuration for endpointing.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).
    """

    mode: NotRequired[EndpointingMode]
    """Endpointing mode. ``"fixed"`` for fixed delay, ``"dynamic"`` for dynamic delay. Defaults to ``"fixed"``."""
    min_delay: NotRequired[float]
    """Minimum time (s) since last detected speech before declaring the
    user's turn complete. Defaults to ``0.5``.
    """
    max_delay: NotRequired[float]
    """Maximum time (s) the agent waits before terminating the turn.
    Defaults to ``3.0``.
    """
    alpha: NotRequired[float]
    """Exponential moving average coefficient for dynamic endpointing.
    The higher the value, the more weight is given to the history.
    Defaults to ``0.9``. Only applies when mode is ``dynamic``.
    """


class InterruptionOptions(TypedDict):
    """Configuration for interruption handling.

    All keys are optional. Missing keys inherit from the session default
    (at the ``Agent`` level) or use the documented defaults
    (at the ``AgentSession`` level).

    ``mode`` absent means the session picks the best available strategy.
    """

    enabled: NotRequired[bool]
    """Whether interruptions are enabled. Defaults to ``True``."""
    mode: NotRequired[InterruptionMode]
    """Interruption handling strategy. ``"adaptive"`` for ML-based
    detection, ``"vad"`` for simple voice-activity detection.
    Absent means auto-detect.
    """
    discard_audio_if_uninterruptible: NotRequired[bool]
    """Drop buffered audio while the agent speaks and cannot be
    interrupted. Defaults to ``True``.
    """
    min_duration: NotRequired[float]
    """Minimum speech length (s) to register as an interruption.
    Defaults to ``0.5``.
    """
    min_words: NotRequired[int]
    """Minimum word count to consider an interruption (STT only).
    Defaults to ``0``.
    """
    resume_false_interruption: NotRequired[bool]
    """Resume the agent's speech after a false interruption.
    Defaults to ``True``.
    """
    false_interruption_timeout: NotRequired[float | None]
    """Seconds of silence after an interruption before it is
    classified as false. ``None`` disables. Defaults to ``2.0``.
    """
    backchannel_boundary: NotRequired[float | tuple[float, float] | None]
    """Seconds near the start/end of each agent turn during which overlapping
    speech classified as a backchannel by the adaptive detector is suppressed
    (events flagged as interruptions still pass through). Use a tuple to apply
    different values for start and end separately. ``None`` disables. Defaults
    to ``(1.0, 1.0)``. End value accounts for STT transcript timestamp
    inaccuracy.
    """


class PreemptiveGenerationOptions(TypedDict):
    """Configuration for preemptive generation."""

    enabled: NotRequired[bool]
    """Whether preemptive generation is enabled. Defaults to ``True``."""
    preemptive_tts: NotRequired[bool]
    """Whether to also run TTS preemptively before the turn is confirmed.
    When ``False`` (default), only LLM runs preemptively; TTS starts once the
    turn is confirmed and the speech is scheduled.
    """
    max_speech_duration: NotRequired[float]
    """Maximum user speech duration (s) for which preemptive generation
    is attempted. Beyond this threshold, preemptive generation is skipped
    since long utterances are more likely to change and users may expect
    slower responses. Defaults to ``10.0``.
    """
    max_retries: NotRequired[int]
    """Maximum number of preemptive generation attempts per user turn.
    The counter resets when the turn completes. Defaults to ``3``.
    """


class UserTurnLimitOptions(TypedDict):
    """Configuration for detecting when a user has been speaking too long
    without the agent successfully responding.

    The framework tracks accumulated word count and wall-clock duration
    across consecutive user turns. Counters only reset when the agent
    transitions to ``speaking`` state (i.e., produces audio output).

    Both thresholds default to ``None`` (disabled). Set at least one to
    enable the feature.
    """

    max_words: NotRequired[int | None]
    """Maximum accumulated word count before triggering. Uses the
    framework's WordTokenizer for counting. ``None`` disables word-based
    limiting. Defaults to ``None``.
    """
    max_duration: NotRequired[float | None]
    """Maximum wall-clock duration (seconds) since the user first started
    speaking in the current accumulation window. ``None`` disables
    duration-based limiting. Defaults to ``None``.
    """


class TurnHandlingOptions(TypedDict):
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

    turn_detection: NotRequired[TurnDetectionMode | None]
    """Strategy for deciding when the user has finished speaking.
    Absent means the session auto-selects.
    """
    endpointing: NotRequired[EndpointingOptions]
    """Endpointing configuration. Defaults to ``{"min_delay": 0.5, "max_delay": 3.0}``."""
    interruption: NotRequired[InterruptionOptions]
    """Interruption handling configuration. Use ``{"enabled": False}`` to disable."""
    preemptive_generation: NotRequired[PreemptiveGenerationOptions]
    """Preemptive generation configuration. Use ``{"enabled": False}`` to disable."""
    user_turn_limit: NotRequired[UserTurnLimitOptions]
    """User turn limit configuration. Use ``{"max_words": 50}`` to enable."""


EndpointingOptions.__module__ = "livekit.agents.voice.turn"
InterruptionOptions.__module__ = "livekit.agents.voice.turn"
PreemptiveGenerationOptions.__module__ = "livekit.agents.voice.turn"
UserTurnLimitOptions.__module__ = "livekit.agents.voice.turn"
TurnHandlingOptions.__module__ = "livekit.agents.voice.turn"
