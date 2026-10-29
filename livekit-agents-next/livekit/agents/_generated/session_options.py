# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/session_options.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NotRequired, TypeAlias, TypedDict

from ..types import NotGivenOr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..voice.agent_session import SessionConnectOptions, TextTransforms
    from ..voice.turn import (
        EndpointingOptions,
        InterruptionOptions,
        TurnDetectionMode,
        TurnHandlingOptions,
    )
    from .expressive import ExpressiveOptions
    from .keyterms import STTContextOptions


# The declared records.


DetectorKind: TypeAlias = Literal["streaming"]
"""Picks the endpointing base and the turn machine's countdown arm."""


class SessionLegs(TypedDict):
    """`LegSet` as the host states it before `start()`, so `session.options` can resolve early."""

    stt: bool
    vad: bool
    using_default_vad: bool
    llm: bool
    tts: bool
    realtime: bool
    detector: DetectorKind | None
    """An end-of-turn detector was built, and which protocol it speaks."""
    adaptive_detector: bool
    audio_enabled: bool
    audio_output: bool
    sink_can_pause: bool


class ResolvedEndpointing(TypedDict):
    """The resolved `endpointing` group, seconds as `f64`."""

    mode: str
    min_delay: float
    max_delay: float
    alpha: float


class ResolvedInterruption(TypedDict):
    """The resolved `interruption` group. `mode` is absent when the session picks."""

    enabled: bool
    mode: NotRequired[str]
    discard_audio_if_uninterruptible: bool
    min_duration: float
    min_words: int
    resume_false_interruption: bool
    false_interruption_timeout: float | None
    backchannel_boundary: tuple[float, float] | None


class ResolvedPreemptiveGeneration(TypedDict):
    """The resolved `preemptive_generation` group."""

    enabled: bool
    preemptive_tts: bool
    max_speech_duration: float
    max_retries: int


class ResolvedUserTurnLimit(TypedDict):
    """The resolved `user_turn_limit` group."""

    max_words: int | None
    max_duration: float | None


class EndpointingOverrides(TypedDict):
    """The session layer's sparse endpointing (`AgentSessionOptions.endpointing_overrides`)."""

    mode: NotRequired[str]
    min_delay: NotRequired[float]
    max_delay: NotRequired[float]
    alpha: NotRequired[float]


class ResolvedSessionOptions(TypedDict):
    """The resolved `AgentSessionOptions` as the host reads it: every public option of
    `session.options`, seconds as `f64`. Built only by `resolved_session_options`, whose exhaustive
    destructure keeps it complete.
    """

    endpointing: ResolvedEndpointing
    interruption: ResolvedInterruption
    preemptive_generation: ResolvedPreemptiveGeneration
    user_turn_limit: ResolvedUserTurnLimit
    turn_detection_mode: str | None
    """The `turn_detection` the session runs on, as the public literal; `None` is `Auto`."""
    endpointing_overrides: EndpointingOverrides
    max_tool_steps: int
    min_consecutive_speech_delay: float
    use_tts_aligned_transcript: bool | None
    tts_text_transforms: list[str] | None
    ivr_detection: bool
    user_away_timeout: float | None
    transcription_timeout: float | None
    aec_warmup_duration: float | None
    session_close_transcript_timeout: float


class InterruptionGateVerdict(TypedDict):
    """`AgentActivity._resolve_interruption_detection`'s verdict: whether to build the
    `AdaptiveInterruptionDetector`, and if not, why. Every key is always present.
    """

    build: bool
    reason: str | None
    """`"cannot_gatekeep"`, `"interruptions_disabled"`, `"asked_vad"`, `"production_default"`."""
    level: str | None
    """`"warning"` or `"info"`."""
    message: str | None


class SessionKwargs(TypedDict):
    """Every option keyword of the reference's `AgentSession.__init__`, once. Host-object kwargs
    (`stt`, `llm`, `tools`, ...) are legs, not options.
    """

    turn_handling: NotGivenOr[TurnHandlingOptions]
    """`turn_handling=`, as written. `None` is `NOT_GIVEN`."""
    stt_context_options: NotGivenOr[STTContextOptions]
    max_tool_steps: int
    use_tts_aligned_transcript: NotGivenOr[bool]
    tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None]
    """`NotGivenOr[Sequence[TextTransforms] | None]`: unsaid, "none", or a set."""
    min_consecutive_speech_delay: float
    expressive: bool | ExpressiveOptions
    aec_warmup_duration: NotGivenOr[float | None]
    """`NotGivenOr[float | None]`: unsaid is the default warmup, `None` none at all (outbound SIP)."""
    ivr_detection: bool
    user_away_timeout: float | None
    """`float | None`: `None` disables the away timer."""
    transcription_timeout: float | None
    """`float | None`: `None` disables the timeout, `0` arms it with nothing left to wait."""
    session_close_transcript_timeout: float
    conn_options: NotGivenOr[SessionConnectOptions]


class UpdateKwargs(TypedDict):
    """`AgentSession.update_options(..)`'s option surface."""

    endpointing_opts: NotGivenOr[EndpointingOptions]
    turn_detection: NotGivenOr[TurnDetectionMode | None]
    """`NotGivenOr[TurnDetectionMode | None]`: `None` reverts to automatic selection (`Auto`)."""
    expressive: NotGivenOr[bool | ExpressiveOptions]
    keyterms: NotGivenOr[list[str]]
    """The user-defined keyterms, replaced; the detected ones stay."""
    interruption: NotGivenOr[InterruptionOptions]
    """Merged over the session's interruption handling; its `mode` is set at construction."""
    min_consecutive_speech_delay: NotGivenOr[float]
    tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None]
