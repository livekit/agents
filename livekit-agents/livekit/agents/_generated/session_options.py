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

from ..types import NotGivenOr

if TYPE_CHECKING:
    from ..voice.agent_session import SessionConnectOptions
    from ..voice.keyterm_detection import KeytermsOptions, STTContextOptions
    from ..voice.tool_executor import ToolHandlingOptions
    from ..voice.transcription.text_transforms import TextTransforms
    from ..voice.turn import EndpointingOptions, TurnDetectionMode, TurnHandlingOptions
    from .expressive import ExpressiveOptions
    from collections.abc import Sequence


class DeprecatedTurnKwargs(TypedDict):
    """The eleven deprecated `AgentSession.__init__` kwargs `_migrate_turn_handling` takes. The
    nullable ones are `Maybe<Option<f64>>`: explicit `None` means "off".
    """

    min_endpointing_delay: NotGivenOr[float]
    max_endpointing_delay: NotGivenOr[float]
    false_interruption_timeout: NotGivenOr[float | None]
    turn_detection: NotGivenOr[TurnDetectionMode | None]
    discard_audio_if_uninterruptible: NotGivenOr[bool]
    min_interruption_duration: NotGivenOr[float]
    min_interruption_words: NotGivenOr[int]
    allow_interruptions: NotGivenOr[bool]
    """Tested `== Given(false)`, **not** `.is_given()`: `allow_interruptions=True` writes nothing."""
    resume_false_interruption: NotGivenOr[bool]
    agent_false_interruption_timeout: NotGivenOr[float | None]
    preemptive_generation: NotGivenOr[bool]


class SessionKwargs(TypedDict):
    """Every option keyword of the reference's `AgentSession.__init__`, once. Host-object kwargs
    (`stt`, `llm`, `tools`, ...) are legs, not options.
    """

    turn_handling: NotGivenOr[TurnHandlingOptions]
    """`turn_handling=`, as written. `None` is `NOT_GIVEN`."""
    stt_context_options: NotGivenOr[STTContextOptions]
    tool_handling: NotGivenOr[ToolHandlingOptions]
    max_tool_steps: int
    use_tts_aligned_transcript: NotGivenOr[bool]
    tts_text_transforms: NotGivenOr[Sequence[TextTransforms] | None]
    """`NotGivenOr[Sequence[TextTransforms] | None]`: unsaid, "none", or a set."""
    min_consecutive_speech_delay: float
    expressive: bool | ExpressiveOptions
    """`bool | ExpressiveOptions`: whether expressive delivery is on; the dict stays host-side."""
    aec_warmup_duration: NotGivenOr[float | None]
    """`NotGivenOr[float | None]`: unsaid is the default warmup, `None` none at all (outbound SIP)."""
    ivr_detection: bool
    user_away_timeout: float | None
    """`float | None`: `None` disables the away timer."""
    transcription_timeout: float | None
    """`float | None`: `None` disables the timeout, `0` arms it with nothing left to wait."""
    session_close_transcript_timeout: float
    conn_options: NotGivenOr[SessionConnectOptions]
    deprecated: DeprecatedTurnKwargs
    """The eleven deprecated turn kwargs, as one sub-dict."""
    keyterms_options: NotGivenOr[KeytermsOptions]
    """Deprecated; `stt_context_options` supersedes it silently when both are given."""


class UpdateKwargs(TypedDict):
    """`AgentSession.update_options(..)`'s option surface (`keyterms` and `expressive` stay host-side)."""

    endpointing_opts: NotGivenOr[EndpointingOptions]
    turn_detection: NotGivenOr[TurnDetectionMode | None]
    """`NotGivenOr[TurnDetectionMode | None]`: `None` reverts to automatic selection (`Auto`)."""
    min_endpointing_delay: NotGivenOr[float]
    max_endpointing_delay: NotGivenOr[float]


class SessionLegs(TypedDict):
    """`LegSet` as the host states it before `start()`, so `session.options` can resolve early."""

    stt: bool
    vad: bool
    using_default_vad: bool
    llm: bool
    tts: bool
    realtime: bool
    detector: Literal["streaming", "batch"] | None
    """`"streaming"`, `"batch"` or `None`."""
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
    mode: str
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


class EndpointingOverrides(TypedDict, total=False):
    """The session layer's sparse endpointing (`AgentSessionOptions.endpointing_overrides`)."""

    mode: str
    min_delay: float
    max_delay: float
    alpha: float


class ResolvedSessionOptions(TypedDict):
    """`SessionConfig` as the host reads it: every public option of `session.options`, seconds as
    `f64`. Built only by `resolved_session_options`, whose exhaustive destructure keeps it complete.
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
