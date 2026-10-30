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
from dataclasses import dataclass, field
from enum import StrEnum, unique
from typing import Literal, TypeAlias

UserState: TypeAlias = Literal["speaking", "listening", "away"]
"""Whether the user is speaking, listening, or has gone away."""


AgentState: TypeAlias = Literal["initializing", "idle", "listening", "thinking", "speaking"]
"""The agent's lifecycle label."""


@dataclass(frozen=True, kw_only=True, slots=True)
class UserStateChangedEvent:
    old_state: UserState
    new_state: UserState
    created_at: float = field(default_factory=time.time)
    type: Literal["user_state_changed"] = "user_state_changed"


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentStateChangedEvent:
    old_state: AgentState
    new_state: AgentState
    created_at: float = field(default_factory=time.time)
    type: Literal["agent_state_changed"] = "agent_state_changed"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserInputTranscribedEvent:
    transcript: str
    is_final: bool
    item_id: str | None = None
    """Provider-specific ID for the transcribed input item, when available."""
    speaker_id: str | None = None
    language: str | None = None
    created_at: float = field(default_factory=time.time)
    type: Literal["user_input_transcribed"] = "user_input_transcribed"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserTranscriptionTimeoutEvent:
    speech_duration: float
    """Total VAD-detected speech (s) in the turn that produced no transcript."""
    vad_speech_started_at: float
    """When VAD first detected speech for this (untranscribed) turn."""
    created_at: float = field(default_factory=time.time)
    type: Literal["user_transcription_timeout"] = "user_transcription_timeout"


@dataclass(frozen=True, kw_only=True, slots=True)
class EotPredictionEvent:
    probability: float
    threshold: float
    inference_duration: float
    """Server-side model inference time."""
    delay: float
    """End of user speech → prediction received latency (s), anchored on the
    VAD-backdated last_speaking_time.
    """
    created_at: float = field(default_factory=time.time)
    type: Literal["eot_prediction"] = "eot_prediction"


@dataclass(frozen=True, kw_only=True, slots=True)
class _AgentBackchannelOpportunityEvent:
    """Internal: a window in which the agent could backchannel (a short
    acknowledgment such as "mm-hmm"), as predicted by the turn detector. Passed to
    ``AgentActivity`` only — not surfaced as a public ``AgentSession`` event yet.

    ``AgentActivity`` owns the decision of what to do with it. The end-of-turn margin
    (``end_of_turn_threshold - end_of_turn_probability``) gives a progressive risk axis:
    a large positive margin means the user is clearly still going, so riskier
    backchannels (yeah/okay/right) are safe; a small margin (or a negative one, where
    ``end_of_turn_probability >= end_of_turn_threshold`` and a reply is imminent) calls
    for safe, less ambiguous ones (hmm/uh-huh) that won't collide with the reply.
    """

    probability: float
    threshold: float
    end_of_turn_probability: float
    end_of_turn_threshold: float
    language: str | None = None
    created_at: float = field(default_factory=time.time)
    type: Literal["agent_backchannel_opportunity"] = "agent_backchannel_opportunity"


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentFalseInterruptionEvent:
    resumed: bool
    """Whether the false interruption was resumed automatically."""
    created_at: float = field(default_factory=time.time)
    type: Literal["agent_false_interruption"] = "agent_false_interruption"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallUpdated:
    """A progress update emitted via ``ctx.update()`` while a tool call runs."""

    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_update_N`` when deferred."""
    call_id: str
    message: str
    type: Literal["tool_call_updated"] = "tool_call_updated"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallEnded:
    """A tool call's single terminal entry."""

    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_final`` when deferred."""
    call_id: str
    message: str | None = None
    """Result or error text; None when there is nothing to voice."""
    status: str
    """`"done"`, `"error"` or `"cancelled"`."""
    type: Literal["tool_call_ended"] = "tool_call_ended"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolReplyUpdated:
    """Lifecycle of the deferred reply that voices buffered tool updates: ``scheduled``
    when queued, then ``completed`` / ``interrupted`` / ``skipped``. One reply may cover
    several calls; an inline first update never gets one.
    """

    update_ids: list[str]
    """``ToolCallUpdated.id`` values this reply covers."""
    status: str
    """`"scheduled"`, `"completed"`, `"interrupted"` or `"skipped"`."""
    speech_id: str
    """Id of the reply speech; ``speech_created`` carries its handle."""
    type: Literal["tool_reply_updated"] = "tool_reply_updated"


@dataclass(frozen=True, kw_only=True, slots=True)
class UserTurnExceededEvent:
    transcript: str
    """Transcript from the current (uncommitted) user turn only.
    Previous turns in the accumulation window are already in the chat context.
    """
    accumulated_transcript: str
    """Full transcript since the start of user speaking."""
    accumulated_word_count: int
    """Total word count since the start of user speaking."""
    duration: float
    """Duration of the user turn in seconds."""
    created_at: float = field(default_factory=time.time)
    type: Literal["user_turn_exceeded"] = "user_turn_exceeded"


@unique
class CloseReason(StrEnum):
    ERROR = "error"
    JOB_SHUTDOWN = "job_shutdown"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    USER_INITIATED = "user_initiated"
    TASK_COMPLETED = "task_completed"


UserStateChangedEvent.__module__ = "livekit.agents.voice.events"
AgentStateChangedEvent.__module__ = "livekit.agents.voice.events"
UserInputTranscribedEvent.__module__ = "livekit.agents.voice.events"
UserTranscriptionTimeoutEvent.__module__ = "livekit.agents.voice.events"
EotPredictionEvent.__module__ = "livekit.agents.voice.events"
_AgentBackchannelOpportunityEvent.__module__ = "livekit.agents.voice.events"
AgentFalseInterruptionEvent.__module__ = "livekit.agents.voice.events"
ToolCallUpdated.__module__ = "livekit.agents.voice.events"
ToolCallEnded.__module__ = "livekit.agents.voice.events"
ToolReplyUpdated.__module__ = "livekit.agents.voice.events"
UserTurnExceededEvent.__module__ = "livekit.agents.voice.events"
CloseReason.__module__ = "livekit.agents.voice.events"
