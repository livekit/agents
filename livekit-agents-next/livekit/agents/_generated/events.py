# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/voice/events.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

import time
from enum import Enum, unique
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..llm import FunctionCall
from ..metrics import AgentSessionUsage
from .language import LanguageCode

if TYPE_CHECKING:
    from ..voice.speech_handle import SpeechHandle

# --- the two state vocabularies ---------------------------------------------------------------

UserState = Literal["speaking", "listening", "away"]
"""Whether the user is speaking, listening, or has gone away."""


AgentState = Literal["initializing", "idle", "listening", "thinking", "speaking"]
"""The agent's lifecycle label."""


# --- the session events -----------------------------------------------------------------------

class UserStateChangedEvent(BaseModel):
    type: Literal["user_state_changed"] = "user_state_changed"
    old_state: UserState
    new_state: UserState
    created_at: float = Field(default_factory=time.time)


class AgentStateChangedEvent(BaseModel):
    type: Literal["agent_state_changed"] = "agent_state_changed"
    old_state: AgentState
    new_state: AgentState
    created_at: float = Field(default_factory=time.time)


class UserInputTranscribedEvent(BaseModel):
    type: Literal["user_input_transcribed"] = "user_input_transcribed"
    transcript: str
    is_final: bool
    item_id: str | None = None
    """Provider-specific ID for the transcribed input item, when available."""
    speaker_id: str | None = None
    language: LanguageCode | None = None
    created_at: float = Field(default_factory=time.time)


class UserTranscriptionTimeoutEvent(BaseModel):
    type: Literal["user_transcription_timeout"] = "user_transcription_timeout"
    speech_duration: float
    """Total VAD-detected speech (s) in the turn that produced no transcript."""
    vad_speech_started_at: float
    """When VAD first detected speech for this (untranscribed) turn."""
    created_at: float = Field(default_factory=time.time)


class EotPredictionEvent(BaseModel):
    type: Literal["eot_prediction"] = "eot_prediction"
    probability: float
    threshold: float
    inference_duration: float
    """Server-side model inference time."""
    delay: float
    """End of user speech → prediction received latency (s), anchored on the
    VAD-backdated last_speaking_time.
    """
    created_at: float = Field(default_factory=time.time)


class _AgentBackchannelOpportunityEvent(BaseModel):
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

    type: Literal["agent_backchannel_opportunity"] = "agent_backchannel_opportunity"
    probability: float
    threshold: float
    end_of_turn_probability: float
    end_of_turn_threshold: float
    language: str | None = None
    created_at: float = Field(default_factory=time.time)


class AgentFalseInterruptionEvent(BaseModel):
    type: Literal["agent_false_interruption"] = "agent_false_interruption"
    resumed: bool
    """Whether the false interruption was resumed automatically."""
    created_at: float = Field(default_factory=time.time)


class SessionUsageUpdatedEvent(BaseModel):
    type: Literal["session_usage_updated"] = "session_usage_updated"
    usage: AgentSessionUsage
    created_at: float = Field(default_factory=time.time)


class SpeechCreatedEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["speech_created"] = "speech_created"
    user_initiated: bool
    """True if the speech was created using public methods like `say` or `generate_reply`"""
    source: Literal["say", "generate_reply"]
    """Source indicating how the speech handle was created"""
    speech_handle: SpeechHandle = Field(..., exclude=True)
    """The speech handle that was created"""
    created_at: float = Field(default_factory=time.time)


# --- the tool lifecycle -----------------------------------------------------------------------

class ToolCallStarted(BaseModel):
    """A function tool call was dispatched."""

    type: Literal["tool_call_started"] = "tool_call_started"
    function_call: FunctionCall


class ToolCallUpdated(BaseModel):
    """A progress update emitted via ``ctx.update()`` while a tool call runs."""

    type: Literal["tool_call_updated"] = "tool_call_updated"
    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_update_N`` when deferred."""
    call_id: str
    message: str


class ToolCallEnded(BaseModel):
    """A tool call's single terminal entry."""

    type: Literal["tool_call_ended"] = "tool_call_ended"
    id: str
    """Entry id: ``call_id`` inline, ``{call_id}_final`` when deferred."""
    call_id: str
    message: str | None = None
    """Result or error text; None when there is nothing to voice."""
    status: Literal["done", "error", "cancelled"]


class ToolReplyUpdated(BaseModel):
    """Lifecycle of the deferred reply that voices buffered tool updates: ``scheduled``
    when queued, then ``completed`` / ``interrupted`` / ``skipped``. One reply may cover
    several calls; an inline first update never gets one.
    """

    type: Literal["tool_reply_updated"] = "tool_reply_updated"
    update_ids: list[str]
    """``ToolCallUpdated.id`` values this reply covers."""
    status: Literal["scheduled", "completed", "interrupted", "skipped"]
    speech_id: str
    """Id of the reply speech; ``speech_created`` carries its handle."""


class ToolExecutionUpdatedEvent(BaseModel):
    """One flat tool-lifecycle update. Discriminate on ``update.type``: ``tool_call_started``
    → ``tool_call_updated`` → ``tool_call_ended`` → ``tool_reply_updated``.
    """

    type: Literal["tool_execution_updated"] = "tool_execution_updated"
    update: Annotated[
        ToolCallStarted | ToolCallUpdated | ToolCallEnded | ToolReplyUpdated,
        Field(discriminator="type"),
    ]
    created_at: float = Field(default_factory=time.time)


# --- limits, errors and the close -------------------------------------------------------------

class UserTurnExceededEvent(BaseModel):
    type: Literal["user_turn_exceeded"] = "user_turn_exceeded"
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
    created_at: float = Field(default_factory=time.time)


@unique
class CloseReason(str, Enum):
    ERROR = "error"
    JOB_SHUTDOWN = "job_shutdown"
    PARTICIPANT_DISCONNECTED = "participant_disconnected"
    USER_INITIATED = "user_initiated"
    TASK_COMPLETED = "task_completed"


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
UserStateChangedEvent.__module__ = "livekit.agents.voice.events"
AgentStateChangedEvent.__module__ = "livekit.agents.voice.events"
UserInputTranscribedEvent.__module__ = "livekit.agents.voice.events"
UserTranscriptionTimeoutEvent.__module__ = "livekit.agents.voice.events"
EotPredictionEvent.__module__ = "livekit.agents.voice.events"
_AgentBackchannelOpportunityEvent.__module__ = "livekit.agents.voice.events"
AgentFalseInterruptionEvent.__module__ = "livekit.agents.voice.events"
SessionUsageUpdatedEvent.__module__ = "livekit.agents.voice.events"
SpeechCreatedEvent.__module__ = "livekit.agents.voice.events"
ToolCallStarted.__module__ = "livekit.agents.voice.events"
ToolCallUpdated.__module__ = "livekit.agents.voice.events"
ToolCallEnded.__module__ = "livekit.agents.voice.events"
ToolReplyUpdated.__module__ = "livekit.agents.voice.events"
ToolExecutionUpdatedEvent.__module__ = "livekit.agents.voice.events"
UserTurnExceededEvent.__module__ = "livekit.agents.voice.events"
CloseReason.__module__ = "livekit.agents.voice.events"
