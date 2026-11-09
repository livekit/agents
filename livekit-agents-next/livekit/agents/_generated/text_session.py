# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/text_session.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from livekit.agents.llm import LLM
    from livekit.agents.voice.agent_run import AgentCallbacks

    from .agent_run import AgentDefinition, RunFinished, RunInputRequirement, StreamEvent
    from .chat_items import ChatItem, ChatMessage
    from .connect import APIConnectOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class FrameId:
    """A stable backend conversation frame, distinct from a call or invocation."""

    value: str


@dataclass(kw_only=True, slots=True)
class ParentCall:
    frame_id: FrameId
    call_id: str


@dataclass(kw_only=True, slots=True)
class ChildCallStatusApproval:
    pass


@dataclass(kw_only=True, slots=True)
class ChildCallStatusReady:
    pass


@dataclass(kw_only=True, slots=True)
class ChildCallStatusRunning:
    frame_id: FrameId


ChildCallStatus: TypeAlias = ChildCallStatusApproval | ChildCallStatusReady | ChildCallStatusRunning


@dataclass(kw_only=True, slots=True)
class PendingChildCall:
    parent: ParentCall
    invocation_id: str
    execution_id: str
    tool_name: str
    agent_name: str
    validated_arguments: object
    identity: str
    """The shared approval gate's canonical validated identity, never the raw model arguments."""
    status: ChildCallStatus


@dataclass(kw_only=True, slots=True)
class SessionFrame:
    id: FrameId
    agent_name: str
    history: list[ChatItem]
    parent: ParentCall | None
    completed: bool
    pending_calls: list[PendingChildCall]
    answer_tool: str | None
    visible_dialogue: list[ChatMessage]
    """Delivery receipts, in conversation order, excluding the initial tool task and fork."""
    starting_history_ids: frozenset[str]
    transferred_message_ids: frozenset[str]
    merge_chat_history: bool
    """Frozen at creation: reconfiguration on a later request cannot change the return policy."""


@dataclass(kw_only=True, slots=True)
class TextSessionState:
    session_id: str
    execution_id: str
    root_frame_id: FrameId
    active_frame_id: FrameId
    ancestors: list[FrameId]
    """Root first, immediate parent last. Completed frames never occur here."""
    frames: dict[FrameId, SessionFrame]
    next_id: int

    def _to_json(self) -> str:
        """The saved form: restore it with ``_from_json``."""
        from .. import _core

        return _core._core_persist_text_session_state(self)

    @classmethod
    def _from_json(cls, data: str) -> TextSessionState:
        """The record ``_to_json`` saved."""
        from .. import _core

        return _core._core_restore_text_session_state(data)


ChatHistory: TypeAlias = Literal["fresh", "fork"]


@dataclass(kw_only=True, slots=True)
class LocalAgentTool:
    agent_name: str
    chat_history: ChatHistory
    merge_chat_history: bool


@dataclass(kw_only=True, slots=True)
class TextSessionAgent:
    """Shared definitions and live resources; no host resource enters the saved state."""

    definition: AgentDefinition
    llm: LLM
    host: AgentCallbacks
    output_type_explicit: bool
    agent_tools: dict[str, LocalAgentTool]
    host_context: object
    """A callback context for this agent, captured at session entry."""
    tools_host: object | None = None
    """Canonical tool pool for this agent, shared across its invocations."""


@dataclass(kw_only=True, slots=True)
class TextSessionRequest:
    session_id: str
    execution_id: str
    task_id: str | None = None
    message: ChatItem
    state: TextSessionState | None
    delegated: bool
    conn_options: APIConnectOptions


@dataclass(kw_only=True, slots=True)
class TextSessionFinished:
    """A settled turn as the host reads it: the active frame's run and the state to save."""

    run: RunFinished
    state: TextSessionState


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateStream:
    sequence: int
    frame_id: FrameId
    agent_name: str
    invocation_id: str
    event: StreamEvent


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateInvocationSettled:
    invocation_id: str


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateParked:
    invocation_id: str


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateIncompleteCleanup:
    pending: list[str]


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateFinished:
    settled: TextSessionFinished


@dataclass(kw_only=True, slots=True)
class TextSessionUpdateFailed:
    error: BaseException
    input_requirements: list[RunInputRequirement] = field(default_factory=list)
    provider_error: BaseException | None = None
    """The original model failure, retained beside its correlated outcome metadata."""


TextSessionUpdate: TypeAlias = (
    TextSessionUpdateStream
    | TextSessionUpdateInvocationSettled
    | TextSessionUpdateParked
    | TextSessionUpdateIncompleteCleanup
    | TextSessionUpdateFinished
    | TextSessionUpdateFailed
)


@dataclass(kw_only=True, slots=True)
class TextSessionDelivery:
    invocation_id: str
    messages: dict[str, str]


TextSessionState.__module__ = "livekit.agents.text_session"
