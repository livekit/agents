# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/text_session.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from livekit.agents.llm import LLM
    from livekit.agents.types import APIConnectOptions
    from livekit.agents.voice.agent_host import AgentHost

    from .._core import ChatItem
    from .agent_run import AgentDefinition, RunError, RunResult, RunUsage, StreamEvent


@dataclass(frozen=True, kw_only=True, slots=True)
class FrameId:
    """A stable backend conversation frame, distinct from a call or invocation."""

    value: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ParentCall:
    frame_id: FrameId
    call_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ChildCallStatusApproval:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class ChildCallStatusReady:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class ChildCallStatusRunning:
    frame_id: FrameId


ChildCallStatus: TypeAlias = ChildCallStatusApproval | ChildCallStatusReady | ChildCallStatusRunning


@dataclass(frozen=True, kw_only=True, slots=True)
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


@dataclass(frozen=True, kw_only=True, slots=True)
class SessionFrame:
    id: FrameId
    agent_name: str
    history: list[ChatItem]
    parent: ParentCall | None
    context_scope: str
    completed: bool
    pending_calls: list[PendingChildCall]
    answer_tool: str | None
    visible_dialogue: list[dict[str, object]]
    """Delivery receipts, in conversation order, excluding the initial tool task and fork."""
    starting_history_ids: frozenset[str]
    transferred_message_ids: frozenset[str]
    merge_chat_history: bool
    """Frozen at creation: reconfiguration on a later request cannot change the return policy."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionState:
    session_id: str
    execution_id: str
    root_frame_id: FrameId
    active_frame_id: FrameId
    ancestors: list[FrameId]
    """Root first, immediate parent last. Completed frames never occur here."""
    frames: dict[FrameId, SessionFrame]
    next_id: int

    def to_json(self) -> str:
        """The saved form: restore it with ``from_json``."""
        from .. import _core

        return _core._core_persist_text_session_state(self)

    @classmethod
    def from_json(cls, data: str) -> TextSessionState:
        """The record ``to_json`` saved."""
        from .. import _core

        return _core._core_restore_text_session_state(data)


ChatHistory: TypeAlias = Literal["fresh", "fork"]


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalAgentTool:
    agent_name: str
    chat_history: ChatHistory
    merge_chat_history: bool
    context_scope: str | None
    """None shares the parent's scope. A supplied reference is application-owned."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorMaxDepth:
    max_depth: int


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorUnknownAgent:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorUnknownScope:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorUnknownTool:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorMissingOutputType:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorInvalidState:
    message: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingErrorMissingAcceptedOutput:
    pass


RoutingError: TypeAlias = (
    RoutingErrorMaxDepth
    | RoutingErrorUnknownAgent
    | RoutingErrorUnknownScope
    | RoutingErrorUnknownTool
    | RoutingErrorMissingOutputType
    | RoutingErrorInvalidState
    | RoutingErrorMissingAcceptedOutput
)


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionTurn:
    result: RunResult
    state: TextSessionState


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionErrorRouting:
    error: RoutingError


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionErrorRun:
    error: RunError


TextSessionError: TypeAlias = TextSessionErrorRouting | TextSessionErrorRun


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionAgent:
    """Shared definitions and live resources; no host resource enters the saved state."""

    definition: AgentDefinition
    llm: LLM
    host: AgentHost
    output_type_explicit: bool
    agent_tools: dict[str, LocalAgentTool]
    host_context: object
    """A callback context for this agent, captured at session entry."""


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionRequest:
    session_id: str
    execution_id: str
    root_scope: str
    contexts: dict[str, object]
    message: ChatItem
    state: TextSessionState | None
    delegated: bool
    conn_options: APIConnectOptions


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionSettled:
    turn: TextSessionTurn
    output: object | None


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateStream:
    sequence: int
    frame_id: FrameId
    agent_name: str
    invocation_id: str
    event: StreamEvent


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateInvocationSettled:
    invocation_id: str
    usage: RunUsage


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateParked:
    invocation_id: str
    usage: RunUsage


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateIncompleteCleanup:
    pending: list[str]


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateFinished:
    settled: TextSessionSettled


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionUpdateFailed:
    error: TextSessionError


TextSessionUpdate: TypeAlias = (
    TextSessionUpdateStream
    | TextSessionUpdateInvocationSettled
    | TextSessionUpdateParked
    | TextSessionUpdateIncompleteCleanup
    | TextSessionUpdateFinished
    | TextSessionUpdateFailed
)


@dataclass(frozen=True, kw_only=True, slots=True)
class TextSessionDelivery:
    invocation_id: str
    messages: dict[str, str]
