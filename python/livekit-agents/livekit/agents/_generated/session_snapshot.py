# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/session/snapshot.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from .chat_items import ChatItem, ChatMessage
    from .delegation_remote import RemoteAgentState
    from .session import AgentInputRequest


@dataclass(frozen=True, kw_only=True, slots=True)
class FrameId:
    """A stable backend conversation frame, distinct from a call or invocation."""

    value: str


@dataclass(kw_only=True, slots=True)
class ParentCall:
    frame_id: FrameId
    call_id: str
    """Canonical call item ID; the output retains the provider's separate wire call_id."""


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
class FrameOptions:
    """A child's fixed policy. Reconfiguring its tool after a restart cannot change this choice."""

    merge_chat_ctx: bool


@dataclass(kw_only=True, slots=True)
class SessionFrame:
    id: FrameId
    agent_name: str
    history: list[ChatItem]
    parent: ParentCall | None
    completed: bool
    output: object | None = None
    error: str | None = None
    input_requests: dict[str, AgentInputRequest] = field(default_factory=dict)
    pending_calls: list[PendingChildCall]
    visible_dialogue: list[ChatMessage]
    """Delivery receipts, in conversation order, excluding the initial tool task and fork."""
    starting_history_ids: frozenset[str]
    options: FrameOptions


@dataclass(kw_only=True, slots=True)
class _SessionSnapshot:
    session_id: str
    execution_id: str
    root_frame_id: FrameId
    active_frame_id: FrameId
    ancestors: list[FrameId]
    """Root first, immediate parent last. Completed frames never occur here."""
    frames: dict[FrameId, SessionFrame]
    next_id: int
    chat_history: list[ChatMessage] = field(default_factory=list)
    """Accepted human input and delivered assistant content across every local agent."""
    remote_agents: dict[str, RemoteAgentState] = field(default_factory=dict)
    """Remote communication spans local agent changes and execution modes."""
