# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/delegation/remote.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from inspect import get_annotations
from typing import TYPE_CHECKING, Literal, TypeAlias

from a2a.types.a2a_pb2 import (
    CancelTaskRequest,
    GetTaskRequest,
    ListTasksRequest,
    ListTasksResponse,
    SendMessageRequest,
    SendMessageResponse,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
)

from .session import InputRequestState

if TYPE_CHECKING:
    from a2a.types.a2a_pb2 import Message
    from pydantic import JsonValue

    from .session import ToolCallRef
    from .session_input import InputRequestRef
    from .types import ErrorKind


@dataclass(kw_only=True, slots=True)
class RemoteFailureRpc:
    error: dict[str, JsonValue]


@dataclass(kw_only=True, slots=True)
class RemoteFailureTransport:
    kind: ErrorKind
    message: str
    status_code: int | None


@dataclass(kw_only=True, slots=True)
class RemoteFailureProtocol:
    message: str


RemoteFailure: TypeAlias = RemoteFailureRpc | RemoteFailureTransport | RemoteFailureProtocol
"""Persisted failure evidence, without credentials, Python exceptions or retry decisions."""


A2AJournalKind: TypeAlias = Literal["request_recorded", "response_received", "request_error"]


@dataclass(kw_only=True, slots=True)
class A2AJournalEntry:
    sequence: int
    """Local observation order; neither remote causal order nor a transport RPC ID."""
    request_id: str
    """Stable logical operation identity shared by its request, responses and errors."""
    kind: A2AJournalKind
    payload: A2AJournalPayload
    """Original typed SDK content. Dispatch markers never append duplicate payloads."""


@dataclass(frozen=True, kw_only=True, slots=True)
class InputRequest:
    state: InputRequestState
    """Response handling for this occurrence, separate from the remote task status."""
    source_message_id: str | None
    """Optional question in the canonical observed-message index."""
    response_message_id: str | None
    """Reserved outgoing A2A message, retained unchanged while its delivery is uncertain."""


@dataclass(kw_only=True, slots=True)
class RemoteTaskRef:
    context_id: str
    task_id: str


@dataclass(kw_only=True, slots=True)
class RemoteRequestState:
    method: str
    """SDK method constant, fixed when the request is prepared."""
    request_event: int
    """Journal sequence of the immutable SDK request; not a second payload copy."""
    dispatch_started: bool
    """Persisted before I/O: an attempt may have happened, not proof of admission."""
    ended: bool
    """The local observation stream settled; this does not complete remote business work.
    Preparing the unchanged request may reopen it only before dispatch started.
    """
    task: RemoteTaskRef | None
    """A known remote task for reconciliation after a lost connection."""
    origin: ToolCallRef | None
    """Local tool call that initiated this operation, not relationship-wide ownership."""
    input_request: InputRequestRef | None
    """A selected input occurrence, if this operation sends its response."""


@dataclass(kw_only=True, slots=True)
class RemoteTaskState:
    task: Task
    """Latest protocol observation, including protocol states not understood by presentation."""
    input_requests: dict[str, InputRequest]
    cancel_requested: bool
    """Explicit intent for this task only; closing a frontend never sets it."""
    next_input: int
    """Monotonic local occurrence identity, independent of question wording or stream order."""


@dataclass(kw_only=True, slots=True)
class RemoteAgentState:
    endpoint: str
    """Pinned destination. Restoration cannot silently redirect existing work."""
    events: list[A2AJournalEntry]
    requests: dict[str, RemoteRequestState]
    """Mutable operation facts; the journal contains communication only."""
    tasks: dict[str, RemoteTaskState]
    messages: dict[str, Message]
    """Canonical observed messages in either direction, indexed by messageId."""
    default_context_id: str | None
    """Reused for new work; observing a different task context does not switch it."""


A2ARequest: TypeAlias = (
    SendMessageRequest
    | GetTaskRequest
    | ListTasksRequest
    | CancelTaskRequest
    | SubscribeToTaskRequest
)


A2AResponse: TypeAlias = SendMessageResponse | StreamResponse | Task | ListTasksResponse


A2AJournalPayload: TypeAlias = A2ARequest | A2AResponse | RemoteFailure


InputRequest.__annotations__ = get_annotations(InputRequest, eval_str=True)
InputRequest.__module__ = "livekit.agents.storage.session_state"
