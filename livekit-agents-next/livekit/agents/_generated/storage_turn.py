# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/turn.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .a2a_client import Message
    from .storage_class import ItemRow, RunUsage, TaskRecord, TaskState
    from .storage_context import AppStatement, ContextScope, ContextScopeWrite


@dataclass(frozen=True, kw_only=True, slots=True)
class TurnContext:
    thread_id: str
    lane: str
    message: dict[str, object]
    batch: list[Message]
    history: list[ItemRow]
    state: object | None
    context_scope_id: str
    context_scopes: list[ContextScope]
    read_only: bool
    task_id: str
    tasks: list[TaskRecord]
    caller_history: dict[str, object] | None
    answer_tool: str | None


@dataclass(frozen=True, kw_only=True, slots=True)
class PendingCall:
    """A conversation tool's pending call, the one an `InputRequired` came from."""

    name: str
    call_id: str
    answer_tool: str | None = None
    """The tool that takes the answer, when the `InputRequired` named one."""


@dataclass(frozen=True, kw_only=True, slots=True)
class HostEvent:
    """A journal event the handler recorded (docs/internals/storage.md#journal):
    `tool_execution_updated` for a call's start, its progress and its end.
    """

    type: str
    payload: object


@dataclass(frozen=True, kw_only=True, slots=True)
class HandlerOutput:
    output: object | None
    new_items: list[object]
    state: object | None
    scope_writes: list[ContextScopeWrite]
    app_writes: list[AppStatement]
    """What the handler's managed transactions staged, applied in the turn's commit."""
    usage: RunUsage
    input_required: str | None
    pending_call: PendingCall | None
    interrupted: bool
    metadata: dict[str, object]
    error: str | None
    events: list[HostEvent]
    status: TaskState | None = None
    """A `Rejected` or `Working` transition's state (docs/server.md#handler-outcomes); `output`
    is then its reason or status message.
    """


@dataclass(frozen=True, kw_only=True, slots=True)
class TurnResult:
    task: TaskRecord
    output: object | None
    thread_id: str
    answered_by: str
    run_id: str | None


Verdict: TypeAlias = Literal["continue", "yield"]


@dataclass(frozen=True, kw_only=True, slots=True)
class Checkpoint:
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class Output:
    output: HandlerOutput


@dataclass(frozen=True, kw_only=True, slots=True)
class Progress:
    task_id: str
    text: str


@dataclass(frozen=True, kw_only=True, slots=True)
class Token:
    task_id: str
    delta: str


TurnEvent: TypeAlias = Checkpoint | Output | Progress | Token
