# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/turn.rs

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .a2a_client import Message
    from .storage_class import ItemRow, RunUsage, TaskRecord
    from .storage_context import ContextScope, ContextScopeWrite


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
    name: str
    call_id: str
    answer_tool: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class HostEvent:
    type: str
    payload: object


@dataclass(frozen=True, kw_only=True, slots=True)
class HandlerOutput:
    output: object | None
    new_items: list[object]
    state: object | None
    scope_writes: list[ContextScopeWrite]
    usage: RunUsage
    input_required: str | None
    pending_call: PendingCall | None
    interrupted: bool
    metadata: dict[str, object]
    error: str | None
    events: list[HostEvent]


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
