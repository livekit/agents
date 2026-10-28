# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-a2a/src/lib.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .storage_run import ItemRow, RunUsage, TaskRow


class TurnOptions(BaseModel):
    """A text door's turn options; a field left out is the default."""

    policy: Policy | dict[str, Policy] | None = None
    """`inbox`: one policy for every channel, or a map by channel over the
    default map; `None` is the default map.
    """
    debounce: float | dict[str, float] | None = None
    """The wait before a turn starts, in seconds: one for every channel, or a map by channel;
    `None` is the defaults.
    """
    wait_timeout: float | None = None
    """How long a waiter polls for the lane before its task fails `lease-timeout`, in seconds;
    `None` is a minute.
    """


Policy = Literal["batch", "interrupt", "queue", "concurrent", "reject"]
"""A message during a run: one of the five policies."""


class Message(BaseModel):
    """One incoming A2A message, as the handler sees it in `ctx.messages`."""

    model_config = ConfigDict(frozen=True)
    message_id: str
    parts: list[dict[str, Any]]
    """v1 parts, as protobuf JSON."""
    metadata: dict[str, Any]
    channel: str
    received_at: float
    """Unix seconds."""
    task_id: str | None
    context_id: str | None
    raw: dict[str, Any]
    """The A2A message as received, less a delegation's `chat_ctx` part."""


Verdict = Literal["continue", "yield"]
"""The turn's answer to a [`Checkpoint`]."""


class TurnContext(BaseModel):
    """What the handler is opened on: the message, the batch, the history, the state, the task."""

    model_config = ConfigDict(frozen=True)
    thread_id: str
    """The database id of the session this turn runs in."""
    lane: str
    """The lane: the registered agent this turn belongs to."""
    message: dict[str, Any]
    """The batch as one user chat message (`batch_to_chat_message`)."""
    batch: list[Message]
    """The individual messages with their channel and time; `batch.last()` is the one answered."""
    history: list[ItemRow]
    """The lane's history as loaded, plus the pending input-required answer when there was one."""
    state: Any | None
    """The state as loaded; `None` for a fresh lane."""
    read_only: bool = False
    """Context field writes require the lane's exclusive lease."""
    task_id: str
    """The task that carries the reply."""
    tasks: list[TaskRow]
    """The lane's task rows, newest first, up to 50."""
    caller_history: Any | None
    """A delegating caller's conversation when the message carried a `chat_ctx` data part, as the
    part's `data`.
    """
    answer_tool: str | None
    """The tool the message is handed to before the model runs: an `InputRequired(tool=)` of
    the last turn named it, and this message is its answer.
    """


class PendingCall(BaseModel):
    """A conversation tool's pending call, the one an `InputRequired` came from."""

    model_config = ConfigDict(frozen=True)
    name: str
    call_id: str
    answer_tool: str | None = None
    """The tool that takes the answer, when the `InputRequired` named one."""


class HostEvent(BaseModel):
    """A journal event the handler recorded (docs/internals/storage.md#journal): `tool_execution_updated` for a
    call's start, its progress and its end.
    """

    model_config = ConfigDict(frozen=True)
    type: str
    payload: Any


class HandlerOutput(BaseModel):
    """What the turn reads from the handler's results: the `RunResult` fields and the context's
    metadata, as JSON. Every field has a default: a handler reports only what it has.
    """

    output: Any | None = None
    """The answer: text, or a typed value as JSON."""
    new_items: list[dict[str, Any]] = Field(default_factory=list)
    """The chat items the runs produced, in order, each with its `id` and `type`; empty when no
    run was made, in which case the turn stores the message and a text answer itself.
    """
    state: Any | None = None
    """The state after the handler; written at commit when it differs from what was loaded."""
    usage: RunUsage = Field(default_factory=RunUsage)
    input_required: str | None = None
    """A conversation tool's `InputRequired(question)`."""
    pending_call: PendingCall | None = None
    """The call that asked, when `input_required` is set."""
    interrupted: bool = False
    """The handler yielded at a checkpoint."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """What the handler put on the task's terminal row, under short keys."""
    error: str | None = None
    """The handler raised: `"<TypeName>: <text>"`; the task fails with it."""
    events: list[HostEvent] = Field(default_factory=list)
    """Journal events the handler recorded, in order, written with the commit."""


class TurnResult(BaseModel):
    """What `Turn::run` returns: the message's task after the turn, and the reply."""

    model_config = ConfigDict(frozen=True)
    task: TaskRow
    output: Any | None
    """The answer, as the artifact holds it: text, or the data of a typed value."""
    thread_id: str
    """The database id of the session the turn ran in."""
    answered_by: str
    """The task that carries the reply: this one, or the later one it was merged into."""
    run_id: str | None
