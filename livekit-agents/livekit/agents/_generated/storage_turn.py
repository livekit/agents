# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/turn/message.rs
#     rust/crates/lk-agents-storage/src/turn.rs
#     rust/crates/lk-agents-storage/src/turn/host.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .storage_run import Item, Ref, RunUsage, TaskRecord

Policy = Literal["batch", "interrupt", "queue", "concurrent", "reject", "live"]
"""A message during a run: one of the five policies of plan 3.3.4, or `live`."""


class Message(BaseModel):
    """One incoming A2A message, as the handler sees it in `ctx.messages`."""

    model_config = ConfigDict(frozen=True)
    message_id: str
    parts: list[Any]
    metadata: dict[str, Any]
    channel: str
    received_at: float
    """Unix seconds."""
    task_id: str | None
    context_id: str | None
    raw: dict[str, Any]
    """The A2A message as received."""


class TurnOptions(BaseModel):
    """`TurnRunner`'s arguments: the policy map, the debounce map, the wait, the state default. A
    field left out is the crate's default.
    """

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
    state_default: Any | None = None
    """The door's default-constructed state as JSON, for a fresh lane; `None` for a door with no
    state, which the `concurrent` policy requires.
    """


class Background(BaseModel):
    """What [`TurnHost::background`] reports."""

    running: list[str] = Field(default_factory=list)
    """The tasks whose work runs on, or whose queue waits: the holder keeps the lane for them."""
    ready: list[str] = Field(default_factory=list)
    """The tasks whose progress and results are delivered now, in one turn with one reply."""


class TurnContext(BaseModel):
    """What `A2ASessionContext` exposes today, the storage half: the message, the batch, the history,
    the state, the task, the lane's tasks, and the boundary the host polls. The host record is
    everything but the boundary, which the binding hands over beside it as a core object.
    """

    model_config = ConfigDict(frozen=True)
    thread_id: str
    """The database id of the session this turn runs in."""
    lane: str
    """The lane: the registered agent this turn belongs to."""
    message: Any
    """The batch as one user chat message, in the stored item form (`batch_to_chat_message`)."""
    batch: list[Message]
    """The individual messages with their channel and time; `batch.last()` is the one answered."""
    history: list[Item]
    """The lane's history as loaded, plus the pending input-required answer when there was one."""
    state: Any | None
    """The state as loaded, or the door's default for a fresh lane; `None` for a door with none."""
    task_id: str
    """The task that carries the reply."""
    tasks: list[TaskRecord]
    """The lane's task rows, newest first, up to 50."""
    agent_stack: list[Ref]
    """Who speaks, bottom first: empty while the door's root agent does; an `AgentTask` on top
    holds the conversation until it completes (docs/agent-tasks.md).
    """
    caller_history: Any | None
    """A delegating caller's conversation when the message carried a `chat_ctx` data part (plan
    7.5), as the part's `data`.
    """
    answer_tool: str | None
    """The tool the message is handed to before the model runs: the last turn's question came
    from it with a reference, and this message is its answer.
    """
    answer_ref: str | None
    """What `answer_tool` continues with the answer, such as the expert's task."""
    live: bool = False
    """The turn runs under `live`: a tool may go on in the background after its first update."""


class PendingCall(BaseModel):
    """A conversation tool's pending call, the one an `InputRequired` came from."""

    model_config = ConfigDict(frozen=True)
    name: str
    call_id: str
    answer_tool: str | None = None
    """The tool that takes the answer, when the question carries a reference."""
    answer_ref: str | None = None
    """What the tool that takes the answer continues with it, such as the expert's task."""


class HostEvent(BaseModel):
    """A journal event the handler recorded (docs/journal.md): `tool_execution_updated` for a
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
    new_items: list[Any] = Field(default_factory=list)
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
    agent_stack: list[Ref] | None = None
    """The agent stack after the handler, when a run moved it or a task on it changed state;
    `None` leaves the stack as loaded.
    """
    interrupted: bool = False
    """The handler yielded at a boundary."""
    background: bool = False
    """A tool of the task still runs in the background: the task stays `working`, and a turn of
    its own answers it when the tool ends.
    """
    metadata: dict[str, Any] = Field(default_factory=dict)
    """What the handler put on the task's terminal row, under short keys."""
    error: str | None = None
    """The handler raised: `"<TypeName>: <text>"`; the task fails with it."""
    events: list[HostEvent] = Field(default_factory=list)
    """Journal events the handler recorded, in order, written with the commit."""


class TurnResult(BaseModel):
    """What `Turn::run` returns: the message's task after the turn, and the reply."""

    model_config = ConfigDict(frozen=True)
    task: TaskRecord
    output: Any | None
    """The answer, as the artifact holds it: text, or the data of a typed value."""
    thread_id: str
    """The database id of the session the turn ran in."""
    answered_by: str
    """The task that carries the reply: this one, or the later one it was merged into."""
    run_id: str | None
