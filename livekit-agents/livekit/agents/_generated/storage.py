# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/schema.rs
#     rust/crates/lk-agents-storage/src/statements.rs
#     rust/crates/lk-agents-storage/src/streams.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatItemRow(BaseModel):
    """One history item of a lane: what the model sees, owned by the developer, editable."""

    model_config = ConfigDict(frozen=True)
    id: str
    """The upsert key. `pos`, the rowid, is the platform's: its page token's sort key."""
    lane: str
    seq: int
    """Assigned by the commit, per lane."""
    message_id: str | None
    """The A2A `messageId` of the message this item came from, the idempotency key."""
    payload: Any
    """The chat item in its in-memory form (`chat_item`); the column holds its protobuf JSON."""


class TaskRow(BaseModel):
    """One A2A task of a lane (plan 5.5): a turn or a submitted `AgentTask`. The A2A `Task` is
    `id = task_id`, `contextId = the thread id`, `status = (status, status_message, updated_at)`,
    plus the fields of `data`.
    """

    model_config = ConfigDict(frozen=True)
    task_id: str
    """The upsert key. `pos`, the rowid, is the platform's: its page token's sort key."""
    lane: str
    seq: int
    """Submission order; the lane runs its runnable tasks in it."""
    name: str | None
    """The `AgentTask` class; `None` for a turn."""
    message_id: str | None
    status: str
    """`submitted`, `working`, `input-required`, `completed`, `failed`, `canceled`, `rejected`."""
    status_message: Any | None
    """`Task.status.message`."""
    updated_at: int
    data: Any
    """The A2A `Task` fields under their A2A names: `history`, `artifacts`, `metadata`."""
    internal: Any | None
    """The framework's own: the `AgentTask` arguments, the pending `ask`, `deadline`, `parent`."""


class TaskData(BaseModel):
    """The `data` column: the A2A `Task` fields a row keeps, under their A2A names; an absent field is
    left out.
    """

    history: list[Any] = Field(default_factory=list)
    """`Task.history`: A2A messages."""
    artifacts: list[Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskInternal(BaseModel):
    """The `internal` column: what the framework keeps to resume an `AgentTask`; an absent field is left
    out.
    """

    args: Any | None = None
    """The `AgentTask` constructor arguments."""
    ask: Any | None = None
    """The pending transition: the tool that takes the answer."""
    deadline: int | None = None
    parent: str | None = None
    cancel: Any | None = None
    """A client's `tasks/cancel` on the running task, with its reason: its holder stops it."""


class LaneRow(BaseModel):
    """One lane: a registered agent's history, state and agent stack, with its lease (plan 3.3.2) and
    its inbox cursor (3.3.4).
    """

    model_config = ConfigDict(frozen=True)
    lane: str
    version: int
    """The fencing token: a run loaded at `v` commits only `WHERE version = v`."""
    state: Any | None
    """The application's typed state, one snapshot per run."""
    agent_stack: Any | None
    """A list of `Ref {name, args}`: who is speaking."""
    lease_holder: str | None
    lease_token: str | None
    lease_until: int
    """Unix seconds on agent-db's clock; `0` is released."""
    waiter: str | None
    """A worker that failed the lease under `interrupt` and is polling for it."""
    drained_id: int
    """The id of the last inbox row a drain consumed."""


class EventRow(BaseModel):
    """One journal entry: every `AgentSessionEvent`, plus `run_started` and `run_ended`, tagged with
    the lane. Append-only, owned by the framework.
    """

    model_config = ConfigDict(frozen=True)
    id: int
    """Insertion order."""
    ts_ms: int
    lane: str
    run_id: str
    type: str
    payload: Any


class InboxRow(BaseModel):
    """One message received while the lane was running, under the `inbox` policy."""

    model_config = ConfigDict(frozen=True)
    id: int
    """Arrival order."""
    lane: str
    task_id: str
    """The task created for the message; also its answer stream."""
    message_id: str
    message: Any
    """The A2A message."""
    received_at: int


class AcquireRequest(BaseModel):
    """The acquire and the load (plan 3.3.2), one batch: the lane under `holder`'s lease, with the
    `run_started` event written only when the lease was taken.
    """

    lane: str
    holder: str
    """The worker, for re-entry (`lease_holder = ?`)."""
    lease_token: str
    run_id: str
    ts_ms: int
    task_id: str = ""
    """The run's task: a `submitted` row of it orders the acquire behind older waiters; empty
    for a run with no row (a voice turn).
    """
    run_started: Any
    """The `run_started` event's payload: the task id, the stream id, the channel."""
    events_after: int = 0
    """Load the lane's events after this id (0: all)."""


class Loaded(BaseModel):
    """What an acquire loaded: the lane at `version`."""

    model_config = ConfigDict(frozen=True)
    version: int
    already_committed: bool
    """Plan 5.2: this `run_id` already committed; the caller releases without running."""
    lane: LaneRow
    items: list[ChatItemRow]
    events: list[EventRow]
    tip: int


class ItemWrite(BaseModel):
    """One history item a commit writes: an upsert on `id`."""

    id: str
    message_id: str | None = None
    payload: Any


class EventWrite(BaseModel):
    """One journal entry a commit appends; `run_ended` is one of them, written last."""

    ts_ms: int
    type: str
    payload: Any


class TaskWrite(BaseModel):
    """One task row a commit upserts: every column but `lane`, `seq` and `updated_at`, which the
    statement fills.
    """

    task_id: str
    status: str
    message_id: str | None = None
    name: str | None = None
    """The `AgentTask` class; `None` for a turn."""
    status_message: Any | None = None
    data: TaskData
    internal: TaskInternal | None = None


LeaseAfter = Literal["release", "renew"]
"""Release (the default) or renew when the runner continues with pending inbox rows or due tasks."""


class Fence(BaseModel):
    """What a fenced commit is conditioned on: the lane at the version the run loaded, under the
    lease token it holds.
    """

    version: int
    lease_token: str


class CommitRequest(BaseModel):
    """The commit (plan 5.2): every write of a run as one batch, fenced on the version loaded."""

    lane: str
    run_id: str
    fence: Fence | None = None
    """`None` is the `concurrent` policy: no lease, no version bump, appends only."""
    items: list[ItemWrite] = Field(default_factory=list)
    delete_items: list[str] = Field(default_factory=list)
    events: list[EventWrite] = Field(default_factory=list)
    tasks: list[TaskWrite] = Field(default_factory=list)
    state: Any | None = None
    """`None` keeps the stored snapshot."""
    agent_stack: Any | None = None
    drained_id: int | None = None
    run_ended: EventWrite
    """The `run_ended` event, written last."""
    after: LeaseAfter = "release"


class CommitResult(BaseModel):
    """What a commit wrote: the lane's version after it and the connection's durable tip."""

    model_config = ConfigDict(frozen=True)
    version: int
    """The version after the bump; the loaded one under `concurrent`."""
    tip: int


class RunEndedRequest(BaseModel):
    """Leaving by exception: the `run_ended` event with the outcome, and the release."""

    lane: str
    run_id: str
    lease_token: str
    ts_ms: int
    payload: Any


class ReceiveRequest(BaseModel):
    """The inbox receive (plan 3.3.4): the row, the task it creates, and the lease attempt, one
    transaction.
    """

    lane: str
    holder: str
    lease_token: str
    task_id: str
    message_id: str
    message: Any
    name: str | None = None
    """The `AgentTask` class; `None` for a turn."""
    data: TaskData = Field(default_factory=TaskData)
    internal: TaskInternal | None = None


class SubmittedTask(BaseModel):
    """The early `submitted` row of [`ThreadDb::insert_submitted_task`]: the columns the inbox
    receive fills, minus the inbox row and the lease.
    """

    lane: str
    task_id: str
    message_id: str | None = None
    name: str | None = None
    """The `AgentTask` class; `None` for a turn."""
    data: TaskData = Field(default_factory=TaskData)
    internal: TaskInternal | None = None


class Received(BaseModel):
    """The receive inserted the row: its id, and the version when this worker took the lease and
    is the runner.
    """

    model_config = ConfigDict(frozen=True)
    outcome: Literal["received"] = "received"
    id: int
    acquired: int | None


class Duplicate(BaseModel):
    """The `messageId` was in the inbox already: nothing inserted; the task it belongs to."""

    model_config = ConfigDict(frozen=True)
    outcome: Literal["duplicate"] = "duplicate"
    task_id: str | None


Receive = Annotated[
    Received | Duplicate,
    Field(discriminator="outcome"),
]


class DrainCheck(BaseModel):
    """The drain check: how many inbox rows are pending, and whether the lease was released."""

    model_config = ConfigDict(frozen=True)
    pending: int
    released: bool


class StreamInfo(BaseModel):
    """A created stream: its id (also the task id it answers) and when it expires."""

    model_config = ConfigDict(frozen=True)
    stream_id: str
    expires_at: int
