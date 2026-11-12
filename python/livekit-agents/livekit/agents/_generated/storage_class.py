# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/storage/mod.rs
#     rust/crates/lk-agents-abi/src/storage/rows.rs
#     rust/crates/lk-agents-abi/src/storage/thread.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from .chat_items import ChatItem


@dataclass(kw_only=True, slots=True)
class LocalStorageOptions:
    """One local project's thread directory; path resolution stays in native storage."""

    directory: str | None
    project_id: str


@dataclass(kw_only=True, slots=True)
class ThreadInfo:
    """A thread: times are unix seconds, `0` for never, TTLs whole seconds."""

    thread_id: str
    scope: str
    """The subject the thread is about, the one threads are listed and granted by."""
    attributes: dict[str, str]
    """A small string map the application sets and reads back."""
    created_at: int
    last_active: int
    expires_at: int
    idle_ttl_seconds: int


@dataclass(kw_only=True, slots=True)
class ThreadPage:
    """One page of `list_threads`, newest first."""

    threads: list[ThreadInfo]
    before: str
    """The next page's cursor; empty at the end."""


@dataclass(kw_only=True, slots=True)
class ThreadItem:
    """One chat item of a thread."""

    agent: str
    seq: int
    """Position in the thread's history."""
    message_id: str
    item: ChatItem


@dataclass(kw_only=True, slots=True)
class ThreadItemPage:
    """One page of `get_thread_items`, oldest first."""

    items: list[ThreadItem]
    after: str
    """The next page's cursor; empty at the end."""


@dataclass(kw_only=True, slots=True)
class ThreadTask:
    """One A2A task of a thread."""

    task_id: str
    thread_id: str
    name: str | None
    """The application's task label."""
    status: TaskState
    updated_at: int


TaskState: TypeAlias = Literal["submitted", "working", "input-required", "completed", "failed", "canceled", "rejected"]
"""A task's state: the storage core's `TaskState` as the host reads it."""


@dataclass(kw_only=True, slots=True)
class Ask:
    """What a task waiting on the person holds: the question and the call that takes the answer."""

    question: str
    tool: str | None = None
    """The tool that asked."""
    call_id: str | None = None
    """The asking call, whose output the answer becomes."""
    answer_tool: str | None = None
    """The tool that takes the answer before the model runs."""


@dataclass(kw_only=True, slots=True)
class TaskInternal:
    """A task's pending question and handler settlement."""

    ask: Ask | None
    answered: bool
    """A turn answered the task `working` (its handler's `Working` outcome): no run holds it."""


@dataclass(kw_only=True, slots=True)
class ItemRow:
    """One history item of a partition."""

    id: str
    message_id: str | None
    """The A2A `messageId` of the message this item came from, the idempotency key."""
    payload: ChatItem


@dataclass(kw_only=True, slots=True)
class TaskData:
    """The A2A `Task` fields a task keeps, as v1 protobuf JSON."""

    history: list[object]
    artifacts: list[object] | None
    metadata: dict[str, object]
    status_message: object | None
    """`Task.status.message`."""


@dataclass(kw_only=True, slots=True)
class TaskRecord:
    """One A2A task of a partition: `id = task_id`, `contextId` the thread id, `status = (status,
    data.status_message, updated_at_ms)`, plus the fields of `data`.
    """

    task_id: str
    partition_key: str
    pos: int
    """Submission order; the partition runs its runnable tasks in it."""
    name: str | None
    """An optional application task label."""
    message_id: str | None
    """Optional initiating A2A message id, used for retry deduplication.
    None for tasks initiated without an incoming message.
    """
    status: TaskState
    updated_at_ms: int
    """Unix milliseconds."""
    data: TaskData
    internal: TaskInternal | None
    """What the framework keeps to resume the task: its pending question and handler settlement."""


@dataclass(kw_only=True, slots=True)
class TaskPage:
    """One page of A2A `tasks/list`."""

    tasks: list[TaskRecord]
    next_page_token: str
    """Empty on the last page."""
    page_size: int
    total_size: int
    """Every page's tasks, in all."""


@dataclass(kw_only=True, slots=True)
class PartitionRow:
    """One partition: a registered agent's state, lease and inbox cursor."""

    partition_key: str
    version: int
    """The fencing token: a run loaded at `v` commits only `WHERE version = v`."""
    state: object | None
    """The application's typed state, one snapshot per run."""
    lease_holder: str | None
    lease_expires_at_ms: int
    """Unix milliseconds on agent-db's clock; `0` is released."""
    drained_id: int
    """The id of the last inbox row a drain consumed."""


@dataclass(kw_only=True, slots=True)
class EventRow:
    """One journal entry: every `AgentSessionEvent`, plus `run_started` and `run_ended`."""

    id: int
    occurred_at_ms: int
    partition_key: str
    run_id: str
    type: str
    payload: object


@dataclass(kw_only=True, slots=True)
class InboxRow:
    """One message received while the partition was running, under the `inbox` policy."""

    id: int
    partition_key: str
    task_id: str
    """The task created for the message; also its answer stream."""
    message_id: str
    message: object
    """The A2A `Message`."""
    received_at_ms: int


RunOutcome: TypeAlias = Literal["completed", "input-required", "failed", "canceled", "rejected"]
"""What `run_ended` records."""


TaskOrder: TypeAlias = Literal["updated", "inserted"]
"""How a task listing is ordered (docs/internals/storage.md#reading-thread-items-and-tasks)."""
