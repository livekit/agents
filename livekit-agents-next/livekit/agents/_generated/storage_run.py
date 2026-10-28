# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/schema.rs
#     rust/crates/lk-agents-storage/src/db.rs
#     rust/crates/lk-agents-storage/src/run.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .storage import Ask, TaskData


class ItemRow(BaseModel):
    """One history item of a lane."""

    model_config = ConfigDict(frozen=True)
    id: str
    message_id: str | None = None
    """The A2A `messageId` of the message this item came from, the idempotency key."""
    payload: dict[str, Any]
    """The `ChatContext.ChatItem`, as protobuf JSON."""


class TaskRow(BaseModel):
    """One A2A task of a lane. The A2A `Task` is
    `id = task_id`, `contextId = the thread id`, `status = (status, data.statusMessage, updated_at)`,
    plus the fields of `data`.
    """

    model_config = ConfigDict(frozen=True)
    task_id: str
    lane: str
    seq: int
    """Submission order; the lane runs its runnable tasks in it."""
    name: str | None
    """An optional application task label."""
    message_id: str | None
    status: TaskState
    updated_at: int
    """Unix milliseconds."""
    data: TaskData
    """The A2A `Task` fields: `history`, `artifacts`, `metadata`."""
    internal: Ask | None
    """What the framework keeps to resume the task: the question it waits on the person to
    answer, while it does.
    """


class LaneRow(BaseModel):
    """One lane: a registered agent's state, lease and inbox cursor."""

    model_config = ConfigDict(frozen=True)
    lane: str
    version: int
    """The fencing token: a run loaded at `v` commits only `WHERE version = v`."""
    state: Any | None = None
    """The application's typed state, one snapshot per run."""
    lease_holder: str | None
    lease_until: int
    """Unix seconds on agent-db's clock; `0` is released."""
    waiter: str | None
    """A worker that failed the lease under `interrupt` and is polling for it."""
    drained_id: int
    """The id of the last inbox row a drain consumed."""


class TaskPage(BaseModel):
    """One page of A2A `tasks/list`."""

    model_config = ConfigDict(frozen=True)
    tasks: list[TaskRow]
    next_page_token: str
    """Empty on the last page."""
    page_size: int
    total_size: int
    """Every page's tasks, in all."""


class RunUsage(BaseModel):
    """Token usage of one run; written as a `usage` event when the run
    made a request or a tool call.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0


TaskState = Literal[
    "submitted", "working", "input-required", "completed", "failed", "canceled", "rejected"
]
"""The framework's task states; the terminal ones are immutable. The A2A edge maps them to its
own.
"""


RunOutcome = Literal["completed", "input-required", "failed", "canceled", "rejected"]
"""What `run_ended` records."""


class RunSummary(BaseModel):
    """One run as the journal tells it: `run_started` and, when it ended, `run_ended`."""

    run_id: str
    lane: str
    task_id: str | None
    channel: str | None
    started_at_ms: int
    ended_at_ms: int | None
    outcome: RunOutcome | None
    reason: str | None
