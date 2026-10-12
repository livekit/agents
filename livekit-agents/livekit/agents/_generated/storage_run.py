# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/run/records.rs
#     rust/crates/lk-agents-storage/src/run.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .storage import TaskData, TaskInternal

TaskState = Literal[
    "submitted", "working", "input-required", "completed", "failed", "canceled", "rejected"
]
"""The A2A task states (plan 5.5); the last four are terminal and immutable."""


RunOutcome = Literal["completed", "input-required", "failed", "canceled", "rejected"]
"""What `run_ended` records (plan 5.1)."""


class Ref(BaseModel):
    """One rebuild rule for everything durable that names code (plan 5.2, rule 9 of 5.3): `name` is an
    `Agent`'s name or an `AgentTask` class name; `args` the constructor arguments, JSON-shaped,
    `None` for a plain `Agent`. On the agent stack a task entry also carries `state`, the task's
    own state as it waits, and `call_id`, the call of the agent below that awaits it. Stored as
    `{"name": .., "args": .., "state": .., "call_id": ..}`.
    """

    model_config = ConfigDict(frozen=True)
    name: str
    args: Any | None = None
    state: Any | None = None
    call_id: str | None = None


class TaskRecord(BaseModel):
    """One task row as the framework and a dashboard read it (plan 5.5)."""

    model_config = ConfigDict(frozen=True)
    id: str
    lane: str
    seq: int
    name: str | None
    """The `AgentTask` class; `None` for a turn."""
    message_id: str | None
    status: str
    """A `TaskState`, as the row spells it."""
    status_message: Any | None
    updated_at: int
    data: TaskData
    """The A2A `Task` fields: `history`, `artifacts`, `metadata`."""
    internal: TaskInternal | None
    """The `AgentTask`'s arguments and pending question."""


class RunSummary(BaseModel):
    """One run as the journal tells it: `run_started` and, when it ended, `run_ended`."""

    run_id: str
    lane: str
    task_id: str | None
    channel: str | None
    started_at_ms: int
    ended_at_ms: int | None
    outcome: str | None
    reason: str | None


class LaneInfo(BaseModel):
    """One lane row, read from outside a run."""

    model_config = ConfigDict(frozen=True)
    name: str
    version: int
    agent_stack: list[Ref]
    lease_holder: str | None
    lease_until: int
    waiter: str | None
    drained_id: int


class Item(BaseModel):
    """One history item in memory: the chat item's JSON (`item_to_payload`, the form `chat_item`
    converts to and from the stored protobuf JSON), with the `id` the row carries. The payload's `type` is the item
    kind: `message`, `function_call`, `function_call_output`, `agent_handoff`.
    """

    model_config = ConfigDict(frozen=True)
    id: str
    payload: Any


class RunUsage(BaseModel):
    """Token usage of one run (`voice.run_result.RunUsage`); written as a `usage` event when the run
    made a request or a tool call.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
