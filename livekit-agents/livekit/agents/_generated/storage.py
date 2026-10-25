# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/schema.rs
#     rust/crates/lk-agents-storage/src/streams.rs
#     rust/crates/lk-agents-py/src/storage.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """The `data` column: the A2A `Task` fields a row keeps, as v1 protobuf JSON; an absent field is
    left out.
    """

    history: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    status_message: dict[str, Any] = None
    """`Task.status.message`."""


class Ask(BaseModel):
    """What a task waiting on the person holds: the question and the call that takes the answer."""

    question: str
    tool: str | None = None
    """The tool that asked."""
    call_id: str | None = None
    """The asking call, whose output the answer becomes."""
    answer_tool: str | None = None
    """The tool that takes the answer before the model runs."""


class EventRow(BaseModel):
    """One journal entry: every `AgentSessionEvent`, plus `run_started` and `run_ended`, tagged with
    the lane. Append-only, owned by the framework.
    """

    model_config = ConfigDict(frozen=True)
    id: int
    ts_ms: int
    lane: str
    run_id: str
    type: str
    payload: Any


class InboxRow(BaseModel):
    """One message received while the lane was running, under the `inbox` policy."""

    model_config = ConfigDict(frozen=True)
    id: int
    lane: str
    task_id: str
    """The task created for the message; also its answer stream."""
    message_id: str
    message: dict[str, Any]
    received_at: int


class StreamInfo(BaseModel):
    """A created stream: its id (also the task id it answers) and when it expires."""

    model_config = ConfigDict(frozen=True)
    stream_id: str
    expires_at: int


class DatabaseInfo(BaseModel):
    """A database made through `DatabaseService.CreateDatabase`."""

    model_config = ConfigDict(frozen=True)
    database_id: str
    """`DB_..`."""
    expires_at: int
    """Unix seconds; `0` for never."""
