# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/schema.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
