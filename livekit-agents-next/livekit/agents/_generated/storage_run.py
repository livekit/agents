# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-storage/src/run.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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
