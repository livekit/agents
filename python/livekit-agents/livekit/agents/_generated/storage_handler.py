# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/a2a/turn.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from livekit.agents._generated.function_tools import InputRequired
from livekit.agents._generated.run_result import RunCompleted, RunInputRequired, RunNoOutput
from livekit.agents._generated.tool_outcomes import Failed, Rejected, Working
from livekit.agents.llm.tool_context import Completed

if TYPE_CHECKING:
    from livekit.agents._generated.run_result import _Settled as _HostHandlerRun


@dataclass(kw_only=True, slots=True)
class HandlerCompleted:
    """A `Completed` transition's value, as JSON, and whether it reaches the person as written."""

    value: object
    verbatim: bool


HandlerRun: TypeAlias = "_HostHandlerRun"


HandlerOutcome: TypeAlias = (
    RunCompleted
    | RunInputRequired
    | RunNoOutput
    | Completed
    | InputRequired
    | Failed
    | Rejected
    | Working
    | object
)
"""What an `a2a_session` handler returned, told apart by its class;
a value is read as its JSON (`to_jsonable_python`).
"""
