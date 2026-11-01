# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/provider_format/tools.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

    from ..llm.tool_context import FunctionTool, RawFunctionTool, ReadBack
    from .tool_context import DuplicateMode, DuplicateScope
    from .tool_flags import ToolFlag


@dataclass(frozen=True)
class InputRequired:
    """The turn or task waits on the person for an answer to ``question``.

        The agent facing the person asks it in its own words, or exactly as written with
        ``verbatim=True``. From a conversation tool the next message is written as this call's
        output. ``tool`` names the tool that takes the answer.
    """

    question: str
    tool: FunctionTool | RawFunctionTool | None = None
    verbatim: bool = False


@dataclass
class FunctionToolInfo:
    name: str
    description: str | None
    flags: ToolFlag
    on_duplicate: DuplicateMode = "allow"
    duplicate_scope: DuplicateScope = "name"
    needs_approval: bool = False
    read_back: ReadBack | None = None
    """``needs_approval=`` as a function: the read-back the person approves."""
    retries: int | None = None
    """``retries=``: this tool's retry budget in place of the agent's ``retries['tools']``."""


@dataclass
class RawFunctionToolInfo:
    name: str
    raw_schema: dict[str, Any]
    flags: ToolFlag
    on_duplicate: DuplicateMode = "allow"
    duplicate_scope: DuplicateScope = "name"
    retries: int | None = None


InputRequired.__module__ = "livekit.agents.llm.tool_context"
FunctionToolInfo.__module__ = "livekit.agents.llm.tool_context"
RawFunctionToolInfo.__module__ = "livekit.agents.llm.tool_context"
