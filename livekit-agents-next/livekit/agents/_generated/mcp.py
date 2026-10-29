# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[host(record)]`
# declarations in:
#     rust/crates/lk-agents-abi/src/llm/mcp.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing import TYPE_CHECKING, NotRequired, TypedDict

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolFlag

    from .tool_context import DuplicateMode, DuplicateScope

class ResolvedToolOptions(TypedDict):
    """Resolved per-tool behavior passed to the host's function-tool decorator."""

    flags: ToolFlag
    """Passed through to the `@function_tool` decorator, using `ToolFlag`."""
    on_duplicate: DuplicateMode
    duplicate_scope: DuplicateScope
    report_progress: bool
    """Forward the tool's MCP progress notifications to `ctx.update()`. Also selects the
    non-blocking call path (`_tool_called_nonblocking`, which takes a `RunContext`).
    """
    retries: int | None
    """The tool's `retries=`; `None` reads the run's tool budget."""


class MCPToolOptions(TypedDict):
    """Per-tool behavior for MCP tools exposed by an MCPToolset."""

    flags: NotRequired[ToolFlag]
    """Flags passed to the @function_tool decorator, using ToolFlag."""
    on_duplicate: NotRequired[DuplicateMode]
    """Behavior when a tool is called multiple times in the same context."""
    duplicate_scope: NotRequired[DuplicateScope]
    """What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too."""
    report_progress: NotRequired[bool]
    """Whether to forward the tool's progress notifications to ctx.update()."""
    retries: NotRequired[int]
    """The tool's retry budget, in place of the agent's ``retries['tools']``."""


MCPToolOptions.__module__ = "livekit.agents.llm.mcp"
