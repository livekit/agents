# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `#[derive(Host)]` declarations in:
#     rust/crates/lk-agents-abi/src/llm/mcp.rs
#
# Edit the Rust declaration and re-run `cargo run -p lk-agents-codegen -- --write`. CI regenerates and fails on a
# non-empty diff, so a hand edit here is reverted rather than merged.

from __future__ import annotations

from typing_extensions import TypedDict

from ..llm.tool_context import DuplicateMode, DuplicateScope, ToolFlag


class MCPToolOptions(TypedDict, total=False):
    """Per-tool behavior for MCP tools exposed by an :class:`MCPToolset`."""

    flags: ToolFlag
    """Flags passed to the @function_tool decorator, e.g. ToolFlag.CANCELLABLE."""
    on_duplicate: DuplicateMode
    """Behavior when a tool is called multiple times in the same context."""
    duplicate_scope: DuplicateScope
    """What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too."""
    report_progress: bool
    """Whether to forward the tool's progress notifications to ctx.update()."""


# --- the frozen public paths -----------------------------------------------------------------
#
# Assigned here rather than in each class body: pydantic resolves annotations against
# `sys.modules[cls.__module__]` as it builds the class, so setting it early sends it
# looking for `Literal` and every foreign type in the re-export module, which imports
# none of them. What this fixes is `repr`, `pickle` and the equivalence gate all
# reporting `_generated.*` for a name whose import path never moved.
MCPToolOptions.__module__ = "livekit.agents.llm.mcp"
