# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/mcp.rs

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from livekit.agents.llm.tool_context import ToolFlag

    from .tool_context import DuplicateMode, DuplicateScope


# Record declarations:


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
    timeout: float | None
    """This tool's request timeout in seconds, in place of the server's `request_timeout`."""


class MCPToolOptions(TypedDict, total=False):
    """Per-tool behavior for MCP tools exposed by an MCPToolset."""

    flags: ToolFlag
    """Flags passed to the @function_tool decorator, using ToolFlag."""
    on_duplicate: DuplicateMode
    """Behavior when a tool is called multiple times in the same context."""
    duplicate_scope: DuplicateScope
    """What counts as a duplicate, e.g. ``"name_and_args"`` to compare arguments too."""
    report_progress: bool
    """Whether to forward the tool's progress notifications to ctx.update()."""
    retries: int
    """The tool's retry budget, in place of the agent's ``retries['tools']``."""
    timeout: float
    """Seconds one call may wait without a reply or progress, in place of the server's
    ``request_timeout``.
    """


MCPToolOptions.__module__ = "livekit.agents.llm.mcp"


MCP_IDLE_CLOSE_SECONDS: Final = 60.0
"""Seconds a connection stays open after its toolset's last run ends, for the next run to reuse."""
