from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..voice.events import RunContext as AsyncRunContext
from .tool_context import Tool, Toolset

if TYPE_CHECKING:
    from ..voice.tool_executor import _ToolExecutor

# Backwards-compatible alias. Tools that type their context as ``AsyncRunContext``
# keep working unchanged — the unified ``RunContext`` carries the same surface.
__all__ = ["AsyncRunContext", "AsyncToolset"]


class AsyncToolset(Toolset):
    """Session-scoped toolset whose tools survive agent handoff.

    Tools added here run through a session-scoped :class:`_ToolExecutor`, so a
    background ``ctx.update()`` started under agent A still gets its reply
    delivered after a handoff to agent B. Tools placed directly on
    ``Agent(tools=...)`` use the activity-scoped executor instead and are
    cancelled/awaited on handoff (depending on ``allow_cancellation``).

    Example::

        @function_tool(on_duplicate="confirm", allow_cancellation=True)
        async def book_flight(ctx, origin: str, destination: str) -> dict:
            await ctx.update(f"Looking up flights {origin} → {destination}...")
            flights = await search(origin, destination)
            await ctx.update(f"Found {len(flights)}, picking the best...")
            return await book_best(flights)

        session = AgentSession(tools=[AsyncToolset(id="booking", tools=[book_flight])])
    """

    # Deprecated; kept for backwards type-import compatibility.
    DuplicateMode = Literal["allow", "replace", "reject", "confirm"]

    def __init__(
        self,
        *,
        id: str,
        tools: Sequence[Tool] | None = None,
        on_duplicate_call: DuplicateMode | None = None,
    ) -> None:
        super().__init__(id=id, tools=tools)
        if on_duplicate_call is not None:
            warnings.warn(
                "AsyncToolset(on_duplicate_call=...) is deprecated; set on_duplicate "
                "on @function_tool per tool instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # session-scoped executor — owning_activity is None so reply targeting
        # follows the session's current agent at delivery time.
        from ..voice.tool_executor import _ToolExecutor

        self._executor: _ToolExecutor = _ToolExecutor(owning_activity=None)

    async def aclose(self) -> None:
        await super().aclose()
        await self._executor.aclose()
