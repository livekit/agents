from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..log import logger
from ..voice.tool_executor import ToolHandlingOptions
from .tool_context import DuplicateMode, Tool, Toolset

if TYPE_CHECKING:
    from ..voice.events import RunContext as AsyncRunContext  # noqa: F401

# AsyncRunContext is a deprecated alias for RunContext kept for one release so user
# tools typing ``ctx: AsyncRunContext`` keep working. __getattr__ surfaces a runtime
# warning to nudge migration to ``RunContext``.
__all__ = ["AsyncRunContext", "AsyncToolset"]


def __getattr__(name: str) -> Any:
    if name == "AsyncRunContext":
        from ..voice.events import RunContext

        logger.warning(
            "AsyncRunContext is deprecated; import RunContext from livekit.agents directly"
        )
        return RunContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class AsyncToolset(Toolset):
    """Session-scoped toolset whose tools survive agent handoff.

    Background updates from tools in this toolset are delivered to whichever agent
    is current at delivery time, so a ``ctx.update()`` started under agent A still
    completes after a handoff to agent B. Tools placed on ``Agent(tools=...)``
    instead use the activity-scoped executor and are cancelled/awaited on handoff.

    Example::

        @function_tool(on_duplicate="confirm", flags=ToolFlag.CANCELLABLE)
        async def book_flight(ctx, origin: str, destination: str) -> dict:
            await ctx.update(f"Looking up flights {origin} → {destination}...")
            flights = await search(origin, destination)
            await ctx.update(f"Found {len(flights)}, picking the best...")
            return await book_best(flights)

        session = LiveSession(tools=[AsyncToolset(id="booking", tools=[book_flight])])
    """

    def __init__(
        self,
        *,
        id: str,
        tools: Sequence[Tool] | None = None,
        tool_handling: ToolHandlingOptions | None = None,
        # deprecated
        on_duplicate_call: DuplicateMode | None = None,
    ) -> None:
        if on_duplicate_call is not None:
            raise TypeError(
                "AsyncToolset(on_duplicate_call=...) has been deprecated; "
                "set `on_duplicate=...` on each @function_tool instead."
            )

        super().__init__(id=id, tools=tools)
        self._async_tool_options_override = (
            tool_handling.get("async_options") if tool_handling is not None else None
        )
