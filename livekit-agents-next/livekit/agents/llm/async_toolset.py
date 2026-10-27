from __future__ import annotations

from collections.abc import Sequence

from ..voice.tool_executor import ToolHandlingOptions
from .tool_context import Tool, Toolset

__all__ = ["AsyncToolset"]


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

        session = VoiceSession(tools=[AsyncToolset(id="booking", tools=[book_flight])])
    """

    def __init__(
        self,
        *,
        id: str,
        tools: Sequence[Tool] | None = None,
        tool_handling: ToolHandlingOptions | None = None,
    ) -> None:
        super().__init__(id=id, tools=tools)
        self._async_tool_options_override = (
            tool_handling.get("async_options") if tool_handling is not None else None
        )
