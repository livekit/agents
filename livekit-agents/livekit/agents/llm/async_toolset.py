from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..log import logger
from ..utils.misc import is_given
from ..voice.tool_executor import (
    ToolHandlingOptions,
    _resolve_async_tool_options,
    _ToolExecutor,
)
from .tool_context import DuplicateMode, Tool, Toolset

if TYPE_CHECKING:
    from ..voice.agent_activity import AgentActivity
    from ..voice.agent_session import AgentSession
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

        session = AgentSession(tools=[AsyncToolset(id="booking", tools=[book_flight])])
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
        self._executor = _ToolExecutor(owning_activity=None)

    def _attach_activity(self, *, activity: AgentActivity | None, session: AgentSession) -> None:
        """Bind this toolset to a scope. ``activity=None`` makes it session-scoped
        (replies survive handoff); otherwise replies stay with ``activity``'s agent."""
        self._executor.set_owning_activity(activity)

        if self._async_tool_options_override is not None:
            resolved = _resolve_async_tool_options(self._async_tool_options_override)
        elif activity is not None and is_given(activity._agent._async_tool_options):
            resolved = _resolve_async_tool_options(activity._agent._async_tool_options)
        else:
            resolved = session._async_tool_options
        self._executor.set_tool_options(resolved)

    async def aclose(self) -> None:
        await super().aclose()
        await self._executor.drain()
        await self._executor.aclose()
