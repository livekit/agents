from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from ..log import logger
from ..utils.misc import is_given
from ..voice.events import RunContext as AsyncRunContext
from ..voice.tool_executor import AsyncToolPrompts, _resolve_async_tool_prompts, _ToolExecutor
from .tool_context import FunctionTool, RawFunctionTool, Tool, Toolset

if TYPE_CHECKING:
    from ..voice.agent_activity import AgentActivity
    from ..voice.agent_session import AgentSession

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
        async_tool_prompts: AsyncToolPrompts | None = None,
        on_duplicate_call: DuplicateMode | None = None,
    ) -> None:
        super().__init__(id=id, tools=tools)

        if on_duplicate_call is not None:
            logger.warning(
                "AsyncToolset(on_duplicate_call=...) is deprecated; set on_duplicate "
                "on @function_tool per tool instead."
            )

            for child in self._iter_function_tools():
                if child.info.on_duplicate != on_duplicate_call:
                    logger.warning(
                        "overwriting on_duplicate=%s on tool %s with %s",
                        child.info.on_duplicate,
                        child.info.name,
                        on_duplicate_call,
                    )
                    child.info.on_duplicate = on_duplicate_call

        self._tool_prompts_override: AsyncToolPrompts | None = async_tool_prompts
        self._executor = _ToolExecutor(owning_activity=None)

    def _iter_function_tools(self) -> list[FunctionTool | RawFunctionTool]:
        out: list[FunctionTool | RawFunctionTool] = []

        def walk(tools: Sequence[Tool | Toolset]) -> None:
            for tool in tools:
                if isinstance(tool, (FunctionTool, RawFunctionTool)):
                    out.append(tool)
                elif isinstance(tool, Toolset):
                    walk(tool.tools)

        walk(self.tools)
        return out

    def _attach_activity(self, *, activity: AgentActivity | None, session: AgentSession) -> None:
        """Attach this toolset to an activity scope.

        When ``activity`` is ``None`` the toolset is session-scoped: tool
        replies survive agent handoff and are delivered to whichever agent
        is current. Otherwise the toolset is agent-scoped and replies are
        delivered to that activity's agent.
        """

        self._executor.set_owning_activity(activity)

        if self._tool_prompts_override is not None:
            resolved = _resolve_async_tool_prompts(self._tool_prompts_override)
        elif activity is not None and is_given(activity._agent._async_tool_prompts):
            resolved = _resolve_async_tool_prompts(activity._agent._async_tool_prompts)
        else:
            resolved = session._async_tool_prompts
        self._executor.set_tool_prompts(resolved)

    async def aclose(self) -> None:
        await super().aclose()
        await self._executor.aclose()
