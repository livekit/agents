"""Unit tests for _AsyncToolManager and the dispatch helpers it enables.

These tests intentionally avoid spinning up a full AgentSession: the manager is
designed to be usable from any tool dispatch site, and the happy paths here exercise
spawn → execute → result without touching session/agent state.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from livekit.agents.llm import FunctionCall, function_tool
from livekit.agents.llm.async_toolset import (
    AsyncRunContext,
    AsyncToolset,
    _AsyncToolManager,
    _has_async_context_param,
)
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.speech_handle import SpeechHandle


def _make_run_ctx(*, call_id: str, name: str = "noop", arguments: str = "{}") -> RunContext:
    """Build a minimal RunContext sufficient for _AsyncToolManager.spawn happy paths.

    No real AgentSession is attached — tests must not call ctx.update() (which would
    enqueue a reply against session.current_agent)."""
    return RunContext(
        session=cast(AgentSession, None),
        speech_handle=SpeechHandle.create(),
        function_call=FunctionCall(call_id=call_id, name=name, arguments=arguments),
    )


@function_tool
async def _sync_tool(x: int) -> int:
    """A normal tool — no AsyncRunContext in its signature."""
    return x * 2


@function_tool
async def _async_echo(ctx: AsyncRunContext, value: str) -> str:
    """Async tool that returns immediately — does not call ctx.update()."""
    return value


class TestHasAsyncContextParam:
    def test_detects_async_run_context(self) -> None:
        assert _has_async_context_param(_async_echo) is True

    def test_skips_plain_run_context(self) -> None:
        assert _has_async_context_param(_sync_tool) is False


class TestAsyncToolsetStillExposesRuntimeTools:
    """AsyncToolset must keep surfacing get_running_tasks / cancel_task in its tool
    list (always-on, by design — see the conversation around this refactor). The
    activity's manager adds them only when its own manager has tasks; AsyncToolset
    keeps the legacy always-on behavior."""

    def test_runtime_tools_present(self) -> None:
        ts = AsyncToolset(id="t", tools=[_async_echo])
        names = {t.info.name for t in ts.tools if hasattr(t, "info")}
        assert "get_running_tasks" in names
        assert "cancel_task" in names


class TestManagerLifecycle:
    async def test_has_running_tasks_toggles(self) -> None:
        manager = _AsyncToolManager(on_duplicate_call="allow")
        assert manager.has_running_tasks is False

        gate = asyncio.Event()

        @function_tool
        async def blocking(ctx: AsyncRunContext) -> str:
            await ctx.update("started")  # marks pending_fut, lets spawn() return
            await gate.wait()
            return "done"

        # spawn returns once the tool calls ctx.update() — task continues in background.
        # NOTE: ctx.update() normally enqueues a reply against session; we patch _enqueue_reply
        # to a no-op for this isolated test.
        async def _noop_enqueue(*args: Any, **kwargs: Any) -> None:
            return None

        manager._enqueue_reply = _noop_enqueue  # type: ignore[method-assign]

        result = await manager.spawn(
            tool=blocking,
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
            raw_arguments={},
        )
        assert "started" in result
        assert manager.has_running_tasks is True

        gate.set()
        # let the background task finish and the done_callback run
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert manager.has_running_tasks is False

    async def test_spawn_returns_tool_value_when_no_update(self) -> None:
        """If a tool returns without ever calling ctx.update(), spawn() resolves to
        the tool's return value directly."""
        manager = _AsyncToolManager(on_duplicate_call="allow")
        result = await manager.spawn(
            tool=_async_echo,
            run_ctx=_make_run_ctx(call_id="call-1", name="_async_echo"),
            raw_arguments={"value": "hello"},
        )
        assert result == "hello"
        # spawn() returns once _pending_fut resolves; the done_callback that
        # removes the task from the registry fires on the next loop iteration.
        await asyncio.sleep(0)
        assert manager.has_running_tasks is False


class TestManagerDuplicate:
    async def test_reject_blocks_second_call(self) -> None:
        manager = _AsyncToolManager(on_duplicate_call="reject")

        gate = asyncio.Event()

        @function_tool
        async def blocking(ctx: AsyncRunContext) -> str:
            await ctx.update("running")
            await gate.wait()
            return "done"

        async def _noop_enqueue(*args: Any, **kwargs: Any) -> None:
            return None

        manager._enqueue_reply = _noop_enqueue  # type: ignore[method-assign]

        first = await manager.spawn(
            tool=blocking,
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
            raw_arguments={},
        )
        assert "running" in first
        assert manager.has_running_tasks is True

        second = await manager.spawn(
            tool=blocking,
            run_ctx=_make_run_ctx(call_id="call-2", name="blocking"),
            raw_arguments={},
        )
        # reject returns the DUPLICATE_REJECT message; no second task spawned.
        assert isinstance(second, str)
        assert "already running" in second
        assert "cancel_task" in second
        # only the first task is live
        assert len(manager._running_tasks) == 1

        gate.set()
        await manager.aclose()


class TestManagerCancel:
    async def test_cancel_running_task(self) -> None:
        manager = _AsyncToolManager(on_duplicate_call="allow")

        gate = asyncio.Event()

        @function_tool
        async def blocking(ctx: AsyncRunContext) -> str:
            await ctx.update("running")
            await gate.wait()
            return "done"

        async def _noop_enqueue(*args: Any, **kwargs: Any) -> None:
            return None

        manager._enqueue_reply = _noop_enqueue  # type: ignore[method-assign]

        await manager.spawn(
            tool=blocking,
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
            raw_arguments={},
        )
        assert manager.has_running_tasks is True

        cancelled = await manager.cancel("call-1")
        assert cancelled is True
        # done_callback should have removed the task from the registry
        assert manager.has_running_tasks is False

    async def test_cancel_unknown_call_id(self) -> None:
        manager = _AsyncToolManager(on_duplicate_call="allow")
        assert await manager.cancel("does-not-exist") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
