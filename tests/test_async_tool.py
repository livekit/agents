"""Tests for async tools (tools whose signature includes :class:`AsyncRunContext`).

Two layers:

* **Unit tests** for :class:`_AsyncToolManager` and the dispatch helpers. These don't
  spin up a real :class:`AgentSession` — they call ``manager.spawn`` directly and
  assert on its observable behavior (running-task registry, duplicate handling,
  cancel, mock substitution).

* **End-to-end tests** that drive a real :class:`AgentSession` with a scripted
  :class:`FakeLLM`. These confirm the full path: LLM tool call → activity's
  manager → optional mock → function output events on the :class:`RunResult`.
  Modeled after ``examples/drive-thru/test_agent.py`` but with FakeLLM so the
  tests are deterministic and don't need API access.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

import pytest

from livekit.agents import Agent, AgentSession, AsyncRunContext
from livekit.agents.llm import (
    FunctionCall,
    FunctionTool,
    FunctionToolCall,
    RawFunctionTool,
    ToolError,
    function_tool,
)
from livekit.agents.llm.async_toolset import (
    REPLY_INSTRUCTIONS,
    UPDATE_TEMPLATE,
    AsyncToolset,
    _AsyncToolManager,
    _has_async_context_param,
    _is_async_toolset_wrapper,
)
from livekit.agents.llm.utils import prepare_function_arguments
from livekit.agents.voice.agent_activity import ActivityClosedError
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.run_result import _run_mock, mock_tools
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_llm import FakeLLM, FakeLLMResponse

# ---------------------------------------------------------------------------
# unit-test helpers
# ---------------------------------------------------------------------------


def _make_run_ctx(*, call_id: str, name: str = "noop", arguments: str = "{}") -> RunContext:
    """Build a minimal RunContext sufficient for _AsyncToolManager.spawn happy paths.

    No real AgentSession is attached — tests must not call ctx.update() (which would
    enqueue a reply against session.current_agent)."""
    return RunContext(
        session=cast(AgentSession, None),
        speech_handle=SpeechHandle.create(),
        function_call=FunctionCall(call_id=call_id, name=name, arguments=arguments),
    )


def _make_call(
    tool: FunctionTool | RawFunctionTool,
    raw_arguments: dict[str, Any] | None = None,
    *,
    mock: Callable | None = None,
) -> Callable[[AsyncRunContext], Awaitable[Any]]:
    """Build the body closure callers hand to ``manager.spawn`` — mirrors what
    generation.py and AsyncToolset._wrap_tool construct in real dispatch."""
    raw = raw_arguments or {}

    async def _call(async_ctx: AsyncRunContext) -> Any:
        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=tool, json_arguments=raw, call_ctx=async_ctx
        )
        if mock is not None:
            return await _run_mock(mock, *fnc_args, **fnc_kwargs)
        return await tool(*fnc_args, **fnc_kwargs)

    return _call


@function_tool
async def _sync_tool(x: int) -> int:
    """A normal tool — no AsyncRunContext in its signature."""
    return x * 2


@function_tool
async def _async_echo(ctx: AsyncRunContext, value: str) -> str:
    """Async tool that returns immediately — does not call ctx.update()."""
    return value


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------


class TestHasAsyncContextParam:
    def test_detects_async_run_context(self) -> None:
        assert _has_async_context_param(_async_echo) is True

    def test_skips_plain_run_context(self) -> None:
        assert _has_async_context_param(_sync_tool) is False


class TestAsyncToolsetStillExposesRuntimeTools:
    """AsyncToolset must keep surfacing get_running_tasks / cancel_task in its tool
    list (always-on, by design). The activity's manager adds them only when its own
    manager has tasks; AsyncToolset keeps the legacy always-on behavior."""

    def test_runtime_tools_present(self) -> None:
        ts = AsyncToolset(id="t", tools=[_async_echo])
        names = {t.info.name for t in ts.tools if hasattr(t, "info")}
        assert "get_running_tasks" in names
        assert "cancel_task" in names


class TestAsyncToolsetWrapperMarker:
    """AsyncToolset's wrapper must carry the _lk_async_wrapper marker so that
    generation.py::_execute_tools_task defers mock handling to the wrapper (which
    routes mocks through the toolset's manager — full async lifecycle). Without
    the marker, dispatch would short-circuit and run the mock inline."""

    def test_wrapped_async_tool_has_marker(self) -> None:
        ts = AsyncToolset(id="t", tools=[_async_echo])
        wrapped = next(
            t for t in ts.tools if getattr(t, "info", None) and t.info.name == "_async_echo"
        )
        assert _is_async_toolset_wrapper(wrapped) is True

    def test_non_async_tool_is_not_wrapped(self) -> None:
        """Tools without AsyncRunContext are passed through unchanged — no wrapper,
        no marker."""
        ts = AsyncToolset(id="t", tools=[_sync_tool])
        passthrough = next(
            t for t in ts.tools if getattr(t, "info", None) and t.info.name == "_sync_tool"
        )
        # _sync_tool is the same module-level instance; helper returns False
        assert _is_async_toolset_wrapper(passthrough) is False


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

        # spawn returns once the tool calls ctx.update(). ctx.update() normally
        # enqueues a reply against session.current_agent; patch it to a no-op
        # here since there's no real session.
        async def _noop_enqueue(*args: Any, **kwargs: Any) -> None:
            return None

        manager._enqueue_reply = _noop_enqueue  # type: ignore[method-assign]

        result = await manager.spawn(
            function_callable=_make_call(blocking),
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
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
            function_callable=_make_call(_async_echo, {"value": "hello"}),
            run_ctx=_make_run_ctx(call_id="call-1", name="_async_echo"),
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
            function_callable=_make_call(blocking),
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
        )
        assert "running" in first
        assert manager.has_running_tasks is True

        second = await manager.spawn(
            function_callable=_make_call(blocking),
            run_ctx=_make_run_ctx(call_id="call-2", name="blocking"),
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
            function_callable=_make_call(blocking),
            run_ctx=_make_run_ctx(call_id="call-1", name="blocking"),
        )
        assert manager.has_running_tasks is True

        cancelled = await manager.cancel("call-1")
        assert cancelled is True
        # done_callback should have removed the task from the registry
        assert manager.has_running_tasks is False

    async def test_cancel_unknown_call_id(self) -> None:
        manager = _AsyncToolManager(on_duplicate_call="allow")
        assert await manager.cancel("does-not-exist") is False


class _RecordingAgent:
    def __init__(self) -> None:
        from livekit.agents.llm.chat_context import ChatContext

        self._chat_ctx = ChatContext.empty()
        self.update_calls: list[Any] = []

    @property
    def chat_ctx(self) -> Any:
        return self._chat_ctx

    async def update_chat_ctx(self, chat_ctx: Any) -> None:
        self.update_calls.append(chat_ctx)
        self._chat_ctx = chat_ctx


class _StubActivity:
    """Stub for AgentActivity used as ``owning_activity`` on _AsyncToolManager."""

    def __init__(self) -> None:
        self.agent = _RecordingAgent()
        self.gate = asyncio.Event()
        self.closed = False

    async def wait_for_idle(self) -> None:
        await self.gate.wait()
        if self.closed:
            raise ActivityClosedError("simulated close")


class _RecordingSession:
    """Stub session capturing generate_reply calls for _deliver_reply tests."""

    def __init__(self, agent: _RecordingAgent) -> None:
        from livekit.agents.llm.chat_context import ChatContext

        self.agent = agent
        self.generate_reply_calls: list[dict] = []
        self.history = ChatContext.empty()
        self._global_run_state: Any = None

    @property
    def current_agent(self) -> _RecordingAgent:
        return self.agent

    def generate_reply(self, **kwargs: Any) -> None:
        self.generate_reply_calls.append(kwargs)


class TestDeferredDelivery:
    """Chat_ctx insert is eager so mid-flight replies see context; generate_reply is gated on idle."""

    async def test_chat_ctx_eager_generate_reply_gated_on_idle(self) -> None:
        activity = _StubActivity()
        manager = _AsyncToolManager(on_duplicate_call="allow", owning_activity=activity)  # type: ignore[arg-type]

        session = _RecordingSession(activity.agent)

        @function_tool
        async def quick(ctx: AsyncRunContext, value: str) -> str:
            # ctx.update marks pending_fut so the post-update path triggers
            # _enqueue_reply with the final return value.
            await ctx.update("starting")
            return value

        ctx = RunContext(
            session=cast(AgentSession, session),
            speech_handle=SpeechHandle.create(),
            function_call=FunctionCall(call_id="c1", name="quick", arguments='{"value": "x"}'),
        )
        await manager.spawn(
            function_callable=_make_call(quick, {"value": "x"}),
            run_ctx=ctx,
        )
        for _ in range(5):
            await asyncio.sleep(0)

        # eager insert: chat_ctx was written immediately at enqueue time so
        # any reply before the gated generate_reply can see the message.
        assert len(activity.agent.update_calls) == 1, (
            "chat_ctx should be eagerly written at enqueue time"
        )
        # generate_reply must still wait for the activity to be idle.
        assert session.generate_reply_calls == []

        activity.gate.set()
        for _ in range(5):
            await asyncio.sleep(0)

        # delivery target == eager target → no duplicate insert
        assert len(activity.agent.update_calls) == 1
        assert len(session.generate_reply_calls) == 1
        await manager.aclose()

    async def test_activity_closed_drops_pending_reply(self) -> None:
        activity = _StubActivity()
        manager = _AsyncToolManager(on_duplicate_call="allow", owning_activity=activity)  # type: ignore[arg-type]

        session = _RecordingSession(activity.agent)

        @function_tool
        async def quick(ctx: AsyncRunContext, value: str) -> str:
            await ctx.update("starting")
            return value

        ctx = RunContext(
            session=cast(AgentSession, session),
            speech_handle=SpeechHandle.create(),
            function_call=FunctionCall(call_id="c1", name="quick", arguments='{"value": "x"}'),
        )
        await manager.spawn(
            function_callable=_make_call(quick, {"value": "x"}),
            run_ctx=ctx,
        )
        for _ in range(5):
            await asyncio.sleep(0)

        activity.closed = True
        activity.gate.set()
        for _ in range(5):
            await asyncio.sleep(0)

        # eager insert happened at enqueue, before close was detected. the
        # drop only kills the generate_reply.
        assert len(activity.agent.update_calls) == 1
        assert session.generate_reply_calls == []
        await manager.aclose()


class TestMockThroughManager:
    """When a mock is supplied for an async tool, it runs through the manager's
    lifecycle just like the real tool would — including AsyncRunContext, ctx.update(),
    duplicate detection, cancel. This is the test surface dispatch sites use for
    bare async tools (see generation.py::_execute_tools_task)."""

    async def test_zero_arg_lambda_mock_replaces_real_tool(self) -> None:
        """A lambda mock with no params still runs through manager.spawn; the
        trim logic discards all kwargs and the lambda's return becomes the result."""
        manager = _AsyncToolManager(on_duplicate_call="allow")

        sentinel = object()
        result = await manager.spawn(
            function_callable=_make_call(
                _async_echo,
                {"value": "ignored"},
                mock=lambda: sentinel,
            ),
            run_ctx=_make_run_ctx(call_id="call-1", name="_async_echo"),
        )
        assert result is sentinel
        await asyncio.sleep(0)
        assert manager.has_running_tasks is False

    async def test_mock_can_exercise_async_lifecycle(self) -> None:
        """A mock that declares ctx: AsyncRunContext gets the live AsyncRunContext
        from the manager and can call ctx.update() — the same async paths the real
        tool exercises."""
        manager = _AsyncToolManager(on_duplicate_call="allow")

        updates: list[str] = []

        async def _noop_enqueue(ctx: AsyncRunContext, items: Any) -> None:
            updates.append("enqueue called")

        manager._enqueue_reply = _noop_enqueue  # type: ignore[method-assign]

        async def mock_search(ctx: AsyncRunContext, value: str) -> str:
            await ctx.update(f"mocked update for {value}")
            return "mocked result"

        # spawn returns once the mock's ctx.update() fires _pending_fut.
        result = await manager.spawn(
            function_callable=_make_call(
                _async_echo,
                {"value": "weather"},
                mock=mock_search,
            ),
            run_ctx=_make_run_ctx(call_id="call-1", name="_async_echo"),
        )
        assert "mocked update for weather" in result
        # the background task should be running until it returns the final result
        assert manager.has_running_tasks is True
        # let the mock's return enqueue the final reply and the task finish
        for _ in range(3):
            await asyncio.sleep(0)
        assert manager.has_running_tasks is False
        # one update from ctx.update() before pending_fut, one from the final return
        assert updates == ["enqueue called"]

    async def test_mock_exception_propagates(self) -> None:
        """A mock that raises before ctx.update() bubbles the exception out of
        spawn()."""
        manager = _AsyncToolManager(on_duplicate_call="allow")

        def mock_raise() -> Any:
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="simulated failure"):
            await manager.spawn(
                function_callable=_make_call(
                    _async_echo,
                    {"value": "x"},
                    mock=mock_raise,
                ),
                run_ctx=_make_run_ctx(call_id="call-1", name="_async_echo"),
            )


# ---------------------------------------------------------------------------
# end-to-end tests with FakeLLM
# ---------------------------------------------------------------------------


class NewsAgent(Agent):
    """Minimal agent with a bare async tool (no AsyncToolset wrap), to exercise
    the activity's manager dispatch path that the basic_agent example uses."""

    def __init__(self) -> None:
        super().__init__(instructions="news assistant")

    @function_tool
    async def search_news(self, ctx: AsyncRunContext, topic: str) -> str:
        """Search for news on a topic.

        Args:
            topic: What the user wants news about
        """
        await ctx.update(f"searching for {topic}...")
        return f"top story on {topic}: nothing happened."


class TestAsyncToolE2E:
    """End-to-end tests driving the whole framework with a scripted FakeLLM.
    Mocks run through the activity's manager — same lifecycle as the real tool."""

    async def test_mock_returns_directly(self) -> None:
        """Mock returns without calling ctx.update — the mock's value lands as
        the function_call_output for the LLM's tool call."""

        async def mock_search(ctx: AsyncRunContext, topic: str) -> str:
            return f"MOCKED: nothing about {topic}"

        llm = FakeLLM(
            fake_responses=[
                # user asks → LLM emits the tool call
                FakeLLMResponse(
                    input="weather news",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="search_news",
                            arguments='{"topic": "weather"}',
                            call_id="call_1",
                        )
                    ],
                ),
                # tool output → LLM acknowledges
                FakeLLMResponse(
                    input="MOCKED: nothing about weather",
                    content="Here is the news.",
                    ttft=0,
                    duration=0,
                ),
            ]
        )

        async with AgentSession(llm=llm) as sess:
            with mock_tools(NewsAgent, {"search_news": mock_search}):
                await sess.start(NewsAgent())
                result = await asyncio.wait_for(sess.run(user_input="weather news"), timeout=5.0)

                result.expect.next_event().is_function_call(
                    name="search_news", arguments={"topic": "weather"}
                )
                fnc_out = result.expect.next_event().is_function_call_output()
                assert "MOCKED" in fnc_out.event().item.output
                assert fnc_out.event().item.is_error is False
                result.expect.next_event().is_message(role="assistant")
                result.expect.no_more_events()

    async def test_mock_raises_after_first_update(self) -> None:
        """Mock calls ``ctx.update()`` then raises :class:`ToolError`. The update
        produces an intermediate ``function_call_output``; the post-update error
        is surfaced through the manager's ``_enqueue_reply`` as a second
        ``function_call_output`` with ``is_error=True`` and the ToolError message
        preserved (non-ToolError exceptions get sanitized to ``"An internal error
        occurred"``)."""

        async def mock_search(ctx: AsyncRunContext, topic: str) -> str:
            await ctx.update(f"searching for {topic}...")
            raise ToolError("simulated post-update failure")

        update_output = UPDATE_TEMPLATE.format(
            function_name="search_news",
            call_id="call_1",
            message="searching for weather...",
        )
        reply_instructions = REPLY_INSTRUCTIONS.format(pending_call_ids=["call_1_finished"])

        llm = FakeLLM(
            fake_responses=[
                # 1) user asks → LLM emits the tool call
                FakeLLMResponse(
                    input="weather news",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="search_news",
                            arguments='{"topic": "weather"}',
                            call_id="call_1",
                        )
                    ],
                ),
                # 2) ctx.update lands as a function_call_output → LLM hedges
                FakeLLMResponse(
                    input=update_output,
                    content="One moment...",
                    ttft=0,
                    duration=0,
                ),
                # 3) background error → _enqueue_reply triggers generate_reply
                # with REPLY_INSTRUCTIONS as a system message → LLM apologizes
                FakeLLMResponse(
                    input=reply_instructions,
                    content="Sorry, that failed.",
                    ttft=0,
                    duration=0,
                ),
            ]
        )

        async with AgentSession(llm=llm) as sess:
            with mock_tools(NewsAgent, {"search_news": mock_search}):
                agent = NewsAgent()
                await sess.start(agent)
                result = await asyncio.wait_for(sess.run(user_input="weather news"), timeout=5.0)

                # tool got called
                result.expect.contains_function_call(name="search_news")

                # the post-update error landed in the agent's chat_ctx via the
                # manager's _enqueue_reply path (not via a SpeechHandle, so it
                # doesn't surface in RunResult.events — we read chat_ctx directly).
                error_outputs = [
                    item
                    for item in agent.chat_ctx.items
                    if item.type == "function_call_output" and item.is_error
                ]
                assert len(error_outputs) == 1, "expected one is_error function_call_output"
                # ToolError message preserved verbatim — non-ToolError exceptions
                # would be sanitized to "An internal error occurred" instead.
                assert "simulated post-update failure" in error_outputs[0].output

                # the agent narrates the failure to the user via the generate_reply
                # the manager triggers from _enqueue_reply. The presence of this
                # message confirms the REPLY_INSTRUCTIONS turn fired.
                assert any(
                    getattr(ev.item, "role", None) == "assistant"
                    and "Sorry" in (ev.item.text_content or "")
                    for ev in result.events
                    if getattr(ev, "type", None) == "message"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
