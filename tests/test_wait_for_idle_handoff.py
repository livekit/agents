from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.agents.voice.agent_activity import ActivityClosedError
from livekit.agents.voice.events import ToolExecutionUpdatedEvent

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

SLOW_CALL = FunctionToolCall(name="slow", arguments="{}", call_id="call_slow")
BOOK_CALL = FunctionToolCall(name="book", arguments="{}", call_id="call_book")


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 5.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(_poll(), timeout=timeout)


async def _wait_for_handoff(session: AgentSession) -> None:
    """Let an in-flight ``update_agent`` finish so teardown sees a settled session."""
    task = session._update_activity_atask
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=10.0)


class _AgentA(Agent):
    """Runs a non-cancellable background tool to hold the handoff open."""

    def __init__(self) -> None:
        super().__init__(instructions="agent a")

    @function_tool
    async def slow(self, ctx: RunContext) -> str:
        """Do slow work in the background."""
        await ctx.update("working")
        _AgentA.slow_started.set()
        await _AgentA.slow_release.wait()
        return "done"

    slow_started = asyncio.Event()
    slow_release = asyncio.Event()


class _AgentB(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="agent b")


def _session_llm(*responses: FakeLLMResponse) -> FakeLLM:
    return FakeLLM(fake_responses=list(responses))


def _tool_call_response(input_text: str, call: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input_text, content="", ttft=0.0, duration=0.0, tool_calls=[call])


async def test_wait_for_idle_retargets_during_live_handoff() -> None:
    """``wait_for_idle`` must follow the handoff while the old activity is closing.

    ``AgentActivity.aclose`` marks itself closed before it awaits the executor drain,
    but ``AgentSession._update_activity`` only swaps ``_activity`` after that await.
    A concurrent ``wait_for_idle`` therefore sees a closed activity that is still
    current and re-raises, even though the session is alive and the incoming
    activity is about to start.
    """
    _AgentA.slow_started = asyncio.Event()
    _AgentA.slow_release = asyncio.Event()

    session = AgentSession(llm=_session_llm(_tool_call_response("go", SLOW_CALL)))
    waiter: asyncio.Task[object] | None = None
    try:
        await session.start(_AgentA())
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)
        await asyncio.wait_for(_AgentA.slow_started.wait(), timeout=5.0)

        old_activity = session._activity
        assert old_activity is not None

        session.update_agent(_AgentB())
        await _wait_until(lambda: old_activity._closed and session._activity is old_activity)
        assert session._closing_task is None

        waiter = asyncio.create_task(session.wait_for_idle(), name="wait_for_idle")
        # let the waiter observe the closing-but-still-current activity; it cannot
        # complete until slow_release unblocks aclose()
        for _ in range(10):
            await asyncio.sleep(0)

        _AgentA.slow_release.set()

        activity = await asyncio.wait_for(waiter, timeout=5.0)
        assert isinstance(activity.agent, _AgentB)
        assert session._activity is activity
    finally:
        _AgentA.slow_release.set()
        if waiter is not None and not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        await _wait_for_handoff(session)
        await asyncio.wait_for(session.aclose(), timeout=10.0)


@pytest.mark.parametrize("wrapper", ["wait_for", "create_task", "nested_task"])
async def test_wait_for_idle_fails_fast_from_a_draining_tool(wrapper: str) -> None:
    """A caller inside the outgoing activity's drain graph must not await the handoff.

    An agent-scoped, non-cancellable tool that calls ``ctx.update()`` keeps running in
    the background and is awaited by ``aclose() -> _tool_executor.drain()``. If it then
    waits for the session to be idle, waiting for the handoff would close a cycle
    (wait -> handoff -> drain -> tool -> wait). Keep the pre-fix fast failure for
    callers the outgoing activity is draining.

    ``wrapper`` covers the wrappers that run the wait in a child task instead of the
    tool's own task: ``asyncio.wait_for`` schedules one before Python 3.12, and
    ``asyncio.create_task`` always does. Such a caller must still fail fast rather than
    reach the inner timeout.
    """
    tool_started = asyncio.Event()
    wait_go = asyncio.Event()
    wait_done = asyncio.Event()
    outcome: BaseException | None = None

    class DrainingToolAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="agent a")

        @function_tool
        async def wait_inside(self, ctx: RunContext) -> str:
            """Wait for idleness from inside a running tool."""
            nonlocal outcome
            await ctx.update("working")
            tool_started.set()
            await wait_go.wait()

            async def _wait_for_idle() -> None:
                await ctx.session.wait_for_idle()

            try:
                if wrapper == "wait_for":
                    # before Python 3.12, wait_for schedules the coroutine in a child task
                    await asyncio.wait_for(_wait_for_idle(), timeout=1.0)
                elif wrapper == "create_task":
                    await asyncio.wait_for(asyncio.create_task(_wait_for_idle()), timeout=1.0)
                else:
                    # nested_task: two levels of child tasks must inherit the guard
                    async def _spawn_wait() -> None:
                        await asyncio.create_task(_wait_for_idle())

                    await asyncio.wait_for(asyncio.create_task(_spawn_wait()), timeout=1.0)
            # asyncio.TimeoutError is a distinct class on 3.10, an alias of the builtin later
            except (ActivityClosedError, asyncio.TimeoutError, TimeoutError) as exc:
                outcome = exc
            wait_done.set()
            return "done"

    call = FunctionToolCall(name="wait_inside", arguments="{}", call_id="call_wait")
    session = AgentSession(llm=_session_llm(_tool_call_response("go", call)))
    try:
        await session.start(DrainingToolAgent())
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        old_activity = session._activity
        assert old_activity is not None

        session.update_agent(_AgentB())
        await _wait_until(lambda: old_activity._closed and session._activity is old_activity)
        assert session._closing_task is None

        wait_go.set()
        # the inner wait_for bounds the deadlock; this outer one fails a hang hard
        await asyncio.wait_for(wait_done.wait(), timeout=2.0)
        assert isinstance(outcome, ActivityClosedError), outcome

        await _wait_for_handoff(session)
        assert isinstance(session.current_agent, _AgentB)
    finally:
        wait_go.set()
        await _wait_for_handoff(session)
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_detached_child_of_draining_tool_retargets() -> None:
    """A detached task spawned by a draining tool is not part of the drain graph.

    The tool is non-cancellable and keeps running after ``ctx.update()``, so the
    handoff's drain awaits its task. The idle wait runs in a separate task the tool
    never awaits: the drain does not wait on it either, so it must re-target to the
    incoming activity instead of copying the fast failure of the task actually being
    drained.
    """
    tool_started = asyncio.Event()
    child_go = asyncio.Event()
    child_done = asyncio.Event()
    release_tool = asyncio.Event()
    child_activity: object | None = None
    child_error: BaseException | None = None
    watchers: list[asyncio.Task[None]] = []

    class DetachedChildAgent(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="agent a")

        @function_tool
        async def background(self, ctx: RunContext) -> str:
            """Spawn a detached idle waiter, then keep working."""
            nonlocal child_activity, child_error
            await ctx.update("working")
            tool_started.set()

            async def _report_when_idle() -> None:
                nonlocal child_activity, child_error
                try:
                    await child_go.wait()
                    child_activity = await ctx.session.wait_for_idle()
                except BaseException as exc:
                    child_error = exc
                finally:
                    child_done.set()

            # not awaited by the tool: the drain only waits for the tool's own task
            watchers.append(asyncio.create_task(_report_when_idle(), name="detached_wait_for_idle"))
            await release_tool.wait()
            return "done"

    call = FunctionToolCall(name="background", arguments="{}", call_id="call_background")
    session = AgentSession(llm=_session_llm(_tool_call_response("go", call)))
    try:
        await session.start(DetachedChildAgent())
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)

        old_activity = session._activity
        assert old_activity is not None

        session.update_agent(_AgentB())
        await _wait_until(lambda: old_activity._closed and session._activity is old_activity)
        assert session._closing_task is None

        # the waiter observes the closing-but-still-current activity
        child_go.set()
        for _ in range(10):
            await asyncio.sleep(0)

        # it must be parked on the incoming activity, not fail fast like the drained tool
        assert not child_done.is_set(), child_error

        # the tool must finish for the handoff's drain (and so the retarget) to proceed
        release_tool.set()
        await asyncio.wait_for(child_done.wait(), timeout=5.0)

        assert child_error is None, child_error
        assert isinstance(getattr(child_activity, "agent", None), _AgentB)
        await _wait_for_handoff(session)
        assert isinstance(session.current_agent, _AgentB)
    finally:
        child_go.set()
        release_tool.set()
        if watchers:
            await asyncio.gather(*watchers, return_exceptions=True)
        await _wait_for_handoff(session)
        await asyncio.wait_for(session.aclose(), timeout=10.0)


async def test_session_scoped_async_tool_reply_survives_handoff_window() -> None:
    """A session-scoped AsyncToolset reply must not be dropped while an agent hands off.

    The tool finishes while the previous activity is still draining. The session-scoped
    executor resolves its target through ``AgentSession.wait_for_idle``; that call used
    to raise ``ActivityClosedError`` and the executor discarded the buffered update and
    final output, so the new agent never spoke the result.
    """
    _AgentA.slow_started = asyncio.Event()
    _AgentA.slow_release = asyncio.Event()
    book_started = asyncio.Event()
    book_release = asyncio.Event()

    @function_tool
    async def book(ctx: RunContext) -> str:
        """Book a flight."""
        await ctx.update("looking up flights")
        book_started.set()
        await book_release.wait()
        await ctx.update("flight booked")
        return "booking reference 123"

    toolset = AsyncToolset(id="booking", tools=[book])
    session = AgentSession(
        llm=_session_llm(
            _tool_call_response("go", SLOW_CALL),
            _tool_call_response("book", BOOK_CALL),
        ),
        tools=[toolset],
    )

    events: list[ToolExecutionUpdatedEvent] = []
    session.on("tool_execution_updated", events.append)

    try:
        await session.start(_AgentA())
        await asyncio.wait_for(session.run(user_input="go"), timeout=5.0)
        await asyncio.wait_for(_AgentA.slow_started.wait(), timeout=5.0)
        await asyncio.wait_for(session.run(user_input="book"), timeout=5.0)
        await asyncio.wait_for(book_started.wait(), timeout=5.0)

        old_activity = session._activity
        assert old_activity is not None
        executor = toolset._executor
        book_task = executor._running_tasks[BOOK_CALL.call_id].exe_task

        session.update_agent(_AgentB())
        await _wait_until(lambda: old_activity._closed and session._activity is old_activity)
        assert session._closing_task is None

        # the tool's update and final output both land in the handoff window
        book_release.set()
        await asyncio.wait_for(asyncio.shield(book_task), timeout=5.0)

        _AgentA.slow_release.set()
        await _wait_for_handoff(session)
        if (reply_task := executor._reply_task) is not None:
            await asyncio.wait_for(asyncio.shield(reply_task), timeout=5.0)

        scheduled_ids = {
            update_id
            for ev in events
            if ev.update.type == "tool_reply_updated" and ev.update.status == "scheduled"
            for update_id in ev.update.update_ids
        }
        assert "call_book_final" in scheduled_ids, f"scheduled call_ids={scheduled_ids}"
    finally:
        book_release.set()
        _AgentA.slow_release.set()
        await _wait_for_handoff(session)
        await asyncio.wait_for(session.aclose(), timeout=10.0)
