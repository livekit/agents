from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentTask, Completed, LiveSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, ToolError

from .fake_llm import FakeLLM, FakeLLMResponse, FakeLLMStream

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _SimpleTask(AgentTask):
    def __init__(self) -> None:
        super().__init__(instructions="simple task")


class _ParentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="parent agent")
        self.entered = asyncio.Event()
        self.exit_started = asyncio.Event()
        self.task_error: BaseException | None = None

    async def on_enter(self) -> None:
        self.entered.set()
        # hold on_enter until the session close has started draining this activity
        # (on_exit runs inside drain() while it holds the activity lock), reproducing
        # a participant disconnect racing `await AgentTask()`
        await self.exit_started.wait()
        try:
            await _SimpleTask()
        except ToolError as e:
            self.task_error = e

    async def on_exit(self) -> None:
        self.exit_started.set()


@pytest.mark.asyncio
@pytest.mark.virtual_time
async def test_cancelled_drain_allows_inline_task() -> None:
    drain_started = asyncio.Event()
    task_completed = asyncio.Event()

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            return Completed("completed")

    class Source(Agent):
        @function_tool
        async def start_task(self) -> str:
            """Run an inline task."""
            result = await InlineTask()
            task_completed.set()
            return result

    session = LiveSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="start_task",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="start_task", arguments="{}", call_id="task")
                    ],
                )
            ]
        )
    )

    async def drain() -> None:
        drain_started.set()
        await session.drain()

    async with session:
        await session.start(Source(instructions="Source agent"))
        assert session._activity is not None
        async with session._activity._lock:
            drain_task = asyncio.create_task(drain())
            await asyncio.wait_for(drain_started.wait(), timeout=5.0)
            drain_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drain_task

        await asyncio.wait_for(session.run(user_input="start_task"), timeout=5.0)
        await asyncio.wait_for(task_completed.wait(), timeout=5.0)


@pytest.mark.asyncio
@pytest.mark.virtual_time
@pytest.mark.parametrize("generation_pending", [False, True])
async def test_public_drain_while_tool_awaits_agent_task(
    generation_pending: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    release_generation = asyncio.Event()
    task_attempted = asyncio.Event()
    task_completed = asyncio.Event()
    tool_task: asyncio.Task | None = None
    drain_task: asyncio.Task | None = None

    async def stream_response(stream: FakeLLMStream) -> None:
        if stream._get_index_text() == "start_task":
            stream._send_chunk(
                tool_calls=[FunctionToolCall(name="start_task", arguments="{}", call_id="task")]
            )
            # Emit the tool call before ending generation to exercise both scheduler waits.
            await release_generation.wait()

    monkeypatch.setattr(FakeLLMStream, "_run", stream_response)

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            assert drain_task is not None and drain_task.done()
            assert scheduler.done()
            return Completed("completed")

    class Source(Agent):
        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            nonlocal tool_task
            tool_task = asyncio.current_task()
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            await release_tool.wait()
            assert activity._lock.locked()
            assert activity.scheduling_paused
            assert not activity._new_turns_blocked
            assert not scheduler.done()
            assert speech._generations[-1].done() == (not generation_pending)
            task_attempted.set()
            result = await InlineTask()
            assert result == "completed"
            task_completed.set()
            return result

    session = LiveSession(llm=FakeLLM())
    try:
        await session.start(Source(instructions="Source agent"))
        activity = session._activity
        assert activity is not None
        scheduler = activity._scheduling_atask
        assert scheduler is not None
        speech = session.generate_reply(user_input="start_task")
        await asyncio.wait_for(tool_started.wait(), timeout=5.0)
        if not generation_pending:
            release_generation.set()
            await asyncio.wait_for(speech._wait_for_generation(), timeout=5.0)

        async def drain() -> None:
            await session.drain()
            assert scheduler.done()
            assert not activity._lock.locked()

        async def wait_paused() -> None:
            while not activity.scheduling_paused:
                await asyncio.sleep(0.01)

        drain_task = asyncio.create_task(drain())
        await asyncio.wait_for(wait_paused(), timeout=5.0)
        release_tool.set()
        await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
        if generation_pending:
            assert activity.current_speech is speech
            assert session._activity_lock.locked()
            assert activity._lock.locked()
            assert not drain_task.done()
            release_generation.set()

        await asyncio.wait_for(asyncio.shield(drain_task), timeout=5.0)
        await asyncio.wait_for(task_completed.wait(), timeout=5.0)
        assert session._activity is activity
        assert not activity.scheduling_paused
    finally:
        release_tool.set()
        release_generation.set()
        if tool_task is not None:
            tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
        if drain_task is not None:
            await asyncio.wait_for(drain_task, timeout=5.0)
        await asyncio.wait_for(session.aclose(), timeout=5.0)


@pytest.mark.asyncio
async def test_aclose_while_on_enter_awaits_agent_task() -> None:
    """Closing the session while on_enter awaits an AgentTask must not deadlock:
    drain() waits for the on_enter task, which waits for the activity handoff,
    which waits for the activity lock held by drain()."""
    session = LiveSession(llm=FakeLLM())
    agent = _ParentAgent()
    await session.start(agent)
    await asyncio.wait_for(agent.entered.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=10.0)

    assert isinstance(agent.task_error, ToolError)
