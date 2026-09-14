from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, ToolError

from .fake_llm import FakeLLM, FakeLLMResponse

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
async def test_aclose_while_on_enter_awaits_agent_task() -> None:
    """Closing the session while on_enter awaits an AgentTask must not deadlock:
    drain() waits for the on_enter task, which waits for the activity handoff,
    which waits for the activity lock held by drain()."""
    session = AgentSession(llm=FakeLLM())
    agent = _ParentAgent()
    await session.start(agent)
    await asyncio.wait_for(agent.entered.wait(), timeout=5.0)

    await asyncio.wait_for(session.aclose(), timeout=10.0)

    assert isinstance(agent.task_error, ToolError)


@pytest.mark.asyncio
@pytest.mark.virtual_time
@pytest.mark.parametrize(
    "phase", ["handoff_requested", "on_exit", "scheduling_paused", "tool_starts_during_drain"]
)
async def test_handoff_while_tool_awaits_agent_task(phase: str) -> None:
    tool_started = asyncio.Event()
    release_tool = asyncio.Event()
    exit_started = asyncio.Event()
    release_exit = asyncio.Event()
    task_attempted = asyncio.Event()
    target_entered = asyncio.Event()
    task_entered = asyncio.Event()
    switch_started = asyncio.Event()
    release_switch = asyncio.Event()
    tool_task: asyncio.Task | None = None
    task_error: ToolError | None = None

    async def wait_until(predicate: Callable[[], bool]) -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    class InlineTask(AgentTask[str]):
        def __init__(self) -> None:
            super().__init__(instructions="Run an inline task")

        async def on_enter(self) -> None:
            task_entered.set()
            self.complete("completed")

    class Target(Agent):
        async def on_enter(self) -> None:
            target_entered.set()

    target = Target(instructions="Target agent")

    class Source(Agent):
        async def on_exit(self) -> None:
            exit_started.set()
            if phase == "on_exit":
                await release_exit.wait()

        @function_tool
        async def start_task(self, ctx: RunContext) -> str:
            """Run an inline task."""
            nonlocal tool_task, task_error
            tool_task = asyncio.current_task()
            ctx.speech_handle.allow_interruptions = False
            tool_started.set()
            await release_tool.wait()
            task_attempted.set()
            try:
                return await InlineTask()
            except ToolError as e:
                task_error = e
                raise

        @function_tool
        async def switch(self) -> Agent:
            """Switch to the target agent."""
            switch_started.set()
            if phase == "tool_starts_during_drain":
                await release_switch.wait()
            return target

    source = Source(instructions="Source agent")
    tool_call_delay = 1.0 if phase == "tool_starts_during_drain" else 0.0
    session = AgentSession(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input=name,
                    content="",
                    ttft=tool_call_delay if name == "start_task" else 0,
                    duration=tool_call_delay if name == "start_task" else 0,
                    tool_calls=[FunctionToolCall(name=name, arguments="{}", call_id=name)],
                )
                for name in ("start_task", "switch")
            ]
        ),
    )
    try:
        await session.start(source)
        activity = session._activity
        assert activity is not None
        if phase == "tool_starts_during_drain":
            session.generate_reply(user_input="switch")
            await asyncio.wait_for(switch_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.current_speech is None), timeout=5.0)
            # The next LLM response is in flight when the handoff tool returns.
            session.generate_reply(user_input="start_task", allow_interruptions=False)
            release_switch.set()
            await asyncio.wait_for(exit_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.scheduling_paused), timeout=5.0)
            assert not tool_started.is_set()
            await asyncio.wait_for(tool_started.wait(), timeout=5.0)
            release_tool.set()
            await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
        else:
            session.generate_reply(user_input="start_task")
            await asyncio.wait_for(tool_started.wait(), timeout=5.0)
            await asyncio.wait_for(wait_until(lambda: activity.current_speech is None), timeout=5.0)

        if phase == "handoff_requested":
            # Hold the transition before drain starts to test synchronous admission blocking.
            async with session._activity_lock:
                session.update_agent(target)
                release_tool.set()
                await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
                assert not exit_started.is_set()
                assert task_error is not None
        elif phase in ("on_exit", "scheduling_paused"):
            session.generate_reply(user_input="switch")
            await asyncio.wait_for(exit_started.wait(), timeout=5.0)
            if phase == "scheduling_paused":
                await asyncio.wait_for(wait_until(lambda: activity.scheduling_paused), timeout=5.0)
            release_tool.set()
            await asyncio.wait_for(task_attempted.wait(), timeout=5.0)
            release_exit.set()

        await asyncio.wait_for(target_entered.wait(), timeout=5.0)
        assert task_error is not None
        assert task_error.message == (
            "An agent transition is in progress, so this tool call cannot continue. "
            "Wait until the transition is complete before retrying, if the tool is "
            "available to the new agent."
        )
        assert not task_entered.is_set()
        assert tool_task is not None and tool_task.done()
        outputs = [
            item
            for item in source.chat_ctx.items
            if item.type == "function_call_output" and item.call_id == "start_task"
        ]
        assert len(outputs) == 1
        assert outputs[0].is_error
        assert outputs[0].output == task_error.message
        await asyncio.wait_for(session.aclose(), timeout=5.0)
    finally:
        release_switch.set()
        release_exit.set()
        if tool_task is not None:
            tool_task.cancel()
            await asyncio.gather(tool_task, return_exceptions=True)
        if session._update_activity_atask is not None:
            await asyncio.wait_for(session._update_activity_atask, timeout=5.0)
        await asyncio.wait_for(session.aclose(), timeout=5.0)
