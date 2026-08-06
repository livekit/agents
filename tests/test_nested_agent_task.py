from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class InnerTask(AgentTask):
    """A task that needs a second user turn to complete (user must trigger 'finish')."""

    def __init__(self) -> None:
        super().__init__(instructions="inner task")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="inner_greeting")

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Called to complete the inner task."""
        self.complete(None)
        return "done"


class OuterTask(AgentTask):
    """A task whose on_enter triggers a tool call that awaits InnerTask."""

    def __init__(self) -> None:
        super().__init__(instructions="outer task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="outer_greeting")

    @function_tool
    async def start_inner(self, ctx: RunContext) -> str:
        """Transitions into InnerTask."""
        await InnerTask()
        self.complete(None)
        return "inner completed"


class RootAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="root agent")

    @function_tool
    async def start_outer(self, ctx: RunContext) -> str:
        """Transitions into OuterTask."""
        await OuterTask()
        return "outer completed"


@pytest.mark.asyncio
async def test_nested_agent_task_no_deadlock():
    """session.run() must return when an AgentTask hands off to a nested task
    that collects additional user input before completing."""
    llm = _build_fake_llm()
    async with AgentSession(llm=llm) as sess:
        await sess.start(RootAgent())

        # This must not deadlock — it should return once the on_enter chain
        # has started, even though InnerTask is still waiting for user input.
        first_result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        assert first_result is not None

        # Now complete InnerTask by triggering the finish tool
        second_result = await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        assert second_result is not None


class SimpleTask(AgentTask):
    """A task that needs a user turn to complete (user must trigger 'finish')."""

    def __init__(self) -> None:
        super().__init__(instructions="simple task")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="task_greeting")

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Called to complete the task."""
        self.complete(None)
        return "done"


class EnterHandoffAgent(Agent):
    """Agent whose on_enter reply calls a tool that awaits an AgentTask.

    The on_enter speech predates any session.run(), so the run doesn't watch it.
    """

    def __init__(self) -> None:
        super().__init__(instructions="root agent")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="enter_greeting")

    @function_tool
    async def start_task(self, ctx: RunContext) -> str:
        """Transitions into SimpleTask."""
        await SimpleTask()
        return "task completed"


@pytest.mark.asyncio
async def test_handoff_from_pre_run_speech():
    """A handoff triggered by a speech created before session.run() (e.g. in
    on_enter) must keep the run alive until the new activity has started;
    otherwise the next run() races the transition and gets rejected."""
    llm = FakeLLM(
        fake_responses=[
            # on_enter generate_reply(instructions="enter_greeting") -> calls start_task;
            # slow enough that run(user_input="hi") starts before the tool call lands
            FakeLLMResponse(
                input="enter_greeting",
                content="",
                ttft=1.0,
                duration=1.0,
                tool_calls=[FunctionToolCall(name="start_task", arguments="{}", call_id="call_1")],
            ),
            # user says "hi" while the handoff is in flight; this is the only
            # speech the first run watches
            FakeLLMResponse(input="hi", content="hello!", ttft=1.0, duration=2.0),
            # SimpleTask on_enter greeting
            FakeLLMResponse(input="task_greeting", content="hello from task", ttft=0, duration=0),
            # user says "bye" -> LLM calls finish
            FakeLLMResponse(
                input="bye",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="call_2")],
            ),
            # after start_task tool output, LLM responds
            FakeLLMResponse(input="task completed", content="all done", ttft=0, duration=0),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(EnterHandoffAgent())

        await asyncio.wait_for(sess.run(user_input="hi"), timeout=5.0)
        assert isinstance(sess.current_agent, SimpleTask)

        await asyncio.wait_for(sess.run(user_input="bye"), timeout=5.0)
        assert isinstance(sess.current_agent, EnterHandoffAgent)


def _build_fake_llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            # user says "go" -> LLM calls start_outer
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="start_outer", arguments="{}", call_id="call_1")],
            ),
            # OuterTask on_enter generate_reply(instructions="outer_greeting")
            # -> LLM calls start_inner
            FakeLLMResponse(
                input="outer_greeting",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="start_inner", arguments="{}", call_id="call_2")],
            ),
            # InnerTask on_enter generate_reply(instructions="inner_greeting")
            # -> LLM just says hello (no tool call yet — needs user input to finish)
            FakeLLMResponse(
                input="inner_greeting",
                content="hello from inner",
                ttft=0,
                duration=0,
            ),
            # user says "done" -> LLM calls finish
            FakeLLMResponse(
                input="done",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="call_3")],
            ),
            # after finish tool output, LLM responds to start_inner tool output
            FakeLLMResponse(
                input="inner completed",
                content="",
                ttft=0,
                duration=0,
            ),
            # after start_outer tool output, LLM responds
            FakeLLMResponse(
                input="outer completed",
                content="all done",
                ttft=0,
                duration=0,
            ),
        ]
    )
