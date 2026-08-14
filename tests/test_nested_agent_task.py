from __future__ import annotations

import asyncio
import contextlib

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall, ToolError

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


class DialogTask(AgentTask):
    """A task that needs a user turn to complete, like a detail-capture dialog."""

    def __init__(self) -> None:
        super().__init__(instructions="dialog task")

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="dialog_greeting")

    @function_tool
    async def finish(self, ctx: RunContext) -> str:
        """Called to complete the dialog."""
        self.complete(None)
        return "done"


class ParallelDialogAgent(Agent):
    """Two tools that each await an AgentTask, for an LLM turn that calls both."""

    def __init__(self) -> None:
        super().__init__(instructions="root agent")
        self.outcomes: list[str] = []

    @function_tool
    async def open_name_dialog(self, ctx: RunContext) -> str:
        """Collects the name."""
        return await self._open("name")

    @function_tool
    async def open_email_dialog(self, ctx: RunContext) -> str:
        """Collects the email."""
        return await self._open("email")

    async def _open(self, which: str) -> str:
        # a refusal is reported back as the tool's output, the way the model would see
        # it; re-raising would instead surface through the awaiting session.run()
        try:
            await DialogTask()
        except ToolError as e:
            self.outcomes.append(f"{which}:refused")
            return f"{which} refused: {e}"
        self.outcomes.append(f"{which}:ran")
        return f"{which} captured"


@pytest.mark.asyncio
async def test_parallel_agent_tasks_refuse_the_second() -> None:
    """Two AgentTasks awaited from one turn's parallel tool calls contend for the same
    activity. Only one may pause it: the other's handoff would be overwritten, leaving it
    waiting on a result nothing can produce, so its function call never returns and the
    speech never finishes. The second is refused with a ToolError instead."""
    llm = FakeLLM(
        fake_responses=[
            # one turn, two tool calls - each tool awaits a DialogTask
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="open_name_dialog", arguments="{}", call_id="call_1"),
                    FunctionToolCall(name="open_email_dialog", arguments="{}", call_id="call_2"),
                ],
            ),
            FakeLLMResponse(input="dialog_greeting", content="what is it?", ttft=0, duration=0),
            # user answers the dialog that did run -> it completes and hands back
            FakeLLMResponse(
                input="done",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="call_3")],
            ),
        ]
    )
    agent = ParallelDialogAgent()
    sess = AgentSession(llm=llm)
    try:
        await sess.start(agent)

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        assert isinstance(sess.current_agent, DialogTask)

        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        assert isinstance(sess.current_agent, ParallelDialogAgent)
    finally:
        # a refused task that instead hung would leave its function call unfinished, and
        # the close waiting on it - bounded so that regression reports these assertions
        # rather than stalling the loop with nothing left to schedule
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(sess.aclose(), timeout=30.0)

    assert sorted(o.split(":")[1] for o in agent.outcomes) == ["ran", "refused"]

    # both calls carry an output: neither func_exec was left awaiting a result forever
    outputs = {
        item.call_id: item.output
        for item in agent.chat_ctx.items
        if item.type == "function_call_output" and item.call_id in ("call_1", "call_2")
    }
    assert set(outputs) == {"call_1", "call_2"}
    # the refusal reaches the model rather than being silently dropped
    assert sum(1 for out in outputs.values() if "refused" in out) == 1


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
