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
async def test_parallel_agent_tasks_run_in_turn() -> None:
    """Two AgentTasks awaited from one turn's parallel tool calls both pause the same
    activity, so they queue and run one after the other. Running them concurrently would
    leave every handoff but the last overwritten, and those tasks waiting on a result
    nothing can produce - function calls that never return and a speech that never ends."""
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
            # each dialog completes on its own user turn
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
        first = sess.current_agent
        assert isinstance(first, DialogTask)

        # the first dialog hands back, and the one queued behind it takes the activity
        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        second = sess.current_agent
        assert isinstance(second, DialogTask) and second is not first

        await asyncio.wait_for(sess.run(user_input="done"), timeout=5.0)
        assert isinstance(sess.current_agent, ParallelDialogAgent)
    finally:
        # a queued task that hung instead of running would leave its function call
        # unfinished and the close waiting on it - bounded so that regression reports
        # these assertions rather than stalling the loop with nothing left to schedule
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(sess.aclose(), timeout=30.0)

    assert sorted(agent.outcomes) == ["email:ran", "name:ran"]

    # both calls carry an output: neither func_exec was left awaiting a result forever
    outputs = {
        item.call_id: item.output
        for item in agent.chat_ctx.items
        if item.type == "function_call_output" and item.call_id in ("call_1", "call_2")
    }
    assert set(outputs) == {"call_1", "call_2"}
    assert not any("refused" in out for out in outputs.values())


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


@pytest.mark.asyncio
async def test_nested_agent_task_from_a_later_turn() -> None:
    """The nested task is awaited from a tool call that arrives on a user turn after the
    outer task is already running, rather than from the outer task's own on_enter reply.

    The tool of a later turn runs in a task the outer handoff never created, so it does
    not inherit the outer's floor hold and has to be let past it explicitly - otherwise
    it waits for a release that only its own completion can produce."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="start_outer", arguments="{}", call_id="call_1")],
            ),
            # the outer task only asks a question on entry - no tool call this turn
            FakeLLMResponse(
                input="outer_greeting", content="what is your name?", ttft=0, duration=0
            ),
            # the nested handoff is triggered by this later user turn
            FakeLLMResponse(
                input="Dana",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[FunctionToolCall(name="start_inner", arguments="{}", call_id="call_2")],
            ),
            FakeLLMResponse(
                input="inner_greeting", content="and your date of birth?", ttft=0, duration=0
            ),
        ]
    )
    sess = AgentSession(llm=llm)
    try:
        await sess.start(RootAgent())

        await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        assert isinstance(sess.current_agent, OuterTask)

        await asyncio.wait_for(sess.run(user_input="Dana"), timeout=5.0)
        assert isinstance(sess.current_agent, InnerTask)
    finally:
        # a nested task that never got the floor leaves its function call unfinished and
        # the close waiting on it - bounded so a regression reports the assertion above
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(sess.aclose(), timeout=30.0)
