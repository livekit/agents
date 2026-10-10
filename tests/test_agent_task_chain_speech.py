"""A run must record the speech of the next AgentTask in a chain.

An agent drives a conversation as a chain of awaited AgentTasks (``await Greeting(); await
Dni(); ...`` in on_enter, or a TaskGroup). A tool of the current task calls ``complete()``,
the parent's coroutine resumes and awaits the next task, and that task's on_enter speaks.
The run that carried the tool call ends when the tool's speech ends unless the parent's
resumption keeps it open until the next task's speech is scheduled.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.beta.workflows import TaskGroup
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class GreetingTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="greeting")

    async def on_enter(self) -> None:
        await self.session.say("am I speaking with the account holder?")

    @function_tool
    async def holder(self, ctx: RunContext) -> str:
        """Called when the user confirms they are the account holder."""
        self.complete(None)
        return "ok"


class IdTask(AgentTask[None]):
    def __init__(self) -> None:
        super().__init__(instructions="id")

    async def on_enter(self) -> None:
        await self.session.say("could you confirm your ID number?")

    @function_tool
    async def confirmed(self, ctx: RunContext) -> str:
        """Called when the user confirms the ID number."""
        self.complete(None)
        return "ok"


class Closing(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="closing")

    async def on_enter(self) -> None:
        await self.session.say("thank you, transferring you now.")


class ChainedEntry(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="entry")

    async def on_enter(self) -> None:
        await GreetingTask()
        await IdTask()
        self.session.update_agent(Closing())


class GroupedEntry(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="entry")

    async def on_enter(self) -> None:
        await (
            TaskGroup(summarize_chat_ctx=False)
            .add(GreetingTask, id="greeting", description="greet the user")
            .add(IdTask, id="id", description="confirm the id number")
        )
        self.session.update_agent(Closing())


def _llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="yes, speaking",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="holder", arguments="{}", call_id="c1")],
            ),
            FakeLLMResponse(
                input="yes, that is right",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="confirmed", arguments="{}", call_id="c2")],
            ),
            FakeLLMResponse(input="ok", content="", ttft=0.1, duration=0.1),
        ]
    )


async def _spoken(sess: AgentSession, text: str) -> None:
    # the start-time handoff into the first task is still in flight when start() returns
    for _ in range(100):
        if any(i.type == "message" and i.text_content == text for i in sess.history.items):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"{text!r} was never spoken")


@pytest.mark.parametrize("entry", [ChainedEntry, GroupedEntry])
async def test_next_task_speech_is_recorded_in_the_run(entry: type[Agent]) -> None:
    async with AgentSession(llm=_llm()) as sess:
        await sess.start(entry())
        await _spoken(sess, "am I speaking with the account holder?")

        result = await asyncio.wait_for(sess.run(user_input="yes, speaking"), timeout=5.0)
        result.expect.next_event().is_function_call(name="holder")
        result.expect.next_event().is_function_call_output()
        assert [e.item.text_content for e in result.events if e.type == "message"] == [
            "could you confirm your ID number?"
        ]

        result = await asyncio.wait_for(sess.run(user_input="yes, that is right"), timeout=5.0)
        result.expect.next_event().is_function_call(name="confirmed")
        result.expect.next_event().is_function_call_output()
        result.expect.contains_agent_handoff(new_agent_type=Closing)
        assert [e.item.text_content for e in result.events if e.type == "message"] == [
            "thank you, transferring you now."
        ]
