from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, AgentTask, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_session import FakeActions, create_session, run_session

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class AskNameTask(AgentTask):
    """A task that needs a future user turn to complete."""

    def __init__(self) -> None:
        super().__init__(instructions="ask name task")

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="ask_name")

    @function_tool
    async def record_name(self, ctx: RunContext, name: str) -> str:
        """Called when the user provides their name."""
        self.complete(None)
        return "recorded"


class SurveyAgent(Agent):
    """Agent whose on_enter spans multiple user turns (awaits an AgentTask)."""

    def __init__(self) -> None:
        super().__init__(instructions="survey agent")

    async def on_enter(self) -> None:
        await AskNameTask()


class Greeter(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="greeter agent")

    @function_tool
    async def start_survey(self, ctx: RunContext) -> None:
        """Called when the user is ready to start the survey."""
        self.session.update_agent(SurveyAgent())


def _build_fake_llm() -> FakeLLM:
    return FakeLLM(
        fake_responses=[
            # turn 1: the greeter routes to SurveyAgent via update_agent()
            FakeLLMResponse(
                input="ready",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(name="start_survey", arguments="{}", call_id="call_1")
                ],
            ),
            # AskNameTask.on_enter -> generate_reply(instructions="ask_name")
            FakeLLMResponse(input="ask_name", content="what is your name?", ttft=0.1, duration=0.1),
            # turn 2: the user answers; the task records the name and completes
            FakeLLMResponse(
                input="Bob",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[
                    FunctionToolCall(
                        name="record_name", arguments='{"name": "Bob"}', call_id="call_2"
                    )
                ],
            ),
        ]
    )


class GreetingAgent(Agent):
    """Agent whose on_enter does async work before speaking — the run must
    keep tracking on_enter so the greeting is captured before run() returns."""

    def __init__(self) -> None:
        super().__init__(instructions="greeting agent")

    async def on_enter(self) -> None:
        await asyncio.sleep(0.5)
        await self.session.generate_reply(instructions="delayed_greeting")


class GreeterToGreeting(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="greeter agent")

    @function_tool
    async def start(self, ctx: RunContext) -> None:
        """Called when the user asks to proceed."""
        self.session.update_agent(GreetingAgent())


@pytest.mark.asyncio
async def test_update_agent_on_enter_output_captured():
    """run() must not complete before the new agent's on_enter has emitted its
    greeting (on_enter is watched, not awaited)."""
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="go",
                content="",
                ttft=0.1,
                duration=0.1,
                tool_calls=[FunctionToolCall(name="start", arguments="{}", call_id="call_1")],
            ),
            FakeLLMResponse(input="delayed_greeting", content="hello!", ttft=0.1, duration=0.1),
        ]
    )
    async with AgentSession(llm=llm) as sess:
        await sess.start(GreeterToGreeting())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        # the delayed greeting must be part of this run's output
        result.expect.contains_message(role="assistant")


@pytest.mark.asyncio
async def test_update_agent_long_on_enter_no_deadlock():
    """session.run() must return when a function tool calls update_agent() and
    the new agent's on_enter awaits an AgentTask that needs more user input.

    The run watches _update_activity_task; if the activity update waits for
    on_enter (which waits for the next user turn), the run can never complete —
    a circular wait, since run() must return before the next turn can be sent.
    """
    llm = _build_fake_llm()
    async with AgentSession(llm=llm) as sess:
        await sess.start(Greeter())

        # must not deadlock: returns once the handoff is done and the
        # AskNameTask question has been spoken
        first_result = await asyncio.wait_for(sess.run(user_input="ready"), timeout=5.0)
        assert first_result is not None

        # the next turn completes the task
        second_result = await asyncio.wait_for(sess.run(user_input="Bob"), timeout=5.0)
        second_result.expect.contains_function_call(name="record_name")


@pytest.mark.asyncio
async def test_handoff_reply_preserves_user_metrics() -> None:
    class HandoffTarget(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="handoff target")

        async def on_enter(self) -> None:
            await self.session.generate_reply(instructions="handoff_reply")

    class HandoffSource(Agent):
        def __init__(self) -> None:
            super().__init__(instructions="handoff source")

        @function_tool
        async def handoff(self, ctx: RunContext) -> None:
            """Hand the current turn to the next agent."""
            target = HandoffTarget()
            self.session.update_agent(target)

    actions = FakeActions()
    actions.add_user_speech(0.5, 2.5, "go")
    actions.add_llm(
        content="",
        tool_calls=[FunctionToolCall(name="handoff", arguments="{}", call_id="call_1")],
    )
    actions.add_llm("hello from the new agent", input="handoff_reply")
    actions.add_tts(1.0)

    session = create_session(actions)
    conversation_events = []
    session.on("conversation_item_added", conversation_events.append)

    await run_session(session, HandoffSource(), drain_delay=1.0)

    assistant_messages = [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "assistant"
    ]
    user_messages = [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "user"
    ]
    assert len(user_messages) == 1
    assert "stopped_speaking_at" in user_messages[0].metrics
    assert len(assistant_messages) == 1
    assert assistant_messages[0].text_content == "hello from the new agent"
    assert "e2e_latency" in assistant_messages[0].metrics
