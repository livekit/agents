from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, LiveSession, RunContext, function_tool
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


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
    async with LiveSession(llm=llm) as sess:
        await sess.start(GreeterToGreeting())

        result = await asyncio.wait_for(sess.run(user_input="go"), timeout=5.0)
        # the delayed greeting must be part of this run's output
        result.expect.contains_message(role="assistant")
