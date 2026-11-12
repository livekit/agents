"""Who words what reaches the person: questions are asked in the facing agent's own words
unless verbatim; verbatim results reach the person as written."""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents import Agent, InputRequired, RunContext, VoiceSession
from livekit.agents.llm import FunctionToolCall

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

QUESTION = "What email address should the confirmation go to?"
PHRASED = "Sure! Where should I send the confirmation?"
ASKED = (
    "Ask the person this, in your own words, as your whole reply, and add nothing else: " + QUESTION
)


def respond(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


def agent_asking(*, verbatim: bool) -> Agent:
    desk = Agent(
        instructions="desk",
        llm=FakeLLM(
            fake_responses=[
                respond("book me", "", FunctionToolCall(name="book", arguments="{}", call_id="b1")),
                respond(ASKED, PHRASED),
            ]
        ),
    )

    @desk.tool
    async def book(ctx: RunContext) -> InputRequired:
        """Book."""
        return InputRequired(QUESTION, verbatim=verbatim)

    return desk


async def test_a_verbatim_question_is_asked_as_written() -> None:
    result = await agent_asking(verbatim=True).run("book me")
    assert result.status == "input_required" and result.question == QUESTION
    assert result.verbatim == QUESTION


async def test_a_call_question_keeps_its_verbatim_request() -> None:
    """On a call, a verbatim ``InputRequired`` settles the run with the question as written."""
    agent = Agent(instructions="Ask for the missing booking detail.")

    @agent.tool
    async def ask() -> InputRequired:
        """Ask the caller for their booking reference."""
        return InputRequired("What is your booking reference?", verbatim=True)

    llm = FakeLLM(
        fake_responses=[
            respond(
                "check my booking", "", FunctionToolCall(name="ask", arguments="{}", call_id="a1")
            )
        ]
    )
    async with VoiceSession(
        llm=llm, vad=None, turn_handling={"turn_detection": "manual"}
    ) as session:
        await session.start(agent)
        result = await asyncio.wait_for(session._run(user_input="check my booking"), timeout=5)
        assert result.status == "input_required"
        assert result.question == "What is your booking reference?"
        assert result.verbatim == result.question
