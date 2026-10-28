"""Who words what reaches the person: questions are asked in the facing agent's own words
unless verbatim; verbatim results reach the person as written."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.llm import FunctionToolCall, ToolResult
from livekit.agents.llm.tool_context import ASK_IN_OWN_WORDS, DELIVERED

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

QUESTION = "What email address should the confirmation go to?"
PHRASED = "Sure! Where should I send the confirmation?"
DISCLAIMER = "Calls are recorded for quality."


def respond(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


def agent_asking(*, verbatim: bool) -> Agent:
    desk = Agent(
        instructions="desk",
        llm=FakeLLM(
            fake_responses=[
                respond("book me", "", FunctionToolCall(name="book", arguments="{}", call_id="b1")),
                respond(ASK_IN_OWN_WORDS.format(question=QUESTION), PHRASED),
            ]
        ),
    )

    @desk.tool
    async def book(ctx: RunContext) -> InputRequired:
        """Book."""
        return InputRequired(QUESTION, verbatim=verbatim)

    return desk


async def test_facing_the_person_a_question_is_asked_in_own_words() -> None:
    result = await agent_asking(verbatim=False).run("book me")
    assert result.status == "input_required" and result.question == PHRASED
    assert result.new_messages()[-1].text_content == PHRASED
    assert result.verbatim is None


async def test_a_verbatim_question_is_asked_as_written() -> None:
    result = await agent_asking(verbatim=True).run("book me")
    assert result.status == "input_required" and result.question == QUESTION
    assert result.verbatim == QUESTION


def agent_saying(*, reply_required: bool) -> Agent:
    desk = Agent(
        instructions="desk",
        llm=FakeLLM(
            fake_responses=[
                respond("hi", "", FunctionToolCall(name="disclose", arguments="{}", call_id="d1")),
                respond(DELIVERED.format(text=DISCLAIMER), "How can I help?"),
            ]
        ),
    )

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(DISCLAIMER, verbatim=True, reply_required=reply_required)

    return desk


async def test_a_verbatim_result_without_reply_ends_the_turn_as_written() -> None:
    result = await agent_saying(reply_required=False).run("hi")
    assert result.status == "completed" and result.output == DISCLAIMER
    assert result.verbatim == DISCLAIMER


async def test_a_verbatim_result_leads_the_answer_and_the_model_continues() -> None:
    result = await agent_saying(reply_required=True).run("hi")
    assert result.status == "completed" and result.output == f"{DISCLAIMER}\n\nHow can I help?"
    assert result.verbatim == DISCLAIMER
    outputs = [i for i in result.new_messages() if i.type == "function_call_output"]
    assert outputs[0].output == DELIVERED.format(text=DISCLAIMER)
