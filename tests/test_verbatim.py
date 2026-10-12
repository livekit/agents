"""Who words what reaches the person: questions are asked in the facing agent's own words
unless verbatim; verbatim results reach the person as written."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, Completed, InputRequired, RunContext
from livekit.agents.llm import FunctionToolCall, ToolResult
from livekit.agents.llm.tool_context import ASK_IN_OWN_WORDS, DELIVERED
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

QUESTION = "What email address should the confirmation go to?"
PHRASED = "Sure! Where should I send the confirmation?"
DISCLAIMER = "Calls are recorded for quality."


def respond(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


def agent_asking(*, verbatim: bool) -> Agent:
    desk = Agent(instructions="desk")

    @desk.tool
    async def book(ctx: RunContext) -> InputRequired:
        """Book."""
        return InputRequired(QUESTION, verbatim=verbatim)

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                respond("book me", "", FunctionToolCall(name="book", arguments="{}", call_id="b1")),
                respond(ASK_IN_OWN_WORDS.format(question=QUESTION), PHRASED),
            ]
        )
    )
    return desk


async def test_facing_the_person_a_question_is_asked_in_own_words() -> None:
    result = await agent_asking(verbatim=False).run("book me")
    assert result.input_required == PHRASED
    assert result.new_messages()[-1].text_content == PHRASED
    assert result.verbatim is None


async def test_a_verbatim_question_is_asked_as_written() -> None:
    result = await agent_asking(verbatim=True).run("book me")
    assert result.input_required == QUESTION
    assert result.verbatim == QUESTION


async def test_asked_by_another_agent_the_question_goes_up_unphrased() -> None:
    token = set_run_hooks(RunHooks(delegated=True))
    try:
        result = await agent_asking(verbatim=False).run("book me")
    finally:
        reset_run_hooks(token)
    assert result.input_required == QUESTION
    assert result.verbatim is None  # the agent facing the person phrases it


def agent_saying(*, reply_required: bool) -> Agent:
    desk = Agent(instructions="desk")

    @desk.tool
    async def disclose(ctx: RunContext) -> ToolResult:
        """Disclose."""
        return ToolResult(DISCLAIMER, verbatim=True, reply_required=reply_required)

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                respond("hi", "", FunctionToolCall(name="disclose", arguments="{}", call_id="d1")),
                respond(DELIVERED.format(text=DISCLAIMER), "How can I help?"),
            ]
        )
    )
    return desk


async def test_a_verbatim_result_without_reply_ends_the_turn_as_written() -> None:
    result = await agent_saying(reply_required=False).run("hi")
    assert result.output == DISCLAIMER
    assert result.verbatim == DISCLAIMER


async def test_a_verbatim_result_leads_the_answer_and_the_model_continues() -> None:
    result = await agent_saying(reply_required=True).run("hi")
    assert result.output == f"{DISCLAIMER}\n\nHow can I help?"
    assert result.verbatim == DISCLAIMER
    outputs = [i for i in result.new_messages() if i.type == "function_call_output"]
    assert outputs[0].output == DELIVERED.format(text=DISCLAIMER)


async def test_a_verbatim_completed_from_a_conversation_tool_is_a_verbatim_result() -> None:
    desk = Agent(instructions="desk")

    @desk.tool
    async def code(ctx: RunContext) -> Completed[str]:
        """The code."""
        return Completed("HTL-7A3K", verbatim=True)

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                respond("code?", "", FunctionToolCall(name="code", arguments="{}", call_id="c1")),
                respond(DELIVERED.format(text="HTL-7A3K"), "Anything else?"),
            ]
        )
    )
    result = await desk.run("code?")
    assert result.output == "HTL-7A3K\n\nAnything else?"
