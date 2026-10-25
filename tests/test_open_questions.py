"""Open questions on a call (docs/delegation.md): the person has one question in front of them,
one per step that asked; the answer to a delegate's question goes to its tool as said with no
model call, any other to the model; a follow-up goes first, and the next question is asked once
the one in front is answered."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, LiveSession, RunContext
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.tool_context import ANSWERS_KEY, WAITING_PREFIX

from .fake_llm import FakeLLMResponse
from .test_agent_api import RecordingLLM

pytestmark = [pytest.mark.unit]


def _call(name: str, call_id: str, **arguments: str) -> FunctionToolCall:
    return FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)


async def test_the_answer_goes_to_the_tool_that_asked_with_no_model_call() -> None:
    desk = Agent(instructions="front desk")
    requests: list[str] = []

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str | InputRequired:
        """The back office.

        Args:
            request: the request, in full
        """
        requests.append(request)
        if request == "a booking":
            return InputRequired("Name and email?", verbatim=True)
        return f"booked for {request}"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)

    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book me in",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[_call("ask", "c1", request="a booking")],
            ),
            FakeLLMResponse(input="booked for Ana", content="All set.", ttft=0, duration=0),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="book me in")
        asked = len(llm.requests)
        await session.run(user_input="Ana")

    assert requests == ["a booking", "Ana"], "the answer went to the tool, as the person said it"
    # the model was not asked about the answer: its next request carries the tool's output
    first_after = llm.requests[asked]
    outputs = [i for i in first_after.items if i.type == "function_call_output"]
    assert outputs[-1].output == "booked for Ana"
    answer_call = [
        i for i in session.history.items if i.type == "function_call" and i.call_id != "c1"
    ][0]
    assert answer_call.extra[ANSWERS_KEY] == "c1"
    placeholder = [
        i for i in session.history.items if i.type == "function_call_output" and i.call_id == "c1"
    ][0]
    assert placeholder.output == f"{WAITING_PREFIX}Name and email?", "it keeps its question"
    said = [i.text_content for i in session.history.items if i.type == "message"]
    assert said == ["book me in", "Name and email?", "Ana", "All set."]


async def test_the_questions_of_one_step_are_one_and_the_model_reads_the_answer() -> None:
    desk = Agent(instructions="front desk")

    @desk.tool
    async def payer(ctx: RunContext) -> InputRequired:
        """Ask for the payer."""
        return InputRequired("Which payer?", verbatim=True)

    @desk.tool
    async def email(ctx: RunContext) -> InputRequired:
        """Ask for the email."""
        return InputRequired("Which email?", verbatim=True)

    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="cover me and send the ticket",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[_call("payer", "p1"), _call("email", "e1")],
            ),
            FakeLLMResponse(input="Aetna, ana@example.com", content="Thanks.", ttft=0, duration=0),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="cover me and send the ticket")
        said = [i.text_content for i in session.history.items if i.type == "message"]
        assert said[1:] == ["Which payer?\n\nWhich email?"], "one question holds both"

        await session.run(user_input="Aetna, ana@example.com")
        said = [i.text_content for i in session.history.items if i.type == "message"]

    assert said[1:] == ["Which payer?\n\nWhich email?", "Aetna, ana@example.com", "Thanks."]
    assert session._activity is None or not session._activity._questions, "nothing waits"
    outputs = {
        i.call_id: i.output for i in session.history.items if i.type == "function_call_output"
    }
    assert outputs["p1"] == f"{WAITING_PREFIX}Which payer?"
    assert outputs["e1"] == f"{WAITING_PREFIX}Which email?"


async def test_a_follow_up_of_the_answered_question_goes_before_the_queue() -> None:
    desk = Agent(instructions="front desk")
    seats = asyncio.Event()

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str | InputRequired:
        """The back office.

        Args:
            request: the request, in full
        """
        if request in ("a flight", "why?"):
            return InputRequired(f"Email for the ticket? ({request})", verbatim=True)
        return f"booked, ticket to {request}"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)

    @desk.tool
    async def seat(ctx: RunContext) -> InputRequired:
        """Ask for a seat."""
        await ctx.update("checking the seats")
        await seats.wait()
        return InputRequired("Window or aisle?", verbatim=True)

    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="a flight with a seat",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[_call("ask", "c1", request="a flight"), _call("seat", "s1")],
            ),
            FakeLLMResponse(
                input="booked, ticket to ana@example.com", content="Booked.", ttft=0, duration=0
            ),
        ]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="a flight with a seat")
        # the seat's question comes back while the email question is in front: it waits
        seats.set()
        activity = session._activity
        assert activity is not None
        for _ in range(100):
            if len(activity._questions) == 2:
                break
            await asyncio.sleep(0.05)
        assert [q.asked for q in activity._questions] == [True, False]

        await session.run(user_input="why?")
        said = [i.text_content for i in session.history.items if i.type == "message"]
        assert said[-1] == "Email for the ticket? (why?)", "the follow-up goes first"

        await session.run(user_input="ana@example.com")
        said = [i.text_content for i in session.history.items if i.type == "message"]

    assert said[-2:] == ["Booked.", "Window or aisle?"]


async def test_a_lane_question_whose_call_is_not_in_the_context_still_relays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delegate's question asked by text is in front of the person when the call starts, but
    the asking call is not in the call's context: the spoken answer still goes to the delegate's
    tool as said, naming the call that asked."""
    from types import SimpleNamespace

    desk = Agent(instructions="front desk")
    got: list[tuple[str, Any]] = []

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str:
        """The back office.

        Args:
            request: the request, in full
        """
        got.append((request, ctx.function_call.extra.get(ANSWERS_KEY)))
        return "booked"

    ask._relays_answer = True  # type: ignore[attr-defined]  (as a delegate's tool)
    row = SimpleNamespace(
        id="AST_q",
        internal=SimpleNamespace(
            ask={"question": "Please confirm: Monday?", "call_id": "c9", "answer_tool": "ask"}
        ),
    )
    monkeypatch.setattr(LiveSession, "_take_lane_questions", lambda self: [row])
    llm = RecordingLLM(
        fake_responses=[FakeLLMResponse(input="booked", content="Booked.", ttft=0, duration=0)]
    )
    async with LiveSession(llm=llm) as session:
        await session.start(desk)
        await session.run(user_input="yes")

    assert got == [("yes", "c9")]
