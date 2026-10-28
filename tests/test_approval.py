"""Approval through ``agent.run()`` on the core (tools.md "Approval", decision 15): a
``needs_approval`` call asks its read-back, only a repeat of the same validated call after the
person's answer runs the body, and each answer is consumed once."""

from __future__ import annotations

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents.llm import FunctionToolCall
from livekit.agents.voice.agent_run import RunHooks, reset_run_hooks, set_run_hooks

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit

APPROVAL_KEY = "lk.approval"


def respond(input: str, content: str = "", *calls: FunctionToolCall) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0, duration=0, tool_calls=list(calls))


def cancel_call(call_id: str, arguments: str = '{"date": "monday"}') -> FunctionToolCall:
    return FunctionToolCall(name="cancel", arguments=arguments, call_id=call_id)


async def test_a_call_runs_only_on_the_repeat_after_the_answer_and_once() -> None:
    ran: list[str] = []
    llm = FakeLLM(
        fake_responses=[
            respond("cancel monday", "", cancel_call("c1")),
            # the repeat omits the default: the validated identity is the same call
            respond("yes", "", cancel_call("c2", '{"date": "monday", "notify": true}')),
            respond("canceled", "Done, it is canceled."),
            respond("cancel monday again", "", cancel_call("c3")),
        ]
    )
    desk = Agent(instructions="desk", llm=llm)

    @desk.tool(needs_approval=True)
    async def cancel(ctx: RunContext, date: str, notify: bool = True) -> str:
        """Cancel the appointment on a date.

        Args:
            date: the day
            notify: tell the clinic
        """
        ran.append(date)
        return "canceled"

    asked = await desk.run("cancel monday")
    assert asked.status == "input_required"
    assert asked.question.startswith("Please confirm: Cancel the appointment on a date")
    assert asked.verbatim == asked.question
    assert ran == []
    first = next(i for i in asked.new_messages() if i.type == "function_call")
    assert first.extra[APPROVAL_KEY] == "asked"

    granted = await desk.run("yes", message_history=asked.all_messages())
    assert granted.status == "completed" and granted.output == "Done, it is canceled."
    assert ran == ["monday"]
    repeat = next(i for i in granted.new_messages() if i.type == "function_call")
    assert repeat.extra[APPROVAL_KEY] == "granted"
    # the history keeps the person's words as the read-back's answer
    answered = [
        i for i in granted.all_messages() if i.type == "function_call_output" and i.call_id == "c1"
    ]
    assert answered[0].output == "yes"

    again = await desk.run("cancel monday again", message_history=granted.all_messages())
    assert again.status == "input_required", "an identical later call asks again"
    assert ran == ["monday"]


async def test_a_read_back_function_takes_the_tools_parameters_or_asks_first() -> None:
    def read_back(ctx: RunContext, date: str, email: str | None = None) -> str | InputRequired:
        return InputRequired("Which email?") if email is None else f"Book {date} to {email}?"

    llm = FakeLLM(
        fake_responses=[
            respond(
                "book monday",
                "",
                FunctionToolCall(name="book", arguments='{"date": "monday"}', call_id="b1"),
            ),
            respond(
                "jo@example.com",
                "",
                FunctionToolCall(
                    name="book",
                    arguments='{"date": "monday", "email": "jo@example.com"}',
                    call_id="b2",
                ),
            ),
        ]
    )
    desk = Agent(instructions="desk", llm=llm)

    @desk.tool(needs_approval=read_back)
    async def book(ctx: RunContext, date: str, email: str | None = None) -> str:
        """Book."""
        return "booked"

    token = set_run_hooks(RunHooks(delegated=True))  # questions pass up as written
    try:
        missing = await desk.run("book monday")
        assert missing.status == "input_required" and missing.question == "Which email?"
        first = next(i for i in missing.new_messages() if i.type == "function_call")
        assert APPROVAL_KEY not in first.extra, "a missing detail is not a read-back"

        asked = await desk.run("jo@example.com", message_history=missing.all_messages())
    finally:
        reset_run_hooks(token)
    assert asked.status == "input_required"
    assert asked.question == "Book monday to jo@example.com?"
