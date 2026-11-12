"""Approval held by the framework: which calls of a ``requires_approval`` tool may run."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from livekit.agents import Agent, AgentTask, Completed, InputRequired, RunContext, function_tool
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput, FunctionToolCall
from livekit.agents.llm.tool_context import (
    APPROVAL_ANSWERED,
    APPROVAL_KEY,
    WAITING_PREFIX,
    AnsweredReadBack,
    answered_read_back,
    approval_scope,
    as_model_reads,
)
from livekit.agents.llm.utils import build_legacy_openai_schema

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


def call(
    call_id: str, args: str = '{"date": "monday"}', name: str = "cancel", mark: str | None = None
) -> FunctionCall:
    extra = {APPROVAL_KEY: mark} if mark else {}
    return FunctionCall(call_id=call_id, name=name, arguments=args, extra=extra)


def asked(call_id: str, args: str = '{"date": "monday"}') -> FunctionCall:
    return call(call_id, args, mark="asked")


def ran(call_id: str, args: str = '{"date": "monday"}') -> FunctionCall:
    return call(call_id, args, mark="granted")


def out(call_id: str, output: str, *, is_error: bool = False) -> FunctionCallOutput:
    return FunctionCallOutput(call_id=call_id, name="cancel", output=output, is_error=is_error)


def read_back(call_id: str, question: str = "Cancel monday?") -> FunctionCallOutput:
    """The output of a call that asked: its question, which stays as it is."""
    return out(call_id, f"{WAITING_PREFIX}{question}")


def said(text: str) -> ChatMessage:
    """The person's next message: the answer."""
    return ChatMessage(role="user", content=[text])


def test_first_call_asks() -> None:
    assert answered_read_back([call("c1")], call("c1")) is None


def test_a_repeat_after_the_answer_has_it() -> None:
    history = [asked("c1"), read_back("c1"), said("yes"), call("c2")]
    answered = answered_read_back(history, call("c2"))
    assert answered is not None
    assert answered.call.call_id == "c1" and answered.question == "Cancel monday?"


def test_repeat_before_the_answer_asks() -> None:
    history = [asked("c1"), read_back("c1"), call("c2")]
    assert answered_read_back(history, call("c2")) is None


def test_an_answer_runs_one_repeat() -> None:
    history = [asked("c1"), read_back("c1"), said("yes"), ran("c2"), out("c2", "canceled")]
    assert answered_read_back(history, call("c3")) is None


def test_a_newer_ask_replaces_the_older() -> None:
    history = [
        asked("c1"),
        read_back("c1"),
        said("yes"),
        asked("c2", '{"date": "tuesday"}'),
        read_back("c2", "Cancel tuesday?"),
        said("yes"),
    ]
    answered = answered_read_back(history, call("c3"))
    assert answered is not None and answered.question == "Cancel tuesday?"


def test_a_missing_detail_answered_is_not_an_approval() -> None:
    # the call asked for an email, not for approval: its answer approves nothing
    history = [call("c1"), read_back("c1", "Which email?"), said("jo@example.com")]
    assert answered_read_back(history, call("c2")) is None


def test_other_tools_do_not_count() -> None:
    history = [asked("x1"), read_back("x1"), said("yes")]
    assert answered_read_back(history, call("c1", name="other")) is None


async def test_the_wrapper_runs_only_when_granted() -> None:
    ran: list[str] = []

    @function_tool(requires_approval=True)
    async def cancel(date: str) -> str:
        """Cancel the appointment on a date."""
        ran.append(date)
        return "canceled"

    assert "approved" not in str(build_legacy_openai_schema(cancel))
    asked = await cancel(date="monday")
    assert isinstance(asked, InputRequired) and asked.verbatim
    assert (
        asked.question
        == "Please confirm: Cancel the appointment on a date (date='monday'). Yes or no?"
    )
    with approval_scope(FunctionCall(call_id="a", name="cancel", arguments="{}"), None):
        assert isinstance(await cancel(date="monday"), InputRequired)
    answered = AnsweredReadBack(call("x", "{}"), asked.question)
    with approval_scope(FunctionCall(call_id="b", name="cancel", arguments="{}"), answered):
        assert await cancel(date="monday") == "canceled"
    assert ran == ["monday"]


async def test_a_repeat_runs_on_the_read_back_the_person_answered() -> None:
    read: list[str] = []

    def read_back(amount: int, card: str | None = None) -> str:
        read.append(card or "")
        return f"Charge ${amount} on the Amex? Yes or no?"

    @function_tool(requires_approval=read_back)
    async def charge(amount: int, card: str | None = None) -> str:
        """Charge."""
        return "charged"

    answered = AnsweredReadBack(asked("c1", '{"amount": 5}'), "Charge $5 on the Amex? Yes or no?")
    # its arguments repeated, in any order: it runs without reading back again
    with approval_scope(call("c2", '{ "amount":5 }'), answered):
        assert await charge(amount=5) == "charged"
    assert read == []
    # other arguments that read back the same: what the person approved
    with approval_scope(call("c3", '{"amount": 5, "card": "amex"}'), answered):
        assert await charge(amount=5, card="amex") == "charged"
    # other arguments that read back otherwise: asked again
    with approval_scope(call("c4", '{"amount": 6}'), answered):
        assert isinstance(await charge(amount=6), InputRequired)


async def test_a_read_back_takes_the_tools_own_parameters() -> None:
    @function_tool(requires_approval=lambda date, nights: f"Book {date} for {nights}? Yes or no?")
    async def book(date: str, nights: int) -> str:
        """Book."""
        return "booked"

    asked = await book(date="monday", nights=2)
    assert isinstance(asked, InputRequired) and asked.question == "Book monday for 2? Yes or no?"
    assert asked.verbatim


async def test_a_read_back_may_ask_for_what_is_missing_first() -> None:
    def read_back(date: str, email: str | None = None) -> str | InputRequired:
        return InputRequired("Which email?") if email is None else f"Book {date} to {email}?"

    @function_tool(requires_approval=read_back)
    async def book(date: str, email: str | None = None) -> str:
        """Book."""
        return "booked"

    first = FunctionCall(call_id="a", name="book", arguments="{}")
    with approval_scope(first, None):
        missing = await book(date="monday")
    assert isinstance(missing, InputRequired) and not missing.verbatim
    assert APPROVAL_KEY not in first.extra
    second = FunctionCall(call_id="b", name="book", arguments="{}")
    with approval_scope(second, None):
        asked_ = await book(date="monday", email="jo@example.com")
    assert isinstance(asked_, InputRequired) and asked_.verbatim
    assert second.extra[APPROVAL_KEY] == "asked"


@dataclass
class Verify(AgentTask[str]):
    def __post_init__(self) -> None:
        super().__init__(instructions="verify", llm=FakeLLM(fake_responses=[]))

    async def on_enter(self, ctx: RunContext) -> Completed[str]:
        return Completed("verified; call the tool again")


async def verify_first(ctx: RunContext) -> str:
    return await Verify()


async def test_a_read_back_may_await_a_task_that_verifies_first() -> None:
    """Guard and re-call: the read-back awaits a task when a precondition is unmet; the task
    takes the conversation, and its value is the call's output."""
    desk = Agent(instructions="desk")

    @desk.tool(requires_approval=verify_first)
    async def cancel(ctx: RunContext) -> str:
        """Cancel."""
        return "canceled"

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="cancel it",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="cancel", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(
                    input="verified; call the tool again", content="ok", ttft=0, duration=0
                ),
            ]
        )
    )
    result = await desk.run("cancel it")
    outputs = [i for i in result.new_messages() if i.type == "function_call_output"]
    assert outputs[0].output == "verified; call the tool again"
    assert result.output == "ok"


def test_the_model_reads_an_answered_read_back_as_not_run() -> None:
    history = [asked("c1"), read_back("c1"), said("yes"), call("c2"), read_back("c2", "Cancel?")]
    seen = as_model_reads(history)
    assert seen[1].output == APPROVAL_ANSWERED.format(answer="yes", name="cancel")
    assert history[1].output == f"{WAITING_PREFIX}Cancel monday?"  # what is stored stays
    assert seen[4].output.startswith(WAITING_PREFIX)  # not answered yet, not an approval ask


@dataclass
class Greet(AgentTask[str]):
    def __post_init__(self) -> None:
        super().__init__(
            instructions="greet",
            llm=FakeLLM(
                fake_responses=[
                    FakeLLMResponse(
                        input="hi",
                        content="",
                        ttft=0,
                        duration=0,
                        tool_calls=[FunctionToolCall(name="finish", arguments="{}", call_id="g1")],
                    )
                ]
            ),
        )

    @function_tool
    async def finish(self, ctx: RunContext) -> Completed[str]:
        """Done."""
        return Completed("greeted")


async def test_a_task_run_as_the_root_ends_the_run_with_its_value() -> None:
    """A task run directly, as a test does, ends the run when it completes."""
    task = Greet()
    result = await task.run("hi")
    assert result.output == "greeted"
    assert task.done()
