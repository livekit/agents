"""Work in the background on a call (design §3, §5): the acknowledgement, the progress spoken, a
result told as new, and a barge-in that stops only the speech. Seen in what reaches the
realtime model and in what the call says."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from livekit.agents import Agent, LiveSession, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.delegation.a2a import DISPATCHED
from livekit.agents.llm import ChatContext
from livekit.agents.voice.tool_executor import REPLY_INSTRUCTIONS_AT_TAIL

from .a2a_driver import Door, Reply, ScriptedLLM, call, no_project_env  # noqa: F401
from .fake_realtime import ScriptedRealtimeModel

pytestmark = [pytest.mark.unit]


async def soon(check: Callable[[], Any], what: str) -> None:
    """Polls ``check`` until it holds."""
    for _ in range(200):
        if check():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"never {what}")


def instructed(ctx: ChatContext) -> str:
    """The instructions a reply was asked with: the request's last ``developer`` message."""
    last = ctx.items[-1] if ctx.items else None
    if last is not None and last.type == "message" and last.role == "developer":
        return last.text_content or ""
    return ""


def told(call_id: str) -> Callable[[ChatContext], bool]:
    """A reply asked to tell the person the outputs of ``call_id``."""
    return lambda ctx: call_id in instructed(ctx)


def answering(text: str) -> Callable[[ChatContext], bool]:
    """A reply to the person's ``text``: the last thing they said."""

    def matches(ctx: ChatContext) -> bool:
        users = [i for i in ctx.items if i.type == "message" and i.role == "user"]
        return bool(users) and users[-1].text_content == text

    return matches


def spoken(session: LiveSession) -> list[str]:
    return [
        i.text_content or ""
        for i in session.history.items
        if i.type == "message" and i.role == "assistant"
    ]


def outputs(ctx: ChatContext) -> list[str]:
    return [i.output for i in ctx.items if i.type == "function_call_output"]


class Booking:
    """A desk whose ``book`` tool reports ``booking now`` and ``still booking``, each after
    its gate, then books."""

    def __init__(self) -> None:
        self.second, self.done = asyncio.Event(), asyncio.Event()
        self.desk = Agent(instructions="front desk")

        @self.desk.tool
        async def book(ctx: RunContext, day: str) -> str:
            """Book a cleaning.

            Args:
                day: the day
            """
            await ctx.update("booking now")
            await self.second.wait()
            await ctx.update("still booking")
            await self.done.wait()
            return f"booked {day}"


def slow_expert(gate: asyncio.Event, ended: list[str], progress: str = "") -> Any:
    """An expert's handler that reports ``progress`` when given, answers once ``gate`` is set,
    and records in ``ended`` that it did."""

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        if progress:
            await ctx.update(progress)
        await gate.wait()
        ended.append(ctx.message.text_content or "")
        return "There are 10 slots."

    return handler


@pytest.mark.rule("R21")
async def test_the_acknowledgement_is_spoken_on_a_call(expert: Callable[..., A2ADelegate]) -> None:
    gate, ended = asyncio.Event(), list[str]()
    gate.set()
    model = ScriptedRealtimeModel(Reply("", [call("ask", "a1", request="slots")]))
    desk = Agent(instructions="front desk", tools=[expert(slow_expert(gate, ended))])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("any slots?")
        await soon(lambda: any("There are 10 slots." in outputs(r) for r in model.requests), "told")

    assert any(DISPATCHED in o for r in model.requests for o in outputs(r))


@pytest.mark.rule("R21")
async def test_text_shows_no_acknowledgement_by_default(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    gate, ended = asyncio.Event(), []
    gate.set()
    model = ScriptedLLM(Reply(tool_calls=[call("ask", "a1", request="slots")]), Reply("10 slots."))
    agent = Agent(instructions="front desk", llm=model, tools=[expert(slow_expert(gate, ended))])

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    sent = await door(handler).send("any slots?")

    assert sent.answer == "10 slots."
    assert not any(DISPATCHED in o for r in model.requests for o in outputs(r))
    assert not any(DISPATCHED in str(f) for f in sent.frames)


@pytest.mark.rule("R22")
async def test_on_a_call_the_experts_progress_is_spoken_and_its_answer_told(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended = asyncio.Event(), []
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="slots")]),
        Reply("Checking the calendar now.", when=lambda c: "Checking" in outputs(c)[-1]),
        Reply("We have 10 slots.", when=told("a1_final")),
    )
    handler = slow_expert(gate, ended, progress="Checking the calendar.")
    desk = Agent(instructions="front desk", tools=[expert(handler, announce=False, streamed=True)])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("any slots?")
        await soon(lambda: spoken(session), "the progress spoken")
        gate.set()
        await soon(lambda: len(spoken(session)) > 1, "the answer told")

    assert spoken(session) == ["Checking the calendar now.", "We have 10 slots."]


@pytest.mark.xfail(strict=True, reason='R58: any newer item reads as "may be covered"')
@pytest.mark.rule("R58")
async def test_a_result_after_only_the_persons_words_is_told_as_new() -> None:
    """The result landed while the call was busy and the person spoke after it: no reply had
    it in front of it, so the model is told to tell it, not that it may have told it."""
    booking, busy = Booking(), asyncio.Event()
    booking.second.set()
    model = ScriptedRealtimeModel(
        Reply("On it.", [call("book", "b1", day="monday")]),
        Reply("", hold=busy, when=answering("one second")),
        Reply("", when=answering("is parking free?")),
        Reply("Monday is booked.", when=told("b1_final")),
    )
    async with LiveSession(llm=model) as session:
        await session.start(booking.desk)
        rt = model.active_session
        rt.hear("book monday")
        await soon(lambda: spoken(session), "the acknowledgement")
        rt.hear("one second")
        await soon(lambda: answering("one second")(model.requests[-1]), "busy")
        booking.done.set()
        await soon(lambda: any("booked" in o for o in outputs(session.history)), "the result")
        rt.hear("is parking free?")
        busy.set()
        await soon(lambda: any(told("b1_final")(r) for r in model.requests), "the result")

    telling = next(r for r in model.requests if told("b1_final")(r))
    assert instructed(telling) == REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids=["b1_final"])


@pytest.mark.rule("B5")
async def test_a_barge_in_stops_the_speech_and_never_the_experts_task(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended, speaking = asyncio.Event(), [], asyncio.Event()
    model = ScriptedRealtimeModel(
        Reply("Let me check.", [call("ask", "a1", request="slots")], hold=speaking),
        Reply("Sure.", when=answering("wait, sorry")),
    )
    desk = Agent(
        instructions="front desk", tools=[expert(slow_expert(gate, ended), announce=False)]
    )
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        rt = model.active_session
        rt.hear("any slots?")
        await asyncio.sleep(0.2)
        rt.hear("wait, sorry")  # over the reply that made the call
        speaking.set()
        gate.set()
        await soon(lambda: ended, "the expert's answer")
        await soon(lambda: "There are 10 slots." in outputs(session.history), "the answer")
        await soon(lambda: "Sure." in spoken(session), "the reply to the barge-in")

    interrupted = [i.interrupted for i in session.history.items if i.type == "message"]
    assert interrupted == [False, True, False, False], "only the reply that made the call stopped"
    assert ended == ["slots"]
