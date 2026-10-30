"""Work in the background on a call (design §3, §5): the acknowledgement, the progress spoken, a
result told as new, and a barge-in that stops only the speech. Seen in what reaches the
realtime model and in what the call says."""

from __future__ import annotations

import asyncio
import contextlib
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
from .test_verbatim_speech import call_desk, last_text

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


def tool_items(ctx: ChatContext) -> list[str]:
    """The calls and outputs in ``ctx``, in order: an output reads ``<call id> out``."""
    return [
        i.call_id if i.type == "function_call" else f"{i.call_id} out"
        for i in ctx.items
        if i.type in ("function_call", "function_call_output")
    ]


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
    gate, ended = asyncio.Event(), list[str]()
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
    gate, ended = asyncio.Event(), list[str]()
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
    gate, ended, speaking = asyncio.Event(), list[str](), asyncio.Event()
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


async def answer_before_the_turn_ends(
    expert: Callable[..., A2ADelegate], progress: str = ""
) -> tuple[LiveSession, ScriptedRealtimeModel]:
    """A call whose expert answers, after ``progress`` when given, while the turn that released
    it still speaks; the answer is told."""
    gate, ended, speaking = asyncio.Event(), list[str](), asyncio.Event()
    gate.set()
    model = ScriptedRealtimeModel(
        Reply("Let me check.", [call("ask", "a1", request="slots")], hold=speaking),
        Reply("We have 10 slots.", when=told("a1_final")),
    )
    desk = Agent(instructions="front desk", tools=[expert(slow_expert(gate, ended, progress))])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("any slots?")
        await soon(lambda: ended, "the expert's answer")
        await asyncio.sleep(0.1)
        speaking.set()
        await soon(lambda: len(spoken(session)) > 1, "the answer told")
    return session, model


@pytest.mark.rule("B71", "B96")
async def test_a_result_before_the_turn_ends_is_told_once_after_it_with_no_acknowledgement(
    expert: Callable[..., A2ADelegate],
) -> None:
    session, model = await answer_before_the_turn_ends(expert)

    assert spoken(session) == ["Let me check.", "We have 10 slots."]
    assert len(model.requests) == 2, "the acknowledgement asked no reply"
    assert tool_items(model.requests[-1]) == ["a1", "a1 out", "a1_final", "a1_final out"]


@pytest.mark.rule("B96")
async def test_progress_and_a_result_before_the_turn_ends_follow_the_acknowledgement(
    expert: Callable[..., A2ADelegate],
) -> None:
    session, model = await answer_before_the_turn_ends(expert, progress="Checking.")

    assert spoken(session) == ["Let me check.", "We have 10 slots."]
    assert tool_items(model.requests[-1]) == [
        "a1",
        "a1 out",
        "a1_update_1",
        "a1_update_1 out",
        "a1_final",
        "a1_final out",
    ]


@pytest.mark.rule("B96")
async def test_updates_while_the_turn_speaks_never_pause_the_tool_and_follow_its_first_output() -> (
    None
):
    speaking, worked = asyncio.Event(), asyncio.Event()
    inserted: list[bool] = []
    model = ScriptedRealtimeModel(
        Reply("On it.", [call("book", "b1", day="monday")], hold=speaking)
    )
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        ctx.update("booking now")
        for n in (1, 2):

            def landed(_: asyncio.Future[None], call_id: str = f"b1_update_{n}") -> None:
                inserted.append(call_id in tool_items(session.history))

            ctx.update(f"still booking {n}").add_done_callback(landed)
        worked.set()
        return f"booked {day}"

    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("book monday")
        await asyncio.wait_for(worked.wait(), 1)
        recorded_while_speaking = list(inserted)
        speaking.set()
        await soon(lambda: len(inserted) == 2, "both updates recorded")

    assert recorded_while_speaking == []
    assert inserted == [True, True], "each future resolved after its insert"
    assert tool_items(model.active_session.chat_ctx) == [
        "b1",
        "b1 out",
        "b1_update_1",
        "b1_update_1 out",
        "b1_update_2",
        "b1_update_2 out",
        "b1_final",
        "b1_final out",
    ]


@pytest.mark.rule("B71")
async def test_a_result_after_the_acknowledgement_started_follows_it(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended, acking = asyncio.Event(), list[str](), asyncio.Event()
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="slots")]),
        Reply(
            "Checking now.", hold=acking, when=lambda c: any(DISPATCHED in o for o in outputs(c))
        ),
        Reply("We have 10 slots.", when=told("a1_final")),
    )
    desk = Agent(instructions="front desk", tools=[expert(slow_expert(gate, ended))])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("any slots?")
        await soon(lambda: len(model.requests) == 2, "the acknowledgement")
        gate.set()
        await soon(lambda: "There are 10 slots." in outputs(session.history), "the answer")
        acking.set()
        await soon(lambda: len(spoken(session)) > 1, "the answer told")

    assert spoken(session) == ["Checking now.", "We have 10 slots."]


@pytest.mark.rule("B71")
async def test_a_result_reply_cut_before_any_of_it_played_is_told_again(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended = asyncio.Event(), list[str]()
    gate.set()
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="slots")]),
        Reply("", hold=asyncio.Event(), when=told("a1_final")),
        Reply("We have 10 slots.", when=told("a1_final")),
        Reply("Sure.", when=answering("hold on")),
    )
    desk = Agent(instructions="front desk", tools=[expert(slow_expert(gate, ended))])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        rt = model.active_session
        rt.hear("any slots?")
        await soon(lambda: any(told("a1_final")(r) for r in model.requests), "the result reply")
        rt.hear("hold on")
        await soon(lambda: {"Sure.", "We have 10 slots."} <= set(spoken(session)), "both told")

    assert sum(told("a1_final")(r) for r in model.requests) == 2


@pytest.mark.rule("B71")
async def test_a_result_reply_cut_after_part_of_it_played_is_not_told_again(
    expert: Callable[..., A2ADelegate],
) -> None:
    gate, ended = asyncio.Event(), list[str]()
    gate.set()
    model = ScriptedRealtimeModel(
        Reply("", [call("ask", "a1", request="slots")]),
        Reply("We have 10 slots.", hold=asyncio.Event(), when=told("a1_final")),
        Reply("Sure.", when=answering("hold on")),
    )
    desk = Agent(instructions="front desk", tools=[expert(slow_expert(gate, ended))])
    async with LiveSession(llm=model) as session:
        await session.start(desk)
        rt = model.active_session
        rt.hear("any slots?")
        await soon(lambda: any(told("a1_final")(r) for r in model.requests), "the result reply")
        await asyncio.sleep(0.2)  # its first words play
        rt.hear("hold on")
        await soon(lambda: "Sure." in spoken(session), "the reply to the barge-in")
        await asyncio.sleep(0.3)

    assert sum(told("a1_final")(r) for r in model.requests) == 1


@pytest.mark.rule("B71")
@pytest.mark.virtual_time
@pytest.mark.no_concurrent
async def test_on_the_pipeline_a_result_before_the_turn_ends_is_told_after_it() -> None:
    model = ScriptedLLM(
        Reply("Let me check.", [call("book", "1")], when=last_text("Book me in")),
        Reply("It is booked.", when=last_text("1_final")),
    )

    async def book(ctx: RunContext) -> str:
        """Book."""
        ctx.update("checking")
        return "booked Tuesday"

    spoken = await call_desk(model, book, [("Let me check.", 1.0), ("It is booked.", 1.0)])

    assert spoken == ["Let me check.", "It is booked."]
    assert len(model.requests) == 2, "the acknowledgement asked no reply"


@pytest.mark.rule("B97")
async def test_a_tool_that_times_out_its_wait_for_an_update_loses_none_of_its_later_ones() -> None:
    speaking = asyncio.Event()
    model = ScriptedRealtimeModel(
        Reply("On it.", [call("book", "b1", day="monday")], hold=speaking),
        Reply("Monday is booked.", when=told("b1_final")),
    )
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        for message in ("booking now", "still booking"):
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ctx.update(message), 0.05)
        return f"booked {day}"

    async with LiveSession(llm=model) as session:
        await session.start(desk)
        model.active_session.hear("book monday")
        await asyncio.sleep(0.3)  # both waits time out while the turn speaks
        speaking.set()
        await soon(lambda: "Monday is booked." in spoken(session), "the result told")

    assert tool_items(model.active_session.chat_ctx) == [
        "b1",
        "b1 out",
        "b1_update_1",
        "b1_update_1 out",
        "b1_final",
        "b1_final out",
    ]
