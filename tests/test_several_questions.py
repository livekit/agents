"""Several open questions in text (design §4, §7 "Questions"): one question is in front of the
person and the others wait in the order they were queued; one answer goes to every expert that
asked in one step; a question cut off by the person's message waits in line behind its reply.
Seen on the A2A wire and in what reaches the model."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.delegation import A2ADelegate
from livekit.agents.storage import AgentDBStorage

from .a2a_driver import (
    Door,
    Reply,
    ScriptedLLM,
    Sent,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
)
from .test_a2a_delegation import _Dropping
from .test_live_inbox import BOOK_MONDAY, NO_INTERRUPT, bookings, listed, metadata_of, said, until

pytestmark = [pytest.mark.unit]

QUEUED = "livekit/queued"


def asking_expert(question: str, answer: str, seen: list[str]) -> Callable[..., Any]:
    """An expert's handler whose agent asks ``question`` as written, then answers ``answer`` with
    whatever the person said; ``seen`` keeps each message the expert got."""
    agent = Agent(
        instructions="expert",
        llm=ScriptedLLM(Reply(tool_calls=[call("check", "k1")]), Reply(answer)),
    )

    @agent.tool
    async def check(ctx: RunContext) -> InputRequired:
        """Check what the expert needs."""
        return InputRequired(question, verbatim=True)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        seen.append(ctx.message.text_content or "")
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required or result.output

    return handler


# ---------------------------------------------------------------------------------------------
# one question in front of the person
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R36", "R29")
async def test_one_question_is_in_front_and_the_next_waits(door: Callable[..., Door]) -> None:
    monday_q, tuesday_q = "Which time on Monday?", "Which time on Tuesday?"
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(tool_calls=[call("book", "b2", day="tuesday")], when=said("and tuesday")),
        Reply("That too.", when=output_of("b2")),
        Reply("Monday at nine.", when=said("nine")),
        Reply("Tuesday at ten.", when=said("ten")),
        asks={"monday": monday_q, "tuesday": tuesday_q},
        days=("monday", "tuesday"),
    )
    monday = desk.send("book monday")
    await desk.in_the_background()
    tuesday = desk.send("and tuesday")
    await desk.in_the_background(2)
    desk.release["monday"].set()
    asked = await asyncio.wait_for(monday, 20)
    desk.release["tuesday"].set()

    # tuesday's question waits behind monday's, not shown
    async def queued() -> dict[str, Any] | None:
        tasks = await desk.door.list_tasks(desk.thread)
        return next((t for t in tasks if metadata_of(t).get(QUEUED)), None)

    waiting = await until(queued, "the second question queued")
    assert asked.status_text == monday_q and Sent([], waiting).status_text != tuesday_q
    assert not tuesday.done()

    # the person's message answers the one in front; then the next is asked
    answered = await asyncio.wait_for(desk.send("nine"), 20)
    assert answered.task["id"] == asked.task["id"] and answered.answer == "Monday at nine."
    next_asked = await asyncio.wait_for(tuesday, 20)
    assert next_asked.state == "input_required" and next_asked.status_text == tuesday_q
    assert not next_asked.metadata.get(QUEUED)
    later = await asyncio.wait_for(desk.send("ten"), 20)
    assert later.task["id"] == next_asked.task["id"] and later.answer == "Tuesday at ten."


@pytest.mark.rule("B41")
@pytest.mark.xfail(
    strict=True,
    reason="B41: under live both answers enter the inbox as the waiting task's; the row needs the message's own task",
)
async def test_two_answers_to_one_question_give_the_second_a_task_of_its_own(
    door: Callable[..., Door],
) -> None:
    question = "Which time on Monday?"
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply("Monday at nine.", when=said("nine")),
        Reply("Noted, ten.", when=said("ten")),
        asks={"monday": question},
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    desk.release["monday"].set()
    asked = await asyncio.wait_for(first, 20)

    sent = await asyncio.wait_for(
        asyncio.gather(
            desk.send("nine"), desk.door.send("ten", context_id=desk.thread, **NO_INTERRUPT)
        ),
        20,
    )
    ids = {s.task["id"] for s in sent}
    assert asked.task["id"] in ids and len(ids) == 2
    # each response carries its own task's frames only
    for s in sent:
        frames = [f.get("statusUpdate") or f.get("artifactUpdate") for f in s.frames[1:]]
        assert {f["taskId"] for f in frames if f} == {s.task["id"]}


# ---------------------------------------------------------------------------------------------
# one answer to every expert that asked
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R47")
async def test_one_answer_goes_to_each_expert_that_asked_in_one_step(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    fares_saw: list[str] = []
    seats_saw: list[str] = []
    fares = expert(asking_expert("Which date?", "Fare booked.", fares_saw), "fares", tool="fares")
    seats = expert(asking_expert("Which date?", "Seat booked.", seats_saw), "seats", tool="seats")
    front = Agent(
        instructions="front desk",
        tools=[fares, seats],
        llm=ScriptedLLM(
            Reply(
                tool_calls=[
                    call("fares", "f1", request="a fare"),
                    call("seats", "s1", request="a seat"),
                ],
                when=said("plan my trip"),
            ),
            Reply("Both are booked."),
        ),
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await front.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler, inbox="live")
    thread = await desk.thread()
    asked = await asyncio.wait_for(desk.send("plan my trip", context_id=thread), 20)
    assert asked.state == "input_required" and asked.status_text == "Which date?"
    answered = await asyncio.wait_for(desk.send("friday", context_id=thread), 20)

    assert answered.answer == "Both are booked."
    assert fares_saw[-1] == "friday" and seats_saw[-1] == "friday"


# ---------------------------------------------------------------------------------------------
# a question cut off by the person's message
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R48")
async def test_a_question_cut_off_by_a_waiting_message_is_asked_after_its_reply(
    door: Callable[..., Door], expert: Callable[..., A2ADelegate]
) -> None:
    question = "What email address?"
    entered, go, replied = asyncio.Event(), asyncio.Event(), asyncio.Event()
    seen: list[str] = []
    agent = Agent(
        instructions="back office",
        llm=ScriptedLLM(
            Reply(tool_calls=[call("book", "e1", slot="first")]),
            Reply("Booked, the confirmation goes to a@b.c.", when=said("a@b.c")),
        ),
    )

    @agent.tool
    async def book(ctx: RunContext, slot: str) -> InputRequired:
        """Book a slot.

        Args:
            slot: the slot
        """
        entered.set()
        await go.wait()
        return InputRequired(question, verbatim=True)

    async def back_office(ctx: A2ASessionContext[Any]) -> Any:
        seen.append(ctx.message.text_content or "")
        result = await agent.run(ctx.message, message_history=ctx.message_history)
        return result.input_required or result.output

    front = Agent(
        instructions="front desk",
        tools=[expert(back_office)],
        llm=ScriptedLLM(
            Reply(
                tool_calls=[call("ask", "c1", request="the first slot")], when=said("the first one")
            ),
            Reply("Sure, the second one.", when=said("yeah maybe the second"), hold=replied),
            Reply("You are booked."),
        ),
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await front.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler, inbox="live")
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("the first one", context_id=thread))
    await asyncio.wait_for(entered.wait(), 20)
    second = asyncio.create_task(desk.send("yeah maybe the second", context_id=thread))
    await listed(desk, thread, 3)  # the two messages and the expert's task
    go.set()

    # while the message's reply is under way the question waits in line, unshown
    async def queued() -> dict[str, Any] | None:
        tasks = await desk.list_tasks(thread)
        return next((t for t in tasks if metadata_of(t).get(QUEUED)), None)

    waiting = await until(queued, "the question queued")
    assert not first.done()
    replied.set()
    cut, answered = await asyncio.wait_for(asyncio.gather(first, second), 20)

    # the message is a task of its own; the question is asked after its reply, and its answer
    # goes to the expert the normal way
    assert answered.task["id"] != cut.task["id"] and answered.answer == "Sure, the second one."
    assert cut.task["id"] == waiting["id"]
    assert cut.state == "input_required" and cut.status_text == question
    done = await asyncio.wait_for(desk.send("a@b.c", context_id=thread), 20)
    assert done.task["id"] == cut.task["id"] and done.answer == "You are booked."
    assert len(seen) == 2 and seen[-1] == "a@b.c"


@pytest.mark.rule("B40")
@pytest.mark.xfail(
    strict=True,
    reason="B40: a resubscribe replays from the start; the wire carries no stream offsets to resume after",
)
async def test_a_continuation_whose_stream_drops_reads_the_answer(
    agentdb_storage: AgentDBStorage,
) -> None:
    seen: list[str] = []
    server = AgentServer(storage=agentdb_storage, agent_name="expert")
    server.a2a_session(endpoint="expert", description="Books seats.")(
        asking_expert("Which date?", "Seat booked.", seen)
    )
    transport = _Dropping(server.http.app, drops=0)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://experts.test", timeout=30.0
    ) as http:
        delegate = A2ADelegate("http://experts.test/expert", httpx_client=http)
        thread = (await agentdb_storage.threads.create_thread()).thread_id
        asked = await delegate.run("a seat", context_id=thread)
        # the answer's stream breaks after its acknowledgment, and is read again
        transport.drops = 1
        answered = await delegate.run("friday", context_id=thread, task_id=asked.task_id)
        await delegate.aclose()

    assert asked.input_required == "Which date?"
    assert answered.state == "completed" and answered.output == "Seat booked."
