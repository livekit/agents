"""A busy agent in text (design §2, §7 "When the agent is busy"): under the ``live`` inbox a message
during a turn interrupts it or waits, every task answers on its own stream, and work in the
background is delivered when no turn runs, as on a call. Seen on the A2A wire and in what reaches
the model."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import pytest

from livekit.agents import Agent, AgentServer, InputRequired, RunContext
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.codec import metadata_from_wire
from livekit.agents.a2a.extension import INTERRUPTED, MERGED_INTO
from livekit.agents.delegation import A2ADelegate
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    FunctionCall,
    FunctionCallOutput,
    FunctionToolCall,
)
from livekit.agents.llm.tool_context import (
    CUT_OFF_KEY,
    CUT_OFF_REPLY,
    DuplicateMode,
    ToolFlag,
    note_cut_off_replies,
)
from livekit.agents.storage import AgentDBStorage
from livekit.agents.types import NOT_GIVEN

from . import fake_agentstore
from .a2a_driver import (
    API_KEY,
    API_SECRET,
    EXPERT,
    Door,
    Reply,
    ScriptedLLM,
    Sent,
    call,
    no_project_env,  # noqa: F401  (autouse: the demo environment must not shape the cards)
    output_of,
    rpc,
    state_of,
    wire,
)

pytestmark = [pytest.mark.unit]


def metadata_of(wire_object: dict[str, Any]) -> dict[str, Any]:
    """A task's or an event's metadata, short keys."""
    return dict(metadata_from_wire(wire_object.get("metadata")))


NO_INTERRUPT: dict[str, Any] = {"livekit/interrupt": False}
DELEGATION: dict[str, Any] = {"livekit/kind": "delegation"}
BACKGROUND = "livekit/background"


def said(text: str) -> Callable[[ChatContext], bool]:
    """A request whose last item is the person's message ``text``."""

    def matches(ctx: ChatContext) -> bool:
        last = ctx.items[-1] if ctx.items else None
        return isinstance(last, ChatMessage) and last.role == "user" and last.text_content == text

    return matches


def told(*call_ids: str) -> Callable[[ChatContext], bool]:
    """A request that delivers the outputs of ``call_ids``, a background tool's progress
    (``b1_update_1``) or end (``b1_final``), with no reply after them yet."""

    def matches(ctx: ChatContext) -> bool:
        outputs = [n for n, i in enumerate(ctx.items) if i.type == "function_call_output"]
        delivered = {ctx.items[n].call_id for n in outputs}  # type: ignore[union-attr]
        later = ctx.items[outputs[-1] + 1 :] if outputs else []
        return set(call_ids) <= delivered and not any(
            getattr(i, "role", None) == "assistant" for i in later
        )

    return matches


def progress(sent: Sent) -> list[str]:
    """The progress texts a response carried, in order."""
    return [
        f["statusUpdate"]["status"]["message"]["parts"][0]["text"]
        for f in sent.frames
        if state_of(f) == "TASK_STATE_WORKING"
        and "message" in f["statusUpdate"]["status"]
        and not metadata_of(f["statusUpdate"]).get(BACKGROUND)
    ]


T = TypeVar("T")


async def until(check: Callable[[], Awaitable[T | None]], what: str) -> T:
    """Polls ``check`` until it returns something truthy, and returns that."""
    for _ in range(400):
        if found := await check():
            return found
        await asyncio.sleep(0.05)
    raise AssertionError(f"never {what}")


async def listed(desk: Door, thread: str, count: int) -> None:
    """Waits until the thread holds ``count`` tasks: the later messages are in, and wait."""

    async def held() -> bool:
        return len(await desk.list_tasks(thread)) >= count

    await until(held, f"{count} tasks")


class Busy:
    """A handler whose first message holds the floor until ``gate``; every later turn answers
    with the texts of the messages it took."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()
        self.batches: list[list[str]] = []

    async def __call__(self, ctx: A2ASessionContext[Any]) -> str:
        texts = [str(m.parts[0]["text"]) for m in ctx.messages]
        self.batches.append(texts)
        if len(self.batches) == 1:
            self.entered.set()
            await self.gate.wait()

            async def reply() -> AsyncIterator[str]:
                yield "a long answer"

            return await ctx.stream(reply())
        await ctx.update("looking")
        return "seen " + " + ".join(texts)

    async def send(self, desk: Door, first: str, *later: str, **flags: Any) -> list[Sent]:
        """Sends ``first``, then each of ``later`` with ``flags`` while it is answered."""
        thread = await desk.thread()
        sends = [asyncio.create_task(desk.send(first, context_id=thread))]
        await asyncio.wait_for(self.entered.wait(), 20)
        for count, text in enumerate(later, 2):
            sends.append(asyncio.create_task(desk.send(text, context_id=thread, **flags)))
            await listed(desk, thread, count)
        self.gate.set()
        return list(await asyncio.wait_for(asyncio.gather(*sends), 30))


# ---------------------------------------------------------------------------------------------
# a message during a turn interrupts it, or waits
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R5")
async def test_a_message_interrupts_the_reply_by_default(door: Callable[..., Door]) -> None:
    busy = Busy()
    stopped, ran = await busy.send(door(busy, inbox="live"), "tell me everything", "never mind")

    assert stopped.state == "canceled"
    assert stopped.metadata[INTERRUPTED] is True
    assert ran.state == "completed"
    assert ran.answer == "seen never mind"


@pytest.mark.rule("R5", "R10")
async def test_messages_that_do_not_interrupt_wait_and_run_as_one_turn(
    door: Callable[..., Door],
) -> None:
    busy = Busy()
    desk = door(busy, inbox="live")
    one, two, three = await busy.send(desk, "one", "two", "three", **NO_INTERRUPT)

    # the reply was not stopped, and the two that waited ran as one turn, with no debounce
    assert one.state == "completed" and one.answer == "a long answer"
    assert busy.batches == [["one"], ["two", "three"]]
    assert two.state == "completed"
    assert two.metadata[MERGED_INTO] == three.task["id"]
    assert three.state == "completed" and three.answer == "seen two + three"


@pytest.mark.rule("R5")
async def test_only_the_live_inbox_reads_the_interrupt_flag(door: Callable[..., Door]) -> None:
    busy = Busy()
    desk = door(busy, inbox="interrupt")
    stopped, ran = await busy.send(desk, "tell me everything", "never mind", **NO_INTERRUPT)

    assert stopped.state == "canceled"
    assert ran.state == "completed"


@pytest.mark.rule("R5")
async def test_a_delegation_never_interrupts_the_experts_reply(
    door: Callable[..., Door], agentdb_storage: AgentDBStorage
) -> None:
    busy = Busy()
    server = AgentServer(
        storage=agentdb_storage, agent_name=EXPERT, api_key=API_KEY, api_secret=API_SECRET
    )
    server.a2a_session(endpoint=EXPERT, description="Knows the fares.", inbox="live")(busy)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http.app),
        base_url="http://experts.test",
        timeout=30.0,
    ) as http:
        delegate = A2ADelegate(f"http://experts.test/{EXPERT}", httpx_client=http)
        model = ScriptedLLM(
            Reply(tool_calls=[call("ask", "c1", request="the fare to Oslo")]),
            Reply("It is 120 euros.", when=output_of("c1")),
        )
        front = Agent(instructions="front desk", llm=model, tools=[delegate])

        async def front_desk(ctx: A2ASessionContext[Any]) -> Any:
            return (await front.run(ctx.message, message_history=ctx.message_history)).output

        desk = door(front_desk)
        thread = await desk.thread()
        # the person talks to the expert too, and its reply is under way
        direct = asyncio.create_task(
            http.post(
                f"/{EXPERT}/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "message/send",
                    "params": {"message": wire("the whole fare table", "d1", context_id=thread)},
                },
            )
        )
        await asyncio.wait_for(busy.entered.wait(), 20)
        asked = asyncio.create_task(desk.send("fare to Oslo?", context_id=thread))
        # the person's task, the expert's reply and the delegation
        await listed(desk, thread, 3)
        busy.gate.set()
        reply, sent = await asyncio.wait_for(asyncio.gather(direct, asked), 30)
        await delegate.aclose()

    assert reply.json()["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    assert busy.batches == [["the whole fare table"], ["the fare to Oslo"]]
    assert sent.answer == "It is 120 euros."


# ---------------------------------------------------------------------------------------------
# each task answers on its own stream
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R2")
async def test_a_message_that_waited_answers_on_its_own_stream(door: Callable[..., Door]) -> None:
    busy = Busy()
    _, waited = await busy.send(door(busy, inbox="live"), "one", "two", **NO_INTERRUPT)

    # the holder ran it, and its progress, its answer and its end came on its own response
    frames = waited.frames[1:]
    assert progress(waited) == ["looking"]
    assert waited.answer == "seen two"
    assert state_of(frames[-1]) == "TASK_STATE_COMPLETED"
    task_ids = {(f.get("statusUpdate") or f["artifactUpdate"])["taskId"] for f in frames}
    assert task_ids == {waited.task["id"]}


@pytest.mark.rule("R2")
async def test_an_answer_continues_its_task_on_the_tasks_own_stream(
    door: Callable[..., Door],
) -> None:
    model = ScriptedLLM(
        Reply(tool_calls=[call("book", "b1")]),
        Reply(tool_calls=[call("book", "b2", email="ana@example.com")]),
        Reply("Booked.", when=output_of("b2")),
    )
    agent = Agent(instructions="bookings", llm=model)

    @agent.tool
    async def book(ctx: RunContext, email: str = "") -> str | InputRequired:
        """Book the visit.

        Args:
            email: the person's email, when they gave it
        """
        return f"booked for {email}" if email else InputRequired("What email?", verbatim=True)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler, inbox="live")
    thread = await desk.thread()
    asked = await desk.send("book me in", context_id=thread)
    assert asked.state == "input_required"

    answered = await desk.send("ana@example.com", context_id=thread)
    assert answered.task["id"] == asked.task["id"]
    assert answered.state == "completed" and answered.answer == "Booked."
    # the response carries the continuation only, not the question the stream held before it
    assert [state_of(f) for f in answered.frames[1:] if "statusUpdate" in f] == [
        "TASK_STATE_COMPLETED"
    ]


@pytest.mark.rule("R2")
async def test_a_response_on_the_writers_worker_reads_nothing_from_the_store(
    door: Callable[..., Door],
) -> None:
    busy = Busy()
    busy.gate.set()
    sent = await door(busy, inbox="live").send("one")

    assert sent.answer == "a long answer"
    assert fake_agentstore.served[sent.task["id"]] == 0


@pytest.mark.rule("R2")
async def test_a_response_on_another_worker_gets_every_frame(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    busy = Busy()
    holder, other = (Door(storage, busy, inbox="live") for storage in agentdb_workers)
    try:
        thread = await holder.thread()
        first = asyncio.create_task(holder.send("one", context_id=thread))
        await asyncio.wait_for(busy.entered.wait(), 20)
        waited = asyncio.create_task(other.send("two", context_id=thread, **NO_INTERRUPT))
        await listed(holder, thread, 2)
        busy.gate.set()
        _, sent = await asyncio.wait_for(asyncio.gather(first, waited), 30)
    finally:
        await holder.http.aclose()
        await other.http.aclose()

    # the holder ran it, and its frames reached the other worker through the store
    assert progress(sent) == ["looking"] and sent.answer == "seen two"
    assert state_of(sent.frames[-1]) == "TASK_STATE_COMPLETED"


# ---------------------------------------------------------------------------------------------
# a tool that reports progress goes on in the background, and its end is delivered later
# ---------------------------------------------------------------------------------------------


@dataclass
class Booking:
    booked: str = ""


class Bookings:
    """A live desk: ``book`` reports progress (``silent`` to the model alone), waits for
    ``release[day]``, then books or asks ``asks[day]``; ``look`` waits silently; "one moment"
    holds the lane until ``gate``. A ``delegated`` desk takes every message as a delegation."""

    def __init__(
        self,
        model: ScriptedLLM,
        *days: str,
        asks: dict[str, str] | None = None,
        on_duplicate: DuplicateMode = "allow",
        flags: ToolFlag = ToolFlag.NONE,
        halfway: bool = False,
        silent: bool = False,
        delegated: bool = False,
    ) -> None:
        self.model = model
        self.delegated = delegated
        self.halfway, self.progressed = asyncio.Event(), asyncio.Event()
        self.release = {day: asyncio.Event() for day in days}
        self.ended: list[str] = []
        self.holding, self.gate, self.looking = asyncio.Event(), asyncio.Event(), asyncio.Event()
        self.agent = Agent(instructions="front desk", llm=model, state_type=Booking)
        self.door: Door
        self.thread = ""

        @self.agent.tool(on_duplicate=on_duplicate, flags=flags)
        async def book(ctx: RunContext[None, Booking], day: str) -> str | InputRequired:
            """Book a cleaning.

            Args:
                day: the day
            """
            await ctx.update("booking now", silent=silent)
            if halfway:
                await self.halfway.wait()
                await ctx.update("half way")
                self.progressed.set()
            await self.release[day].wait()
            self.ended.append(day)
            if asks and day in asks:
                return InputRequired(asks[day], verbatim=True)
            ctx.state.booked = day
            return f"booked {day}"

        @self.agent.tool(flags=flags)
        async def look(ctx: RunContext[None, Booking], day: str) -> str:
            """Look for a free slot.

            Args:
                day: the day
            """
            self.looking.set()
            await self.release[day].wait()
            return f"free on {day}"

    async def __call__(self, ctx: A2ASessionContext[Booking]) -> Any:
        if ctx.message.text_content == "what did I book?":
            return f"booked: {ctx.state.booked}"
        if ctx.message.text_content == "one moment":
            self.holding.set()
            await self.gate.wait()
            return "Here."
        return await self.agent.run(
            ctx.message, message_history=ctx.message_history, state=ctx.state
        )

    def send(self, text: str) -> asyncio.Task[Sent]:
        metadata = DELEGATION if self.delegated else {}
        return asyncio.create_task(self.door.send(text, context_id=self.thread, **metadata))

    async def in_the_background(self, count: int = 1) -> list[dict[str, Any]]:
        return await in_the_background(self.door, self.thread, count)

    def outputs(self) -> dict[str, FunctionCallOutput]:
        """Every call output the model read, by call id."""
        return {
            i.call_id: i
            for r in self.model.requests
            for i in r.items
            if isinstance(i, FunctionCallOutput)
        }


async def in_the_background(desk: Door, thread: str, count: int = 1) -> list[dict[str, Any]]:
    """Waits until ``count`` of the thread's tasks are ``working`` with work in the background."""

    async def working() -> list[dict[str, Any]] | None:
        tasks = [
            task
            for task in await desk.list_tasks(thread)
            if task["status"]["state"] == "TASK_STATE_WORKING" and metadata_of(task).get(BACKGROUND)
        ]
        return tasks if len(tasks) >= count else None

    return await until(working, f"{count} tasks in the background")


async def bookings(door: Callable[..., Door], *replies: Reply, **options: Any) -> Bookings:
    """A ``Bookings`` desk for ``days`` behind a live door, on a thread of its own."""
    days = options.pop("days", ("monday",))
    desk = Bookings(ScriptedLLM(*replies), *days, **options)
    desk.door = door(desk, inbox="live", state=Booking)
    desk.thread = await desk.door.thread()
    return desk


BOOK_MONDAY = (Reply(tool_calls=[call("book", "b1", day="monday")], when=said("book monday")),)


@pytest.mark.rule("R8", "R9", "R11")
async def test_a_tool_that_reports_progress_runs_on_beside_the_next_message(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply("Reminder set.", when=said("and a reminder")),
        Reply("Booked for Monday.", when=told("b1_final")),
    )
    first = desk.send("book monday")
    # the booking's turn ended on its progress; the task waits for the booking
    (task,) = await desk.in_the_background()

    # the next message runs at once, beside the booking
    second = await asyncio.wait_for(desk.send("and a reminder"), 20)
    assert second.state == "completed" and second.answer == "Reminder set."
    assert not first.done()

    desk.release["monday"].set()
    booked = await asyncio.wait_for(first, 20)
    # the booking's end answered its own task, which said on its stream it went on meanwhile
    assert booked.task["id"] == task["id"]
    assert booked.state == "completed" and booked.answer == "Booked for Monday."
    assert booked.metadata[BACKGROUND] is False
    assert any(
        state_of(f) == "TASK_STATE_WORKING" and metadata_of(f["statusUpdate"]).get(BACKGROUND)
        for f in booked.frames
    )
    # the state the booking wrote after its turn committed is kept
    assert (await desk.send("what did I book?")).answer == "booked: monday"


@pytest.mark.rule("R8", "R28", "R55")
async def test_later_progress_is_sent_as_it_is_and_the_model_reads_it_with_the_end(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply("Booked.", when=told("b1_update_1", "b1_final")),
        halfway=True,
        delegated=True,
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    # the first progress ended the delegated turn with no model request after it
    assert len(desk.model.requests) == 1
    desk.halfway.set()
    await asyncio.wait_for(desk.progressed.wait(), 20)
    desk.release["monday"].set()
    booked = await asyncio.wait_for(first, 20)

    # the progress went out as a working frame, and the model read it only with the end
    assert progress(booked) == ["booking now", "half way"]
    assert booked.answer == "Booked." and len(desk.model.requests) == 2


@pytest.mark.rule("R28", "R58")
async def test_results_that_end_during_a_turn_come_back_in_one_reply_on_each_task(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(tool_calls=[call("book", "b2", day="tuesday")], when=said("and tuesday")),
        Reply("Both are booked.", when=told("b1_final", "b2_final")),
        Reply("You're welcome.", when=said("thanks")),
        days=("monday", "tuesday"),
    )
    monday = desk.send("book monday")
    await desk.in_the_background()
    tuesday = desk.send("and tuesday")
    await desk.in_the_background(2)
    # both end while a turn runs, and wait for it
    busy = desk.send("one moment")
    await asyncio.wait_for(desk.holding.wait(), 20)
    for release in desk.release.values():
        release.set()

    async def both_ended() -> bool:
        return len(desk.ended) == 2

    await until(both_ended, "both ended")
    desk.gate.set()
    sent = await asyncio.wait_for(asyncio.gather(monday, tuesday, busy), 20)

    assert [(s.state, s.answer) for s in sent] == [
        ("completed", "Both are booked."),
        ("completed", "Both are booked."),
        ("completed", "Here."),
    ]
    # as on a call: told last by a system message the history does not keep, free to act on
    # the results
    model = desk.model
    (delivery,) = [r for r in model.requests if told("b1_final", "b2_final")(r)]
    assert model.tool_choices[model.requests.index(delivery)] is NOT_GIVEN
    tail = delivery.items[-1]
    assert isinstance(tail, ChatMessage) and tail.role == "system"
    await desk.send("thanks")
    assert all(
        getattr(i, "text_content", None) != tail.text_content for i in model.requests[-1].items
    )


@pytest.mark.rule("R29")
async def test_a_question_of_work_in_the_background_is_asked_and_its_task_waits(
    door: Callable[..., Door],
) -> None:
    question = "Which time on Monday?"
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(tool_calls=[call("book", "b2", day="tuesday")], when=said("and tuesday")),
        Reply("Tuesday booked.", when=told("b2_final")),
        asks={"monday": question},
        days=("monday", "tuesday"),
    )
    first = desk.send("book monday")
    (task,) = await desk.in_the_background()
    requests = len(desk.model.requests)
    desk.release["monday"].set()
    asked = await asyncio.wait_for(first, 20)

    # as on a call, the question is what the person gets, with no reply generated for it
    assert asked.state == "input_required" and asked.status_text == question
    assert len(desk.model.requests) == requests

    # the next message continues the task; the frame that leaves work running drops the question
    second = desk.send("and tuesday")
    await desk.in_the_background()
    desk.release["tuesday"].set()
    continued = await asyncio.wait_for(second, 20)
    assert continued.task["id"] == task["id"] and continued.answer == "Tuesday booked."
    running = [
        f["statusUpdate"]["status"]
        for f in continued.frames
        if state_of(f) == "TASK_STATE_WORKING" and metadata_of(f["statusUpdate"]).get(BACKGROUND)
    ]
    assert running and all("message" not in s for s in running)


@pytest.mark.rule("R55")
async def test_a_reply_beside_the_call_is_the_only_acknowledgement(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        Reply("On it.", tool_calls=[call("book", "b1", day="monday")]),
        Reply("A second acknowledgement.", when=output_of("b1")),
    )
    first = desk.send("book monday")
    (task,) = await desk.in_the_background()
    assert len(desk.model.requests) == 1
    assert task["status"]["message"]["parts"] == [{"text": "On it."}]
    desk.release["monday"].set()
    await asyncio.wait_for(first, 20)


@pytest.mark.rule("R55")
async def test_a_tool_that_ends_right_after_its_progress_is_called_once(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door, *BOOK_MONDAY, Reply("Booked.", when=told("b1_final")), delegated=True
    )
    desk.release["monday"].set()
    booked = await asyncio.wait_for(desk.send("book monday"), 20)

    # the progress ends the turn; the end is delivered once the lane is idle
    assert desk.ended == ["monday"]
    assert booked.state == "completed" and booked.answer == "Booked."
    assert desk.model.tool_choices == [NOT_GIVEN, NOT_GIVEN]


@pytest.mark.rule("R55", "R57")
async def test_the_person_gets_the_reply_to_a_report(door: Callable[..., Door]) -> None:
    """A turn facing the person words its progress, in a reply that calls no tools."""
    desk = await bookings(door, *BOOK_MONDAY, Reply("Booking Monday now.", when=output_of("b1")))
    first = desk.send("book monday")
    (task,) = await desk.in_the_background()

    assert task["status"]["message"]["parts"] == [{"text": "Booking Monday now."}]
    assert desk.model.tool_choices == [NOT_GIVEN, "none"]
    desk.release["monday"].set()
    await asyncio.wait_for(first, 20)


@pytest.mark.rule("R57")
async def test_a_silent_update_reaches_only_the_model(door: Callable[..., Door]) -> None:
    desk = await bookings(door, *BOOK_MONDAY, Reply("Booked.", when=told("b1_final")), silent=True)
    first = desk.send("book monday")
    (task,) = await desk.in_the_background()
    assert "message" not in task["status"] and len(desk.model.requests) == 1
    desk.release["monday"].set()
    booked = await asyncio.wait_for(first, 20)

    assert progress(booked) == []
    assert "booking now" in desk.outputs()["b1"].output


@pytest.mark.rule("R30", "B25")
async def test_a_duplicate_of_a_call_in_the_background_follows_the_tools_policy(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(tool_calls=[call("book", "b2", day="monday")], when=said("book it again")),
        Reply("It is still being booked.", when=output_of("b2")),
        on_duplicate="reject",
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    again = await asyncio.wait_for(desk.send("book it again"), 20)
    desk.release["monday"].set()
    await asyncio.wait_for(first, 20)

    assert again.answer == "It is still being booked."
    refused = desk.outputs()["b2"]
    # as on a call, the refusal is what the model reads, and names no tool to cancel with
    assert "already running" in refused.output and "lk_agents" not in refused.output


@pytest.mark.rule("R30")
@pytest.mark.parametrize("cancellable", [False, True])
async def test_a_replace_cancels_the_running_call_only_when_it_is_cancellable(
    door: Callable[..., Door], cancellable: bool
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(tool_calls=[call("book", "b2", day="monday")], when=said("book it again")),
        Reply("Again.", when=output_of("b2")),
        on_duplicate="replace",
        flags=ToolFlag.CANCELLABLE if cancellable else ToolFlag.NONE,
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    again = desk.send("book it again")

    async def decided() -> bool:
        # the refusal is read at once; a replaced first call ends its task
        return bool({"b2", "b1_final"} & desk.outputs().keys())

    await until(decided, "the duplicate was decided")
    desk.release["monday"].set()
    await asyncio.wait_for(asyncio.gather(first, again), 20)

    # without the flag the duplicate is refused; with it the first is cancelled, and says so
    outputs = desk.outputs()
    assert outputs["b2"].is_error is not cancellable
    assert outputs["b1_final"].is_error is cancellable


@pytest.mark.rule("B53")
async def test_a_cancellable_tool_in_the_background_can_be_cancelled(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(
            tool_calls=[
                call("lk_agents_get_running_tasks", "g1"),
                FunctionToolCall(
                    name="lk_agents_cancel_task", arguments='{"call_id": "b1"}', call_id="x1"
                ),
            ],
            when=said("cancel it"),
        ),
        Reply("Cancelled.", when=output_of("x1")),
        Reply("The booking was cancelled.", when=told("b1_final")),
        flags=ToolFlag.CANCELLABLE,
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    await asyncio.wait_for(desk.send("cancel it"), 20)
    ended = await asyncio.wait_for(first, 20)

    # as on a call the model lists and cancels the work, and its task still hears it ended
    outputs = desk.outputs()
    assert "'b1'" in outputs["g1"].output and not outputs["x1"].is_error
    assert outputs["b1_final"].is_error
    assert ended.state == "completed" and ended.answer == "The booking was cancelled."


# ---------------------------------------------------------------------------------------------
# stopping: an interrupt and a client's cancel
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("R6")
async def test_a_message_stops_the_reply_while_it_streams(door: Callable[..., Door]) -> None:
    held = asyncio.Event()
    desk = await bookings(
        door,
        Reply("Here is everything", hold=held, when=said("tell me everything")),
        Reply("Fine.", when=said("never mind")),
    )
    first = desk.send("tell me everything")

    async def streaming() -> bool:
        return len(desk.model.requests) == 1

    await until(streaming, "the reply streams")
    ran = await asyncio.wait_for(desk.send("never mind"), 20)
    stopped = await asyncio.wait_for(first, 20)
    held.set()

    # the task ends with what was shown of the reply as its answer
    assert stopped.state == "canceled" and stopped.metadata[INTERRUPTED] is True
    assert stopped.answer == "Here is everything"
    assert ran.answer == "Fine."


def cut_off(ctx: ChatContext, call_id: str) -> bool:
    """The request notes that the reply to ``call_id``'s output was cut off."""
    out = next(i for i in ctx.items if isinstance(i, FunctionCallOutput) and i.call_id == call_id)
    return out.output.endswith(CUT_OFF_REPLY)


@pytest.mark.rule("R68", "B57")
def test_only_the_newest_message_reads_the_note_on_a_cut_off_reply() -> None:
    """The output of a flagged call gets the note on a copy, until the person speaks again."""
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="book monday")
    ctx.insert(FunctionCall(call_id="b1", name="book", arguments="{}", extra={CUT_OFF_KEY: "true"}))
    stored = FunctionCallOutput(call_id="b1", name="book", output="booked", is_error=False)
    ctx.insert(stored)
    ctx.add_message(role="user", content="are you there")
    request = ctx.copy()
    note_cut_off_replies(request)
    assert cut_off(request, "b1") and stored.output == "booked"

    ctx.add_message(role="user", content="thanks")
    request = ctx.copy()
    note_cut_off_replies(request)
    assert not cut_off(request, "b1")


@pytest.mark.rule("R6", "R68", "B57")
async def test_a_reply_cut_off_before_it_showed_leaves_no_message(
    door: Callable[..., Door],
) -> None:
    held = asyncio.Event()
    desk = await bookings(
        door,
        Reply(hold=held, when=said("tell me everything")),
        Reply("Fine.", when=said("never mind")),
    )
    first = desk.send("tell me everything")

    async def streaming() -> bool:
        return len(desk.model.requests) == 1

    await until(streaming, "the reply streams")
    await asyncio.wait_for(desk.send("never mind"), 20)
    held.set()
    await asyncio.wait_for(first, 20)

    # the person's two messages, one after the other: nothing was hidden from them
    last_two = desk.model.requests[-1].items[-2:]
    assert [(i.role, i.text_content) for i in last_two] == [  # type: ignore[union-attr]
        ("user", "tell me everything"),
        ("user", "never mind"),
    ]


@pytest.mark.rule("R6", "R68", "B57")
async def test_a_reply_to_a_result_cut_off_while_it_streams_is_noted_on_the_result(
    door: Callable[..., Door],
) -> None:
    held = asyncio.Event()
    desk = await bookings(
        door,
        *BOOK_MONDAY,
        Reply(hold=held, when=told("b1_final")),
        Reply("Fine.", when=said("never mind")),
        delegated=True,
    )
    first = desk.send("book monday")
    await desk.in_the_background()
    desk.release["monday"].set()

    async def replying() -> bool:
        return len(desk.model.requests) == 2

    await until(replying, "the delivery's reply streams")
    await asyncio.wait_for(desk.send("never mind"), 20)
    held.set()
    await asyncio.wait_for(first, 20)

    # the next turn's model reads that the person was not told the result
    assert cut_off(desk.model.requests[-1], "b1_final")


@pytest.mark.rule("R6", "R68", "B57")
async def test_a_reply_to_a_result_cut_off_before_it_began_is_noted_on_the_result(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        Reply(tool_calls=[call("look", "l1", day="monday")], when=said("look monday")),
        Reply("Fine.", when=said("never mind")),
    )
    first = desk.send("look monday")
    await asyncio.wait_for(desk.looking.wait(), 20)
    second = desk.send("never mind")
    await listed(desk.door, desk.thread, 2)
    desk.release["monday"].set()
    await asyncio.wait_for(asyncio.gather(first, second), 20)

    # the result came back as the next message waited: its turn stopped before the reply
    assert cut_off(desk.model.requests[-1], "l1")


@pytest.mark.rule("R7", "B27")
@pytest.mark.parametrize("cancellable", [False, True])
async def test_an_interrupt_stops_only_a_cancellable_tool(
    door: Callable[..., Door], cancellable: bool
) -> None:
    desk = await bookings(
        door,
        Reply(tool_calls=[call("look", "l1", day="monday")], when=said("look monday")),
        Reply("Fine.", when=said("never mind")),
        flags=ToolFlag.CANCELLABLE if cancellable else ToolFlag.NONE,
    )
    first = desk.send("look monday")
    await asyncio.wait_for(desk.looking.wait(), 20)
    second = desk.send("never mind")
    await listed(desk.door, desk.thread, 2)
    if not cancellable:
        desk.release["monday"].set()  # it runs to its end
    stopped, ran = await asyncio.wait_for(asyncio.gather(first, second), 20)

    assert stopped.state == "canceled" and ran.answer == "Fine."
    assert desk.outputs()["l1"].is_error is cancellable


@pytest.mark.rule("R7")
async def test_an_interrupted_task_waits_for_its_work_in_the_background(
    door: Callable[..., Door],
) -> None:
    desk = await bookings(
        door,
        Reply(
            tool_calls=[call("book", "b1", day="monday"), call("look", "l1", day="tuesday")],
            when=said("book monday"),
        ),
        Reply("Fine.", when=said("never mind")),
        Reply("Booked.", when=told("b1_final")),
        days=("monday", "tuesday"),
    )
    first = desk.send("book monday")
    # the booking goes to the background while the look holds the turn
    await asyncio.wait_for(desk.looking.wait(), 20)
    second = desk.send("never mind")
    await listed(desk.door, desk.thread, 2)
    desk.release["tuesday"].set()
    ran = await asyncio.wait_for(second, 20)
    await desk.in_the_background()
    desk.release["monday"].set()
    booked = await asyncio.wait_for(first, 20)

    assert ran.answer == "Fine."
    assert booked.state == "completed" and booked.answer == "Booked."


@pytest.mark.rule("R12", "B27")
async def test_a_cancel_stops_the_running_tool_after_its_protected_step(
    door: Callable[..., Door],
) -> None:
    model = ScriptedLLM(Reply(tool_calls=[call("pay", "p1")]))
    agent = Agent(instructions="front desk", llm=model)
    entered, charged, never = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @agent.tool
    async def pay(ctx: RunContext) -> str:
        """Charge the card."""
        with ctx.uninterruptible():
            entered.set()
            await charged.wait()
        await never.wait()
        return "paid"

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await agent.run(ctx.message, message_history=ctx.message_history)

    desk = door(handler)
    thread = await desk.thread()
    first = asyncio.create_task(desk.send("pay", context_id=thread))
    await asyncio.wait_for(entered.wait(), 20)
    (task,) = await desk.list_tasks(thread)
    await desk.cancel(task["id"], thread)
    # the task runs on until its protected step ended
    await asyncio.sleep(0.5)
    assert not first.done()
    charged.set()
    stopped = await asyncio.wait_for(first, 20)

    said = str(stopped.status_text)
    assert stopped.state == "canceled" and said.startswith("stopped after:") and "pay" in said


@pytest.mark.rule("R12")
async def test_a_cancel_stops_work_in_the_background(door: Callable[..., Door]) -> None:
    desk = await bookings(door, *BOOK_MONDAY, delegated=True)
    first = desk.send("book monday")
    (task,) = await desk.in_the_background()
    answered = (await desk.door.cancel(task["id"], desk.thread))["result"]
    stopped = await asyncio.wait_for(first, 20)

    # the call answers at once, the request stays off the wire, and the task ends on its stream
    assert answered["status"]["state"] == "TASK_STATE_WORKING"
    assert not any("cancel" in key for f in [answered, stopped.task] for key in metadata_of(f))
    assert stopped.state == "canceled" and str(stopped.status_text).startswith("stopped after:")
    # the work stopped, and no reply was made for it
    assert desk.ended == [] and len(desk.model.requests) == 1


@pytest.mark.rule("R13")
async def test_a_cancel_on_an_ended_task_changes_nothing(door: Callable[..., Door]) -> None:
    desk = await bookings(door, Reply("Hello.", when=said("hi")))
    done = await asyncio.wait_for(desk.send("hi"), 20)
    refused = await desk.door.cancel(done.task["id"], desk.thread)

    assert refused["error"]["code"] == -32002  # TaskNotCancelableError
    (task,) = await desk.door.list_tasks(desk.thread)
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"


# ---------------------------------------------------------------------------------------------
# a request that fails outside a turn
# ---------------------------------------------------------------------------------------------


@pytest.mark.rule("B49")
async def test_a_request_that_fails_answers_a_fixed_sentence(door: Callable[..., Door]) -> None:
    desk = door()
    thread = await desk.thread()
    response = await rpc(desk.http, "tasks/list", {"contextId": thread, "pageSize": "a few"})
    error = response.json()["error"]
    assert error["message"] == "the request failed before it finished"
    assert "ValueError" not in response.text and "a few" not in response.text
