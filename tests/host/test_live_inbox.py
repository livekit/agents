"""An `inbox="live"` endpoint: one owner keeps a held tool body running while later messages get
their own tasks and answers, and the body's result returns to the task that started it."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field

import httpx

from livekit.agents import Agent, APIStatusError, InputRequired, RunContext
from livekit.agents._generated.a2a_metadata import KIND, MERGED_INTO
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.llm import ChatContext
from livekit.agents.storage import ContextStore
from tests.fakes import FakeLLM, Hang, Reply, tool_call, until

from .a2a_door import message_send

Door = Callable[..., AbstractAsyncContextManager[httpx.AsyncClient]]


@dataclass
class BookingService:
    """The business system the held tool calls: it answers only once released."""

    gate: asyncio.Event = field(default_factory=asyncio.Event)
    started: list[str | None] = field(default_factory=list)
    effects: list[str | None] = field(default_factory=list)
    acknowledged: asyncio.Event = field(default_factory=asyncio.Event)
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    pause: Hang = field(default_factory=Hang)


def front_desk(service: BookingService) -> Agent:
    def reply(chat_ctx: ChatContext) -> Reply:
        last = chat_ctx.items[-1]
        if last.type == "message" and last.role == "user":
            return {
                "Reserve a room.": tool_call("reserve"),
                "When is check-in?": "Check-in is at 3pm.",
                "Book a taxi.": tool_call("taxi"),
                "Which room?": tool_call("choose"),
                "The double.": "The double it is.",
                "Is breakfast included?": [service.pause, "Breakfast is included."],
                "Book a taxi.\nWhen is check-in?": "Your taxi is booked and check-in is at 3pm.",
                "Cancel it.\nCancel all.": APIStatusError("refused", status_code=400),
            }[last.text_content or ""]
        if last.type == "function_call_output" and last.output == "Taxi T1":
            return "Your taxi is T1."
        # A delivery turn finds the held call's output in place, where the call was made.
        if any(
            i.type == "function_call_output" and i.output == "Reservation R1"
            for i in chat_ctx.items
        ):
            return "Booked reservation R1."
        service.acknowledged.set()
        return "I'm checking availability."

    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM(reply))

    @agent.tool
    async def reserve(ctx: RunContext) -> str:
        """Reserve the room."""
        service.started.append(ctx.task_id)
        await ctx.update("Checking the booking service.")
        service.updated.set()
        await service.gate.wait()
        service.effects.append(ctx.task_id)
        return "Reservation R1"

    @agent.tool
    async def taxi(ctx: RunContext) -> str:
        """Book a taxi."""
        return "Taxi T1"

    @agent.tool
    async def choose(ctx: RunContext) -> InputRequired:
        """Ask which room."""
        return InputRequired("Single or double?")

    return agent


def send(
    text: str,
    message_id: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
    kind: str | None = None,
) -> dict[str, object]:
    body = message_send(text, message_id, context_id=context_id)
    message = body["params"]["message"]  # type: ignore[index]
    if task_id is not None:
        message["taskId"] = task_id
    if kind is not None:
        message["metadata"] = {KIND: kind}
    return body


@asynccontextmanager
async def streaming(door: httpx.AsyncClient, body: dict[str, object]) -> AsyncIterator[Stream]:
    stream = Stream(door, body | {"method": "message/stream"})
    stream.reader = asyncio.create_task(stream.read())
    try:
        yield stream
    finally:
        stream.reader.cancel()
        await asyncio.gather(stream.reader, return_exceptions=True)


class Stream:
    """A ``message/stream`` read in the background, its events kept as they arrive."""

    def __init__(self, door: httpx.AsyncClient, body: dict[str, object]) -> None:
        self.door, self.body = door, body
        self.events: list[dict[str, object]] = []
        self.reader: asyncio.Task[None]

    async def read(self) -> None:
        async with self.door.stream("POST", "/v1", json=self.body) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    self.events.append(json.loads(line.removeprefix("data:"))["result"])

    @property
    def task(self) -> dict[str, object]:
        return self.events[0]["task"]  # type: ignore[return-value]

    def progress(self) -> list[str]:
        return [
            e["statusUpdate"]["status"]["message"]["parts"][0]["text"]  # type: ignore[index]
            for e in self.events
            if "statusUpdate" in e
            and e["statusUpdate"]["status"]["state"] == "TASK_STATE_WORKING"  # type: ignore[index]
            and "message" in e["statusUpdate"]["status"]  # type: ignore[index, operator]
        ]

    def final(self) -> dict[str, object]:
        return self.events[-1]["statusUpdate"]  # type: ignore[return-value]

    def answer(self) -> list[str]:
        return [
            part["text"]
            for e in self.events
            if "artifactUpdate" in e
            for part in e["artifactUpdate"]["artifact"]["parts"]  # type: ignore[index]
            if "text" in part
        ]


async def get_task(door: httpx.AsyncClient, task_id: str, context_id: str) -> dict[str, object]:
    body = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tasks/get",
        "params": {"id": task_id, "contextId": context_id},
    }
    return (await door.post("/v1", json=body)).json()["result"]  # type: ignore[no-any-return]


def handler_for(agent: Agent) -> Callable[[A2ASessionContext], object]:
    async def handler(ctx: A2ASessionContext) -> object:
        return await agent.run(ctx.message, context=ctx.store.bind())

    return handler


async def test_a_held_tool_keeps_its_task_while_a_side_question_and_another_task_complete(
    a2a_door: Door,
) -> None:
    """The exit story of a live backend: the booking body is held at its service, a side
    question and an independent taxi booking each get their own task and answer, then the
    release delivers one reservation to the booking task alone."""
    service = BookingService()
    async with (
        a2a_door(handler_for(front_desk(service)), inbox="live") as door,
        streaming(door, send("Reserve a room.", "m-booking")) as booking,
    ):
        await until(lambda: booking.progress(), "the held tool's progress was not streamed")
        booking_task, thread = booking.task["id"], booking.task["contextId"]
        assert booking.progress() == ["Checking the booking service."]
        assert service.started == [booking_task]

        side = (
            await door.post("/v1", json=send("When is check-in?", "m-side", context_id=thread))
        ).json()
        taxi = (
            await door.post("/v1", json=send("Book a taxi.", "m-taxi", context_id=thread))
        ).json()

        side_task, taxi_task = side["result"]["task"], taxi["result"]["task"]
        assert side_task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert side_task["artifacts"][0]["parts"][0]["text"] == "Check-in is at 3pm."
        assert taxi_task["status"]["state"] == "TASK_STATE_COMPLETED"
        assert taxi_task["artifacts"][0]["parts"][0]["text"] == "Your taxi is T1."
        assert len({booking_task, side_task["id"], taxi_task["id"]}) == 3
        assert service.effects == [], "the service is still held"
        held = await get_task(door, booking_task, thread)
        assert held["status"]["state"] == "TASK_STATE_WORKING"

        service.gate.set()
        await until(lambda: booking.reader.done(), "the booking task did not settle on release")

    assert booking.final()["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "Booked reservation R1." in "".join(booking.answer())
    assert service.started == [booking_task]
    assert service.effects == [booking_task], "one business effect, on the booking's task"
    for task in (side_task, taxi_task):
        assert MERGED_INTO not in json.dumps(task)


async def test_only_an_explicit_task_id_continues_a_task_on_a_live_endpoint(
    a2a_door: Door,
) -> None:
    """A message without `taskId` is new work for the current agent with its own task, even
    while another task waits for input; the question is answered only by naming its task."""
    service = BookingService()
    async with a2a_door(handler_for(front_desk(service)), inbox="live") as door:
        asked = (await door.post("/v1", json=send("Which room?", "m-ask"))).json()["result"]["task"]
        thread = asked["contextId"]
        assert asked["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

        unaddressed = (
            await door.post("/v1", json=send("When is check-in?", "m-new", context_id=thread))
        ).json()["result"]["task"]
        assert unaddressed["id"] != asked["id"]
        assert unaddressed["status"]["state"] == "TASK_STATE_COMPLETED"
        assert unaddressed["artifacts"][0]["parts"][0]["text"] == "Check-in is at 3pm."
        still = await get_task(door, asked["id"], thread)
        assert still["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"

        answered = (
            await door.post(
                "/v1",
                json=send("The double.", "m-answer", context_id=thread, task_id=asked["id"]),
            )
        ).json()["result"]["task"]

    assert answered["id"] == asked["id"]
    assert answered["status"]["state"] == "TASK_STATE_COMPLETED"
    assert answered["artifacts"][0]["parts"][0]["text"] == "The double it is."
    for task in (asked, unaddressed, answered):
        assert MERGED_INTO not in json.dumps(task)


async def test_a_per_channel_map_runs_delegation_live_and_batches_direct_messages(
    a2a_door: Door,
) -> None:
    """`inbox={"delegation": "live", "a2a": "batch"}`: delegated messages each keep their own
    task under the live owner, while a burst of direct messages is answered as one turn, the
    earlier task pointing at the later with `mergedInto`, all before the held booking ends."""
    service = BookingService()
    async with (
        a2a_door(
            handler_for(front_desk(service)), inbox={"delegation": "live", "a2a": "batch"}
        ) as door,
        streaming(door, send("Reserve a room.", "m-booking", kind="delegation")) as booking,
    ):
        await until(lambda: booking.progress(), "the delegated tool's progress was not streamed")
        thread = booking.task["contextId"]
        async with streaming(
            door, send("Is breakfast included?", "m-side", context_id=thread, kind="delegation")
        ) as side:
            await until(service.pause.started.is_set, "the side question's turn did not start")
            async with streaming(
                door, send("Book a taxi.", "m-direct-1", context_id=thread)
            ) as first:
                await until(lambda: first.events, "the first direct message was not admitted")
                async with streaming(
                    door, send("When is check-in?", "m-direct-2", context_id=thread)
                ) as last:
                    await until(lambda: last.events, "the second direct message was not admitted")
                    service.pause.release()
                    await until(
                        lambda: side.reader.done() and first.reader.done() and last.reader.done(),
                        "the side question and the direct burst were not answered",
                    )
        assert service.effects == [] and not booking.reader.done(), "the booking is still held"
        service.gate.set()
        await until(lambda: booking.reader.done(), "the booking did not settle")

    assert booking.final()["status"]["state"] == "TASK_STATE_COMPLETED"
    assert "Booked reservation R1." in "".join(booking.answer())
    assert service.effects == [booking.task["id"]]
    assert side.final()["status"]["state"] == "TASK_STATE_COMPLETED"
    assert side.answer() == ["Breakfast is included."]
    assert (
        first.final()["status"]["state"]
        == last.final()["status"]["state"]
        == ("TASK_STATE_COMPLETED")
    )
    assert last.answer() == ["Your taxi is booked and check-in is at 3pm."]
    assert MERGED_INTO not in json.dumps(last.final())
    assert json.dumps(first.final()).count(last.task["id"]) == 1, "the earlier one is merged"
    assert MERGED_INTO in json.dumps(first.final())
    for stream in (booking, side):
        assert MERGED_INTO not in json.dumps(stream.events)


async def test_a_standalone_bound_run_settles_its_held_tool_before_returning() -> None:
    """Outside a live endpoint a bound run is not a hidden live owner: admitted progress does
    not release the run, which returns only once the body it started has delivered its result."""
    service = BookingService()
    store = ContextStore()
    run = asyncio.create_task(front_desk(service).run("Reserve a room.", context=store.bind()))
    await until(service.updated.is_set, "the tool's progress was not admitted")
    assert not run.done(), "the run returned while its tool body was still held"
    assert not service.acknowledged.is_set(), "progress released the model turn"

    service.gate.set()
    result = await asyncio.wait_for(run, 2)

    assert result.status == "completed"
    assert result.output == "Booked reservation R1."
    assert service.effects == [None]


async def test_an_unbound_run_in_a_live_handler_is_not_a_hidden_live_owner(a2a_door: Door) -> None:
    """A handler's run without the endpoint's binding settles its own body: progress reaches the
    task, but it releases no model turn, and the task ends with the body's result."""
    service = BookingService()
    agent = front_desk(service)

    async def handler(ctx: A2ASessionContext) -> object:
        return await agent.run(ctx.message)

    async with (
        a2a_door(handler, inbox="live") as door,
        streaming(door, send("Reserve a room.", "m-booking")) as booking,
    ):
        await until(service.updated.is_set, "the tool's progress was not admitted")
        assert not booking.reader.done() and not service.acknowledged.is_set()
        service.gate.set()
        await until(lambda: booking.reader.done(), "the booking did not settle")

    assert booking.final()["status"]["state"] == "TASK_STATE_COMPLETED"
    assert booking.answer() == ["Booked reservation R1."]
    assert service.effects == [booking.task["id"]]


async def test_a_live_turns_model_failure_fails_the_task_with_the_providers_error(
    a2a_door: Door,
) -> None:
    """A provider error under a live owner reaches the handler as the provider raised it, as it
    does outside live; it used to surface as a `TypeError` building a fallback `APIError`."""
    agent = Agent(name="desk", instructions="Help.", llm=FakeLLM([KeyError("boom")]))
    raised: list[BaseException] = []

    async def handler(ctx: A2ASessionContext) -> object:
        try:
            return await agent.run(ctx.message, context=ctx.store.bind())
        except BaseException as error:
            raised.append(error)
            raise

    async with a2a_door(handler, inbox="live") as door:
        task = (await door.post("/v1", json=send("Start", "m-start"))).json()["result"]["task"]

    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert [type(error) for error in raised] == [KeyError]


async def test_every_waiter_of_a_failed_live_turn_raises_the_providers_error(
    a2a_door: Door,
) -> None:
    """A live turn failing at the provider fails every request it served with the provider's own
    error; the first waiter used to take it, leaving the others a broken fallback."""
    service = BookingService()
    agent = front_desk(service)
    raised: list[BaseException] = []

    async def handler(ctx: A2ASessionContext) -> object:
        try:
            return await agent.run(ctx.message, context=ctx.store.bind())
        except BaseException as error:
            raised.append(error)
            raise

    async with (
        a2a_door(handler, inbox={"delegation": "live", "a2a": "batch"}) as door,
        streaming(door, send("Reserve a room.", "m-booking", kind="delegation")) as booking,
    ):
        await until(lambda: booking.progress(), "the delegated tool's progress was not streamed")
        thread = booking.task["contextId"]
        async with streaming(
            door, send("Is breakfast included?", "m-side", context_id=thread, kind="delegation")
        ) as side:
            await until(service.pause.started.is_set, "the side question's turn did not start")
            async with streaming(door, send("Cancel it.", "m-1", context_id=thread)) as first:
                await until(lambda: first.events, "the first direct message was not admitted")
                async with streaming(door, send("Cancel all.", "m-2", context_id=thread)) as last:
                    await until(lambda: last.events, "the second direct message was not admitted")
                    service.pause.release()
                    await until(
                        lambda: side.reader.done() and first.reader.done() and last.reader.done(),
                        "the side question and the failed burst did not settle",
                    )
        service.gate.set()
        await until(lambda: booking.reader.done(), "the booking did not settle")

    assert last.final()["status"]["state"] == "TASK_STATE_FAILED"
    assert raised, "the failed turn raised in its handler"
    assert all(isinstance(error, APIStatusError) for error in raised), raised
