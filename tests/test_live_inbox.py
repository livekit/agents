"""The ``live`` inbox (docs/delegation.md): a message runs as soon as the floor is free, messages
that waited together run as one turn, a message that interrupts stops the reply, and a tool that
reports progress goes on in the background while the next message runs."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, RunContext
from livekit.agents._generated.storage_turn import Background
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.a2a.extension import EXTENSION_URI
from livekit.agents.llm import FunctionToolCall
from livekit.agents.storage import AgentDBStorage
from livekit.agents.storage.hub import LaneHub
from livekit.agents.storage.resolve import Thread
from livekit.agents.voice.agent_run import JOINT_QUESTION
from livekit.agents.voice.tool_executor import REPLY_INSTRUCTIONS_AT_TAIL, UPDATE_TEMPLATE

from .fake_llm import FakeLLM, FakeLLMResponse
from .test_a2a_server import (  # noqa: F401  (no_project_env: autouse)
    AGENT,
    http_client,
    make_server,
    no_project_env,
    rpc,
    sse_results,
    state_of,
    wire,
)
from .test_agent_api import RecordingLLM

pytestmark = [pytest.mark.unit]

NO_INTERRUPT = {"livekit/interrupt": False}


def told(*call_ids: str) -> str:
    """What a front desk's model is told last when work ended: tell the results, which a fake
    model's answer is keyed by."""
    return REPLY_INSTRUCTIONS_AT_TAIL.format(
        call_ids=", ".join(f"{call_id}_final" for call_id in call_ids)
    )


async def stream(http: Any, message: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        r["result"] for r in sse_results(await rpc(http, "message/stream", {"message": message}))
    ]


async def tasks_of(storage: AgentDBStorage, thread_id: str) -> dict[str, Any]:
    return {t.id: t for t in await Thread(storage, thread_id, agent_name=AGENT).tasks()}


async def until(check: Any, timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not await check():
        assert asyncio.get_running_loop().time() < deadline, "timed out"
        await asyncio.sleep(0.1)


async def test_a_message_runs_beside_a_tool_in_the_background(
    agentdb_storage: AgentDBStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A booking reports progress and keeps running; a correction runs at once beside it; the
    booking's end answers its own task. The correction's request is in the holder's process:
    it tells the holder through the lane's hub, not at the holder's next read of the inbox."""
    woke: list[Background] = []
    background = LaneHub.background

    async def spied(self: LaneHub, wait: float) -> Background:
        woke.append(await background(self, wait))
        return woke[-1]

    monkeypatch.setattr(LaneHub, "background", spied)
    entered = asyncio.Event()
    release = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        entered.set()
        await release.wait()
        return f"booked {day}"

    progress = UPDATE_TEMPLATE.format(function_name="book", call_id="b1", message="booking now")
    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="book monday",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="book", arguments='{"day": "monday"}', call_id="b1")
                    ],
                ),
                FakeLLMResponse(input=progress, content="On it.", ttft=0, duration=0),
                FakeLLMResponse(
                    input="and a reminder", content="Reminder set.", ttft=0, duration=0
                ),
                FakeLLMResponse(input=told("b1"), content="Booked for Monday.", ttft=0, duration=0),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        result = await desk.run(ctx.message, message_history=ctx.message_history)
        return result.output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))
        await asyncio.wait_for(entered.wait(), 20)

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(
                t.status == "working" and t.data.metadata.get("livekit/background") for t in rows
            )

        await until(backgrounded)
        # the floor is free: the next message runs at once, beside the booking
        second = await asyncio.wait_for(
            stream(http, wire("and a reminder", "m2", context_id=thread_id)), 20
        )
        assert state_of(second[-1]) == "TASK_STATE_COMPLETED"
        assert not first.done(), "the booking's task waits for the booking"
        assert any(b.arrived for b in woke), "the holder was told of the correction"

        release.set()
        first_frames = await asyncio.wait_for(first, 30)

    t1 = first_frames[0]["task"]["id"]
    assert state_of(first_frames[-1]) == "TASK_STATE_COMPLETED"
    working = [r for r in first_frames if state_of(r) == "TASK_STATE_WORKING"]
    assert any(
        (w["statusUpdate"].get("metadata") or {}).get(f"{EXTENSION_URI}/background") is True
        for w in working
    ), "the task told its caller it went on in the background"
    done = (await tasks_of(agentdb_storage, thread_id))[t1]
    assert done.status == "completed"
    assert (
        done.data.artifacts and done.data.artifacts[0]["parts"][0]["text"] == "Booked for Monday."
    )
    assert done.data.metadata.get("livekit/background") is False


async def test_messages_that_waited_together_run_as_one_turn(
    agentdb_storage: AgentDBStorage,
) -> None:
    """While a reply is produced, two messages that do not interrupt wait; they run together
    once the floor is free, and the earlier one is merged into the later one."""
    gate = asyncio.Event()
    entered = asyncio.Event()
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        batches.append([m.parts[0]["text"] for m in ctx.messages])
        if len(batches) == 1:
            entered.set()
            await gate.wait()
        return "seen " + " + ".join(batches[-1])

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("one", "m1", context_id=thread_id)))
        await asyncio.wait_for(entered.wait(), 20)
        second = asyncio.create_task(
            stream(http, wire("two", "m2", context_id=thread_id, **NO_INTERRUPT))
        )

        async def second_waiting() -> bool:
            return len(await tasks_of(agentdb_storage, thread_id)) == 2

        await until(second_waiting)
        third = asyncio.create_task(
            stream(http, wire("three", "m3", context_id=thread_id, **NO_INTERRUPT))
        )

        async def both_waiting() -> bool:
            return len(await tasks_of(agentdb_storage, thread_id)) == 3

        await until(both_waiting)
        await asyncio.sleep(0.5)
        gate.set()
        r1, r2, r3 = await asyncio.wait_for(asyncio.gather(first, second, third), 30)

    assert batches == [["one"], ["two", "three"]]
    assert state_of(r1[-1]) == "TASK_STATE_COMPLETED"
    rows = await tasks_of(agentdb_storage, thread_id)
    t2, t3 = r2[0]["task"]["id"], r3[0]["task"]["id"]
    assert rows[t2].status == "completed"
    assert rows[t2].data.metadata["livekit/mergedInto"] == t3
    assert rows[t3].data.artifacts[0]["parts"][0]["text"] == "seen two + three"
    assert state_of(r3[-1]) == "TASK_STATE_COMPLETED"


async def test_a_message_that_interrupts_stops_the_reply(
    agentdb_storage: AgentDBStorage,
) -> None:
    """A message from the person stops the reply the agent is producing; that task ends
    ``canceled`` and the message runs."""
    gate = asyncio.Event()
    entered = asyncio.Event()

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        if ctx.message.text_content == "long question":
            entered.set()
            await gate.wait()

            async def tokens() -> Any:
                yield "a long answer"

            return await ctx.stream(tokens())
        return "short answer"

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("long question", "m1", context_id=thread_id)))
        await asyncio.wait_for(entered.wait(), 20)
        second = asyncio.create_task(stream(http, wire("never mind", "m2", context_id=thread_id)))

        async def waiting() -> bool:
            return len(await tasks_of(agentdb_storage, thread_id)) == 2

        await until(waiting)
        await asyncio.sleep(0.3)
        gate.set()
        r1, r2 = await asyncio.wait_for(asyncio.gather(first, second), 30)

    assert state_of(r1[-1]) == "TASK_STATE_CANCELED"
    rows = await tasks_of(agentdb_storage, thread_id)
    canceled = rows[r1[0]["task"]["id"]]
    assert canceled.data.metadata["livekit/interrupted"] is True
    assert state_of(r2[-1]) == "TASK_STATE_COMPLETED"


@pytest.mark.parametrize("shown", [True, False], ids=["streaming", "before-the-first-token"])
async def test_a_message_stops_the_reply_while_it_streams(
    agentdb_storage: AgentDBStorage, shown: bool
) -> None:
    """The model is still writing its reply when a message that interrupts lands: the reply
    stops there, what was shown is kept as cut off, the task ends ``canceled``, and the message
    runs next, without waiting for the reply's end."""
    llm = RecordingLLM(
        fake_responses=[
            # the text arrives at once and the stream stays open, as a long reply does; or the
            # model is slow to start
            FakeLLMResponse(
                input="tell me everything",
                content="Part one of a long answer.",
                ttft=0 if shown else 20,
                duration=20,
            ),
            FakeLLMResponse(input="never mind", content="Sure.", ttft=0, duration=0),
        ]
    )
    desk = Agent(instructions="front desk", llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return (await desk.run(ctx.message, message_history=ctx.message_history)).output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        started = asyncio.get_running_loop().time()
        first = asyncio.create_task(
            stream(http, wire("tell me everything", "m1", context_id=thread_id))
        )
        await until(lambda: asyncio.sleep(0, result=len(llm.requests) == 1))
        await asyncio.sleep(0.5)
        second = asyncio.create_task(stream(http, wire("never mind", "m2", context_id=thread_id)))
        r1, r2 = await asyncio.wait_for(asyncio.gather(first, second), 30)
        took = asyncio.get_running_loop().time() - started

    assert took < 15, "the second message waited for the first reply to end"
    assert state_of(r1[-1]) == "TASK_STATE_CANCELED"
    rows = await tasks_of(agentdb_storage, thread_id)
    cut_off = rows[r1[0]["task"]["id"]]
    assert cut_off.data.metadata["livekit/interrupted"] is True
    # what was shown is the task's answer, the same on its stream and in its row
    streamed = "".join(
        p.get("text", "")
        for r in r1
        if "artifactUpdate" in r
        for p in r["artifactUpdate"]["artifact"].get("parts", [])
    )
    stored = [p["text"] for a in cut_off.data.artifacts or [] for p in a["parts"]]
    assert stored == (["Part one of a long answer."] if shown else [])
    assert streamed == "".join(stored)
    assert state_of(r2[-1]) == "TASK_STATE_COMPLETED"
    items = (await Thread(agentdb_storage, thread_id, agent_name=AGENT).chat_history()).items
    said = [
        (i.text_content, i.interrupted)
        for i in items
        if i.type == "message" and i.role == "assistant"
    ]
    cut = [("Part one of a long answer.", True)] if shown else []
    assert said == [*cut, ("Sure.", False)]


async def test_a_cancel_stops_work_in_the_background(agentdb_storage: AgentDBStorage) -> None:
    """``tasks/cancel`` of a task whose tool runs in the background stops the tool; the task
    ends ``canceled`` and the lane is let go."""
    entered = asyncio.Event()
    stopped = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext) -> str:
        """Book."""
        await ctx.update("booking now")
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            stopped.set()
            raise
        return "booked"

    progress = UPDATE_TEMPLATE.format(function_name="book", call_id="b1", message="booking now")
    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="book",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="book", arguments="{}", call_id="b1")],
                ),
                FakeLLMResponse(input=progress, content="On it.", ttft=0, duration=0),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return (await desk.run(ctx.message, message_history=ctx.message_history)).output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book", "m1", context_id=thread_id)))
        await asyncio.wait_for(entered.wait(), 20)

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(t.data.metadata.get("livekit/background") for t in rows)

        await until(backgrounded)
        (task_id,) = (await tasks_of(agentdb_storage, thread_id)).keys()
        canceled = await rpc(http, "tasks/cancel", {"id": task_id, "contextId": thread_id})
        assert canceled.json()["result"]["status"]["state"] == "TASK_STATE_CANCELED"
        frames = await asyncio.wait_for(first, 20)

    assert stopped.is_set()
    assert state_of(frames[-1]) == "TASK_STATE_CANCELED"
    row = (await tasks_of(agentdb_storage, thread_id))[task_id]
    assert row.status == "canceled"


async def test_a_task_waits_for_all_its_work_in_the_background(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Two lookups of one request go on in the background; the first to end gets a turn, and
    the task answers only when the second has ended too."""
    gates = {"A1": asyncio.Event(), "B2": asyncio.Event()}
    desk = Agent(instructions="support")

    @desk.tool
    async def lookup(ctx: RunContext, order: str) -> str:
        """Look an order up.

        Args:
            order: the order number
        """
        await ctx.update(f"looking up {order}")
        await gates[order].wait()
        return f"{order} shipped"

    def call(order: str, call_id: str) -> FunctionToolCall:
        return FunctionToolCall(name="lookup", arguments=f'{{"order": "{order}"}}', call_id=call_id)

    progress = UPDATE_TEMPLATE.format(function_name="lookup", call_id="l2", message="looking up B2")
    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="check A1 and B2",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[call("A1", "l1"), call("B2", "l2")],
                ),
                FakeLLMResponse(input=progress, content="Looking.", ttft=0, duration=0),
                FakeLLMResponse(input=told("l1"), content="A1 shipped.", ttft=0, duration=0),
                FakeLLMResponse(input=told("l2"), content="Both shipped.", ttft=0, duration=0),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return (await desk.run(ctx.message, message_history=ctx.message_history)).output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(
            stream(http, wire("check A1 and B2", "m1", context_id=thread_id))
        )

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(t.data.metadata.get("livekit/background") for t in rows)

        await until(backgrounded)
        gates["A1"].set()
        await asyncio.sleep(2.0)
        (task_id,) = (await tasks_of(agentdb_storage, thread_id)).keys()
        assert (await tasks_of(agentdb_storage, thread_id))[task_id].status == "working"
        gates["B2"].set()
        frames = await asyncio.wait_for(first, 30)

    assert state_of(frames[-1]) == "TASK_STATE_COMPLETED"
    row = (await tasks_of(agentdb_storage, thread_id))[task_id]
    assert row.status == "completed"
    assert row.data.artifacts[0]["parts"][0]["text"] == "Both shipped."


async def test_work_that_ends_together_gets_one_reply_for_both_tasks(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Two bookings run in the background and end together: one turn resumes both, the model
    gives one reply, and each task gets it as its answer on its own stream. The earlier task is
    marked answered with the later one, not merged into it."""
    release = asyncio.Event()
    started: list[str] = []
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        started.append(day)
        await release.wait()
        return f"booked {day}"

    def progress(call_id: str) -> str:
        return UPDATE_TEMPLATE.format(function_name="book", call_id=call_id, message="booking now")

    def booking(text: str, day: str, call_id: str) -> FakeLLMResponse:
        call = FunctionToolCall(name="book", arguments=f'{{"day": "{day}"}}', call_id=call_id)
        return FakeLLMResponse(input=text, content="", ttft=0, duration=0, tool_calls=[call])

    llm = RecordingLLM(
        fake_responses=[
            booking("book monday", "monday", "b1"),
            FakeLLMResponse(input=progress("b1"), content="On it.", ttft=0, duration=0),
            booking("and tuesday", "tuesday", "b2"),
            FakeLLMResponse(input=progress("b2"), content="That too.", ttft=0, duration=0),
            FakeLLMResponse(input=told("b1", "b2"), content="Both are booked.", ttft=0, duration=0),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        result = await desk.run(ctx.message, message_history=ctx.message_history)
        return result.output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))
        await until(lambda: asyncio.sleep(0, result=started == ["monday"]))
        second = asyncio.create_task(stream(http, wire("and tuesday", "m2", context_id=thread_id)))
        await until(lambda: asyncio.sleep(0, result=started == ["monday", "tuesday"]))
        before = len(llm.requests)
        release.set()
        frames = await asyncio.wait_for(asyncio.gather(first, second), 30)

    assert len(llm.requests) == before + 1, "one model request words both results"
    resumed = llm.requests[-1]
    assert resumed.items[-1].text_content == told("b1", "b2"), (
        "the replies while the work ran never had its results: the model tells them"
    )
    tasks = await tasks_of(agentdb_storage, thread_id)
    t1, t2 = (f[0]["task"]["id"] for f in frames)
    for task_id, task_frames in zip((t1, t2), frames, strict=True):
        assert state_of(task_frames[-1]) == "TASK_STATE_COMPLETED"
        row = tasks[task_id]
        assert row.status == "completed"
        assert row.data.artifacts and row.data.artifacts[0]["parts"][0]["text"] == (
            "Both are booked."
        )
        assert "livekit/mergedInto" not in row.data.metadata
    assert tasks[t1].data.metadata.get("livekit/answeredWith") == t2


async def test_work_that_ends_together_and_asks_pauses_only_the_task_that_asked(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Two pieces of work end together and one of them asks the person: one reply words the
    other's result, then the question is asked. The task whose work asked waits
    ``input-required`` with the question; the other completes with the reply, which carries
    its result, not the question."""
    release = asyncio.Event()
    started: list[str] = []
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str | InputRequired:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        started.append(day)
        await release.wait()
        if day == "tuesday":
            return InputRequired("Which time on Tuesday, morning or afternoon?")
        return f"booked {day}"

    def progress(call_id: str) -> str:
        return UPDATE_TEMPLATE.format(function_name="book", call_id=call_id, message="booking now")

    def booking(text: str, day: str, call_id: str) -> FakeLLMResponse:
        call = FunctionToolCall(name="book", arguments=f'{{"day": "{day}"}}', call_id=call_id)
        return FakeLLMResponse(input=text, content="", ttft=0, duration=0, tool_calls=[call])

    question = "Which time on Tuesday, morning or afternoon?"
    resumed = [
        told(*ids) + "\n" + JOINT_QUESTION.format(question=question)
        for ids in (("b1", "b2"), ("b2", "b1"))
    ]
    llm = RecordingLLM(
        fake_responses=[
            booking("book monday", "monday", "b1"),
            FakeLLMResponse(input=progress("b1"), content="On it.", ttft=0, duration=0),
            booking("and tuesday", "tuesday", "b2"),
            FakeLLMResponse(input=progress("b2"), content="That too.", ttft=0, duration=0),
            *(
                FakeLLMResponse(input=text, content="Monday is booked.", ttft=0, duration=0)
                for text in resumed
            ),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))
        await until(lambda: asyncio.sleep(0, result=started == ["monday"]))
        second = asyncio.create_task(stream(http, wire("and tuesday", "m2", context_id=thread_id)))
        await until(lambda: asyncio.sleep(0, result=started == ["monday", "tuesday"]))
        release.set()
        frames = await asyncio.wait_for(asyncio.gather(first, second), 30)

    tasks = await tasks_of(agentdb_storage, thread_id)
    monday, tuesday = (f[0]["task"]["id"] for f in frames)
    assert tasks[monday].status == "completed"
    assert tasks[monday].data.artifacts[0]["parts"][0]["text"] == "Monday is booked."
    assert tasks[monday].data.metadata.get("livekit/answeredWith") == tuesday
    assert tasks[tuesday].status == "input-required"
    assert tasks[tuesday].status_message["parts"][0]["text"] == question
    assert state_of(frames[0][-1]) == "TASK_STATE_COMPLETED"
    assert state_of(frames[1][-1]) == "TASK_STATE_INPUT_REQUIRED"
    # the person reads the reply, then the question, once each
    items = (await Thread(agentdb_storage, thread_id, agent_name=AGENT).chat_history()).items
    said = [i.text_content for i in items if getattr(i, "role", None) == "assistant"]
    assert said[-2:] == ["Monday is booked.", question]


@dataclass
class Booking:
    booked: str | None = None


async def test_state_a_tool_writes_in_the_background_is_kept(
    agentdb_storage: AgentDBStorage,
) -> None:
    """A tool in the background writes the state after its turn committed: the turn that
    resumes it takes the same live state object and commits the write. In process, the door's
    ``turn()`` returns when its own task ends, while the holder still works for another."""
    release = {"monday": asyncio.Event(), "tuesday": asyncio.Event()}
    started: list[str] = []
    desk = Agent(instructions="front desk", state_type=Booking)

    @desk.tool
    async def book(ctx: RunContext[None, Booking], day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        started.append(day)
        await release[day].wait()
        ctx.state.booked = day
        return f"booked {day}"

    def progress(call_id: str) -> str:
        return UPDATE_TEMPLATE.format(function_name="book", call_id=call_id, message="booking now")

    def booking(text: str, day: str, call_id: str) -> FakeLLMResponse:
        call = FunctionToolCall(name="book", arguments=f'{{"day": "{day}"}}', call_id=call_id)
        return FakeLLMResponse(input=text, content="", ttft=0, duration=0, tool_calls=[call])

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                booking("book monday", "monday", "b1"),
                FakeLLMResponse(input=progress("b1"), content="On it.", ttft=0, duration=0),
                booking("and tuesday", "tuesday", "b2"),
                FakeLLMResponse(input=progress("b2"), content="That too.", ttft=0, duration=0),
                FakeLLMResponse(input=told("b1"), content="Monday.", ttft=0, duration=0),
                FakeLLMResponse(input=told("b2"), content="Tuesday.", ttft=0, duration=0),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Booking]) -> Any:
        result = await desk.run(ctx.message, message_history=ctx.message_history, state=ctx.state)
        return result.output

    _, endpoint = make_server(agentdb_storage, handler, inbox="live", state=Booking)
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id

    def turn(text: str, message_id: str) -> asyncio.Task[Any]:
        message = {
            "messageId": message_id,
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
        }
        return asyncio.create_task(endpoint.service.turn(message, thread_id=thread_id))

    async def booked() -> Any:
        db = await agentdb_storage.open(thread_id)
        try:
            lane = await db.lane(AGENT)
        finally:
            db.close()
        return (
            lane.state.get("booked") if lane is not None and isinstance(lane.state, dict) else None
        )

    monday = turn("book monday", "m1")
    await until(lambda: asyncio.sleep(0, result=started == ["monday"]))
    tuesday = turn("and tuesday", "m2")
    await until(lambda: asyncio.sleep(0, result=started == ["monday", "tuesday"]))
    release["monday"].set()
    first = await asyncio.wait_for(monday, 20)
    assert first.result.task.status == "completed" and first.result.output == "Monday."
    assert not tuesday.done(), "the holder still works for the other task"
    assert await booked() == "monday", "the write after the tool's turn committed is kept"
    release["tuesday"].set()
    second = await asyncio.wait_for(tuesday, 20)
    assert second.result.task.status == "completed"
    assert await booked() == "tuesday"


async def test_a_result_is_told_after_replies_that_only_acknowledged_the_work(
    agentdb_storage: AgentDBStorage,
) -> None:
    """The s8 shape: Monday's booking runs, the person changes to Tuesday, and Monday's result
    comes after replies that only acknowledged the work. No reply had the result in front of
    it, so the turn that resumes it is told to tell it, and it does."""
    release = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        if day == "monday":
            await release.wait()
        return f"booked {day}"

    def booking(text: str, day: str, call_id: str, said: str) -> FakeLLMResponse:
        call = FunctionToolCall(name="book", arguments=f'{{"day": "{day}"}}', call_id=call_id)
        return FakeLLMResponse(input=text, content=said, ttft=0, duration=0, tool_calls=[call])

    llm = RecordingLLM(
        fake_responses=[
            booking("book monday", "monday", "b1", "Booking Monday."),
            booking("make it tuesday instead", "tuesday", "b2", "Booking Tuesday."),
            FakeLLMResponse(input=told("b2"), content="Tuesday is booked.", ttft=0, duration=0),
            FakeLLMResponse(input=told("b1"), content="Monday is booked too.", ttft=0, duration=0),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return (await desk.run(ctx.message, message_history=ctx.message_history)).output

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(t.data.metadata.get("livekit/background") for t in rows)

        await until(backgrounded)
        second = await stream(http, wire("make it tuesday instead", "m2", context_id=thread_id))
        release.set()
        frames = await asyncio.wait_for(first, 30)

    assert state_of(second[-1]) == state_of(frames[-1]) == "TASK_STATE_COMPLETED"
    row = (await tasks_of(agentdb_storage, thread_id))[frames[0]["task"]["id"]]
    assert row.data.artifacts[0]["parts"][0]["text"] == "Monday is booked too."
    items = (await Thread(agentdb_storage, thread_id, agent_name=AGENT).chat_history()).items
    said = [i.text_content for i in items if i.type == "message" and i.role == "assistant"]
    assert said[-1] == "Monday is booked too."


DELEGATION = {"livekit/kind": "delegation"}


async def test_a_read_back_is_answered_only_by_a_message_of_its_task(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Two tasks wait on one lane: one on a read-back, the other on a question. The other
    task's answer, and its repeat of the read-back's call, do not approve the call: it reads
    back again in that task, and nothing runs."""
    canceled: list[str] = []
    desk = Agent(instructions="back office")

    async def read_back(ctx: RunContext, day: str) -> str:
        return f"Cancel {day}? Yes or no?"

    @desk.tool(requires_approval=read_back)
    async def cancel(ctx: RunContext, day: str) -> str:
        """Cancel the appointment.

        Args:
            day: the day
        """
        canceled.append(day)
        return f"canceled {day}"

    @desk.tool
    async def remind(ctx: RunContext) -> InputRequired:
        """Set a reminder."""
        return InputRequired("Which phone number?", verbatim=True)

    def asks(text: str, name: str, call_id: str, **arguments: str) -> FakeLLMResponse:
        call = FunctionToolCall(name=name, arguments=json.dumps(arguments), call_id=call_id)
        return FakeLLMResponse(input=text, content="", ttft=0, duration=0, tool_calls=[call])

    desk.update_options(
        llm=FakeLLM(
            fake_responses=[
                asks("cancel monday", "cancel", "c1", day="monday"),
                asks("and a reminder", "remind", "r1"),
                # the reminder's task answers yes, and repeats the other task's call
                asks("yes", "cancel", "c2", day="monday"),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = await stream(http, wire("cancel monday", "m1", context_id=thread_id, **DELEGATION))
        second = await stream(
            http, wire("and a reminder", "m2", context_id=thread_id, **DELEGATION)
        )
        reminder = second[0]["task"]["id"]
        third = await stream(
            http,
            {**wire("yes", "m3", context_id=thread_id, **DELEGATION), "taskId": reminder},
        )

    assert state_of(first[-1]) == state_of(second[-1]) == "TASK_STATE_INPUT_REQUIRED"
    assert canceled == [], "another task's yes approved nothing"
    assert state_of(third[-1]) == "TASK_STATE_INPUT_REQUIRED"
    tasks = await tasks_of(agentdb_storage, thread_id)
    assert tasks[reminder].status_message["parts"][0]["text"] == "Cancel monday? Yes or no?"


async def test_a_continued_task_acts_only_on_its_own_request(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Two bookings wait on one lane for the email. The answer continues the first one: its
    model reads its own booking and not the other task's, which came after it paused, and
    books only its own day. The lane keeps one history with both."""
    booked: list[str] = []
    desk = Agent(instructions="back office")

    @desk.tool
    async def book(ctx: RunContext, day: str, email: str | None = None) -> str | InputRequired:
        """Book a cleaning.

        Args:
            day: the day
            email: where the confirmation goes
        """
        if email is None:
            return InputRequired("Which email?", verbatim=True)
        booked.append(day)
        return f"booked {day}"

    class Booker(RecordingLLM):
        """Books again the last day it reads, with the email it was given."""

        def chat(self, *, chat_ctx: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            days = [
                json.loads(i.arguments)["day"]
                for i in chat_ctx.items
                if i.type == "function_call" and i.name == "book"
            ]
            if days:
                call = FunctionToolCall(
                    name="book",
                    arguments=json.dumps({"day": days[-1], "email": "a@b.co"}),
                    call_id="b3",
                )
                self.fake_response_map["a@b.co"] = FakeLLMResponse(
                    input="a@b.co", content="", ttft=0, duration=0, tool_calls=[call]
                )
            return super().chat(chat_ctx=chat_ctx, **kwargs)

    def booking(text: str, day: str, call_id: str) -> FakeLLMResponse:
        call = FunctionToolCall(name="book", arguments=json.dumps({"day": day}), call_id=call_id)
        return FakeLLMResponse(input=text, content="", ttft=0, duration=0, tool_calls=[call])

    llm = Booker(
        fake_responses=[
            booking("book monday", "monday", "b1"),
            booking("book thursday", "thursday", "b2"),
            FakeLLMResponse(input="booked monday", content="Monday.", ttft=0, duration=0),
            FakeLLMResponse(input="booked thursday", content="Thursday.", ttft=0, duration=0),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = await stream(http, wire("book monday", "m1", context_id=thread_id, **DELEGATION))
        await stream(http, wire("book thursday", "m2", context_id=thread_id, **DELEGATION))
        monday = first[0]["task"]["id"]
        answer = await stream(
            http, {**wire("a@b.co", "m3", context_id=thread_id, **DELEGATION), "taskId": monday}
        )

    assert booked == ["monday"]
    assert state_of(answer[-1]) == "TASK_STATE_COMPLETED"
    read = llm.requests[-2]  # the answer's request; the last words the result
    assert "book thursday" not in [i.text_content for i in read.items if i.type == "message"]
    items = (await Thread(agentdb_storage, thread_id, agent_name=AGENT).chat_history()).items
    said = [i.text_content for i in items if i.type == "message" and i.role == "user"]
    assert said == ["book monday", "book thursday", "a@b.co"], "one history keeps both"


async def test_progress_after_the_reply_beside_the_call_asks_no_second_reply(
    agentdb_storage: AgentDBStorage,
) -> None:
    """The model replied beside its call; the call's first update is progress: it reaches the
    task's status and the model's context, and the model is not asked to reply again. The
    result still gets its reply."""
    release = asyncio.Event()
    desk = Agent(instructions="front desk")

    @desk.tool
    async def book(ctx: RunContext, day: str) -> str:
        """Book a cleaning.

        Args:
            day: the day
        """
        await ctx.update("booking now")
        await release.wait()
        return f"booked {day}"

    progress = UPDATE_TEMPLATE.format(function_name="book", call_id="b1", message="booking now")
    call = FunctionToolCall(name="book", arguments='{"day": "monday"}', call_id="b1")
    llm = RecordingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="book monday", content="On it.", ttft=0, duration=0, tool_calls=[call]
            ),
            FakeLLMResponse(input=progress, content="A mix-up?", ttft=0, duration=0),
            FakeLLMResponse(
                input=REPLY_INSTRUCTIONS_AT_TAIL.format(call_ids="b1_final"),
                content="Booked for Monday.",
                ttft=0,
                duration=0,
            ),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(stream(http, wire("book monday", "m1", context_id=thread_id)))

        async def backgrounded() -> bool:
            rows = (await tasks_of(agentdb_storage, thread_id)).values()
            return any(t.data.metadata.get("livekit/background") for t in rows)

        await until(backgrounded)
        release.set()
        frames = await asyncio.wait_for(first, 30)

    assert state_of(frames[-1]) == "TASK_STATE_COMPLETED"
    last = [r.items[-1] for r in llm.requests]
    assert not any(i.type == "function_call_output" and i.output == progress for i in last)
    working = [r for r in frames if state_of(r) == "TASK_STATE_WORKING"]
    assert any("booking now" in json.dumps(w) for w in working), "the progress was shown"
    items = (await Thread(agentdb_storage, thread_id, agent_name=AGENT).chat_history()).items
    said = [i.text_content for i in items if i.type == "message" and i.role == "assistant"]
    assert said == ["On it.", "Booked for Monday."]


class _ChoosingLLM(FakeLLM):
    """A fake model that records the tool choice and, as gemma sometimes does, calls a tool
    even under ``tool_choice="none"``."""

    def __init__(self, *, fake_responses: list[FakeLLMResponse]) -> None:
        super().__init__(fake_responses=fake_responses)
        self.choices: list[tuple[str, Any]] = []

    def chat(self, *, chat_ctx: Any, tool_choice: Any = None, **kwargs: Any) -> Any:  # type: ignore[override]
        last = chat_ctx.items[-1]
        self.choices.append((getattr(last, "output", None) or last.text_content, tool_choice))
        return super().chat(chat_ctx=chat_ctx, **kwargs)


async def test_the_reply_to_progress_in_text_calls_no_tools(
    agentdb_storage: AgentDBStorage,
) -> None:
    """The call's progress is the step's only new input: the model replies to it with no tools,
    as on a call, and a call it makes anyway is not run, so the request is not sent again
    (H9)."""
    release = asyncio.Event()
    asked: list[str] = []
    desk = Agent(instructions="front desk")

    @desk.tool
    async def ask(ctx: RunContext, request: str) -> str:
        """The back office.

        Args:
            request: the request
        """
        asked.append(request)
        await ctx.update("Checking the calendar.")
        await release.wait()
        return "Monday at 10:00 is free."

    progress = UPDATE_TEMPLATE.format(
        function_name="ask", call_id="a1", message="Checking the calendar."
    )
    again = FunctionToolCall(name="ask", arguments='{"request": "monday slots?"}', call_id="a2")
    llm = _ChoosingLLM(
        fake_responses=[
            FakeLLMResponse(
                input="any slots monday?",
                content="",
                ttft=0,
                duration=0,
                tool_calls=[
                    FunctionToolCall(name="ask", arguments='{"request": "monday"}', call_id="a1")
                ],
            ),
            FakeLLMResponse(
                input=progress, content="Checking now.", ttft=0, duration=0, tool_calls=[again]
            ),
            FakeLLMResponse(input=told("a1"), content="Monday at 10 is free.", ttft=0, duration=0),
        ]
    )
    desk.update_options(llm=llm)

    async def handler(ctx: A2ASessionContext[Any]) -> Any:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    server, _ = make_server(agentdb_storage, handler, inbox="live")
    thread_id = (await agentdb_storage.threads.create_thread()).thread_id
    async with http_client(server) as http:
        first = asyncio.create_task(
            stream(http, wire("any slots monday?", "m1", context_id=thread_id))
        )
        await until(lambda: asyncio.sleep(0, result=len(llm.choices) >= 2))
        release.set()
        frames = await asyncio.wait_for(first, 30)

    assert llm.choices[1] == (progress, "none"), "the reply to progress calls no tool"
    assert llm.choices[0][1] != "none"
    assert asked == ["monday"], "the request went once"
    assert state_of(frames[-1]) == "TASK_STATE_COMPLETED"
