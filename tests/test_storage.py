"""The storage package (plan 5): the Rust turn with the door's host, the lease across workers,
resolution, the voice lane; against the local agent-db of ``storage_fixture.py``, every test
skipped when the data plane is not up. The run itself (a raising run, a committed run id, a
stale commit) is tested in the crate: ``rust/crates/lk-agents-storage/tests``."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from livekit.agents import Agent, InputRequired, RunContext, RunResult, _core, persisted
from livekit.agents._generated.a2a_metadata import (
    CHANNEL,
    INTERRUPTED,
    INTERRUPTION_REASON,
    MERGED_INTO,
    REASON,
)
from livekit.agents._generated.storage_class import LaneRow, TaskRecord
from livekit.agents._generated.storage_options import TurnOptions
from livekit.agents._generated.storage_turn import TurnResult
from livekit.agents.llm import ChatItem, ChatMessage, FunctionToolCall
from livekit.agents.llm.utils import build_legacy_openai_schema
from livekit.agents.storage import Storage
from livekit.agents.storage._protocol import item_to_payload, payload_to_item
from livekit.agents.storage.context import A2ASessionContext, _Handler, _TurnHost
from livekit.agents.storage.errors import RunRejected, TooLarge
from livekit.agents.storage.resolve import Thread, _resolve_thread

from .fake_llm import FakeLLM, FakeLLMResponse
from .storage_fixture import lease_ttl, let_the_lease_lapse, new_session

pytestmark = [pytest.mark.unit]

LANE = "front-desk"


@dataclass
class Patient:
    verified: int = 0


@dataclass
class AppContext:
    patient: Patient = persisted(default_factory=Patient)


def a2a(
    text: str, message_id: str, *, channel: str | None = None, task_id: str | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if channel:
        msg["metadata"] = {CHANNEL: channel}
    if task_id:
        msg["taskId"] = task_id
    return msg


async def echo(ctx: A2ASessionContext) -> str:
    return "reply to: " + ctx.message.text_content


Workers = tuple[Storage, Storage]


async def run_turn(
    storage: Storage,
    thread_id: str,
    lane: str,
    message: dict[str, Any],
    policy: str | None,
    handler: _Handler,
    *,
    debounce: timedelta | None = None,
    task_id: str | None = None,
    wait_timeout: float | None = None,
) -> TurnResult:
    """One message through the Rust turn with the door's host, as the A2A service runs it."""
    options = TurnOptions(
        policy=policy,
        debounce=debounce.total_seconds() if debounce is not None else None,
        wait_timeout=wait_timeout,
    )
    host = _TurnHost(
        handler,
        thread=Thread(storage, thread_id, agent_name=lane),
    )
    core = await storage._open(thread_id)
    try:
        return await _core.Turn(core, lane, host, options).run(message, task_id)
    finally:
        core.close()


async def history_items(storage: Storage, thread_id: str, lane: str) -> list[ChatItem]:
    return (await Thread(storage, thread_id, agent_name=lane).chat_history()).items


async def tasks_of(storage: Storage, thread_id: str, lane: str) -> list[TaskRecord]:
    return await Thread(storage, thread_id, agent_name=lane).lane(lane).tasks(limit=500)


async def task_of(storage: Storage, thread_id: str, lane: str, task_id: str) -> TaskRecord | None:
    return next((t for t in await tasks_of(storage, thread_id, lane) if t.task_id == task_id), None)


async def lane_row(storage: Storage, thread_id: str, lane: str) -> LaneRow | None:
    core = await storage._open(thread_id)
    try:
        return await core.lane_info(lane)
    finally:
        core.close()


async def run_outcomes(storage: Storage, thread_id: str) -> list[str]:
    """What ``run_ended`` recorded, oldest first."""
    core = await storage._open(thread_id)
    try:
        events = await core.journal()
    finally:
        core.close()
    return [str(e.payload["outcome"]) for e in events if e.type == "run_ended"]


# -- the thin slice --


def test_a_stored_items_extra_reads_back_exactly() -> None:
    """Decision 117 (Q29): ``extra`` crosses the proto losslessly, so a string that looks like
    JSON stays a string; a raw string an earlier row stored still reads as that string."""
    extra = {"json_like": '{"a": 1}', "number_like": "42", "text": "é", "object": {"a": 1}}
    message = ChatMessage(role="user", content=["hi"], extra=extra)
    payload = item_to_payload(message)
    assert payload_to_item(payload).extra == extra
    payload["message"]["extra"] = {"text": "asked"}
    assert payload_to_item(payload).extra == {"text": "asked"}


async def test_two_workers_share_one_session(thread_storages: Workers) -> None:
    """Nothing stays in memory between runs: the second worker loads what the first committed."""
    a, b = thread_storages
    sid = await new_session(a, LANE)
    seen: list[list[str]] = []

    async def handler(ctx: A2ASessionContext) -> str:
        seen.append([i.text_content or "" for i in ctx.message_history.items])
        app = AppContext()
        await ctx.store.load_context(app)
        app.patient.verified += 1
        await ctx.store.save_context(app)
        return f"seen {app.patient.verified}"

    first = await run_turn(a, sid, LANE, a2a("hello", "m1"), "queue", handler)
    second = await run_turn(b, sid, LANE, a2a("again", "m2"), "queue", handler)

    assert first.output == "seen 1" and second.output == "seen 2"
    assert seen == [[], ["hello", "seen 1"]]
    assert first.task.status == "completed" and first.task.data.artifacts
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["hello", "seen 1", "again", "seen 2"]
    assert await run_outcomes(b, sid) == ["completed", "completed"]


@dataclass
class Notes:
    text: str = ""


@dataclass
class NotesContext:
    notes: Notes = persisted(default_factory=Notes)


async def test_a_save_over_64_kib_raises_too_large_and_commits_nothing(
    thread_storage: Storage,
) -> None:
    """docs/internals/storage.md "What is saved": state is under 64 KiB; a larger save raises
    ``TooLarge`` and the turn's saved fields stay as they were."""
    sid = await new_session(thread_storage, LANE)
    raised: list[BaseException] = []

    async def handler(ctx: A2ASessionContext) -> str:
        app = NotesContext()
        await ctx.store.load_context(app)
        app.notes.text = "small"
        await ctx.store.save_context(app)
        app.notes.text = "x" * (64 * 1024)
        try:
            await ctx.store.save_context(app)
        except TooLarge as e:
            raised.append(e)
        return "ok"

    async def reader(ctx: A2ASessionContext) -> str:
        app = NotesContext()
        await ctx.store.load_context(app)
        return app.notes.text

    await run_turn(thread_storage, sid, LANE, a2a("save", "m1"), "queue", handler)
    read = await run_turn(thread_storage, sid, LANE, a2a("read", "m2"), "queue", reader)
    assert len(raised) == 1
    assert read.output == "small"


async def test_a_raising_handler_fails_the_task_and_keeps_the_message(
    thread_storage: Storage,
) -> None:
    storage = thread_storage
    sid = await new_session(storage, LANE)

    async def boom(ctx: A2ASessionContext) -> str:
        raise RuntimeError("boom")

    result = await run_turn(storage, sid, LANE, a2a("hello", "m1"), "queue", boom)
    assert result.task.status == "failed"
    assert result.task.data.status_message["parts"][0]["text"] == "RuntimeError: boom"
    assert [i.text_content for i in await history_items(storage, sid, LANE)] == ["hello"]
    assert await run_outcomes(storage, sid) == ["failed"]


# -- the journal --


async def test_the_journal_records_a_tool_call_and_its_progress(thread_storages: Workers) -> None:
    a, _ = thread_storages
    sid = await new_session(a, LANE)
    front = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="first",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(input="found", content="done", ttft=0, duration=0),
            ]
        ),
    )

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        await ctx.update("looking")
        return "found"

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await front.run(ctx.message, message_history=ctx.message_history)

    r = await run_turn(a, sid, LANE, a2a("first", "m1"), "queue", handler)
    assert r.task.status == "completed"

    journal = await Thread(a, sid, agent_name=LANE).journal()
    assert journal[0].type == "run_started" and journal[-1].type == "run_ended"
    tools = [e.payload for e in journal if e.type == "tool_execution_updated"]
    assert [p["phase"] for p in tools] == ["started", "ended", "update"]
    assert tools[0]["name"] == "lookup" and tools[0]["callId"] == "c1"
    assert tools[1]["isError"] is False and tools[2]["text"] == "looking"
    assert "conversation_item_added" in {e.type for e in journal}


# -- interrupt --


async def test_the_interrupting_turn_outlives_the_lease_ttl(thread_storages: Workers) -> None:
    """The waiter that took the lane keeps it for as long as its turn runs: a reply slower than
    the lease TTL is still committed (the renewal timer runs on the waiter's lease too)."""
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def stuck(ctx: A2ASessionContext) -> str:
        entered.set()
        await released.wait()
        return "never sent"

    async def slow(ctx: A2ASessionContext) -> str:
        await asyncio.sleep(lease_ttl(a) + 1)
        return "late reply"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("first", "m1"), "interrupt", stuck))
    await entered.wait()
    second = asyncio.create_task(run_turn(b, sid, LANE, a2a("second", "m2"), "interrupt", slow))
    for _ in range(300):
        row = await lane_row(b, sid, LANE)
        if row is not None and (row.waiter or "").startswith("worker-b/"):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("the waiter was never set")
    released.set()

    r1 = await first
    r2 = await second
    # the first handler has no checkpoint inside it, so it completes rather than yields
    assert r1.task.status == "completed"
    assert r2.task.status == "completed" and r2.output == "late reply"
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["first", "never sent", "second", "late reply"]


async def test_interrupt_yields_and_the_newer_task_answers_both(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    tool_entered = asyncio.Event()
    waiter_set = asyncio.Event()
    front = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="first",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[FunctionToolCall(name="lookup", arguments="{}", call_id="c1")],
                ),
                FakeLLMResponse(input="found", content="never sent", ttft=0, duration=0),
                FakeLLMResponse(input="second", content="both answered", ttft=0, duration=0),
            ]
        ),
    )

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        tool_entered.set()
        await waiter_set.wait()
        return "found"

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await front.run(ctx.message, message_history=ctx.message_history)

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("first", "m1"), "interrupt", handler))
    await tool_entered.wait()
    second = asyncio.create_task(run_turn(b, sid, LANE, a2a("second", "m2"), "interrupt", handler))
    # B fails the lease and records itself as waiting; A sees it at its next checkpoint
    for _ in range(100):
        row = await lane_row(b, sid, LANE)
        if row is not None and (row.waiter or "").startswith("worker-b/"):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("the waiter was never set")
    waiter_set.set()

    r1 = await first
    r2 = await second
    assert r1.task.status == "canceled"
    assert r1.task.data.metadata[INTERRUPTED] is True
    assert r1.task.data.metadata[INTERRUPTION_REASON] == "new-message"
    assert r1.task.data.artifacts is None and r1.output is None
    assert r2.task.status == "completed" and r2.output == "both answered"
    # what ran is in the history, with no reply from the first turn
    items = await history_items(a, sid, LANE)
    assert [i.type for i in items] == [
        "message",
        "function_call",
        "function_call_output",
        "message",
        "message",
    ]
    assert items[-1].text_content == "both answered"
    assert set(await run_outcomes(a, sid)) == {"canceled", "completed"}


# -- inbox --


async def test_inbox_batches_a_burst_into_one_reply(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext) -> str:
        batches.append([m.parts[0].text for m in ctx.messages])
        assert ctx.message.content == [m.parts[0].text for m in ctx.messages]
        return "one reply"

    debounce = timedelta(seconds=0.3)
    first = asyncio.create_task(
        run_turn(a, sid, LANE, a2a("where", "m1", channel="sms"), None, handler, debounce=debounce)
    )
    await asyncio.sleep(0.05)
    second = asyncio.create_task(
        run_turn(
            b, sid, LANE, a2a("are you", "m2", channel="sms"), None, handler, debounce=debounce
        )
    )
    r1, r2 = await asyncio.gather(first, second)

    assert batches == [["where", "are you"]]
    assert r1.task.status == "completed" and not r1.task.data.artifacts
    assert r1.task.data.metadata[MERGED_INTO] == r2.task.task_id
    assert r1.answered_by == r2.task.task_id
    assert r2.task.status == "completed" and r2.output == "one reply"
    items = await history_items(a, sid, LANE)
    assert [i.role for i in items if i.type == "message"] == ["user", "assistant"]
    assert items[0].content == ["where", "are you"]
    row = await lane_row(a, sid, LANE)
    assert row is not None and row.drained_id == 2 and row.lease_until == 0


async def test_message_id_resend_returns_the_existing_task(
    thread_storage: Storage,
) -> None:
    storage = thread_storage
    sid = await new_session(storage, LANE)
    first = await run_turn(
        storage, sid, LANE, a2a("hi", "m1", channel="sms"), None, echo, debounce=timedelta(0)
    )
    again = await run_turn(
        storage, sid, LANE, a2a("hi", "m1", channel="sms"), None, echo, debounce=timedelta(0)
    )
    assert again.task.task_id == first.task.task_id and again.output == first.output
    # under interrupt the task row holds the message id
    first = await run_turn(storage, sid, LANE, a2a("yo", "m2"), "interrupt", echo)
    again = await run_turn(storage, sid, LANE, a2a("yo", "m2"), "interrupt", echo)
    assert again.task.task_id == first.task.task_id
    assert len(await history_items(storage, sid, LANE)) == 4


# -- the lease --


async def test_a_lapsed_lease_fails_the_orphan_worker_lost(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    never = asyncio.Event()
    batches: list[list[str]] = []

    async def stuck(ctx: A2ASessionContext) -> str:
        entered.set()
        await never.wait()
        return "late"

    async def handler(ctx: A2ASessionContext) -> str:
        batches.append([m.parts[0].text for m in ctx.messages])
        return "answered"

    # worker A receives, its drain picks the task up, and it dies before its commit
    dead = asyncio.create_task(
        run_turn(
            a,
            sid,
            LANE,
            a2a("first", "m1", channel="sms"),
            None,
            stuck,
            debounce=timedelta(0),
            task_id="AST_1",
        )
    )
    await entered.wait()
    dead.cancel()
    await asyncio.gather(dead, return_exceptions=True)
    await let_the_lease_lapse(a)

    # worker B takes the lane for the next message: the orphan fails worker-lost, not run again
    r2 = await run_turn(
        b, sid, LANE, a2a("second", "m2", channel="sms"), None, handler, debounce=timedelta(0)
    )
    assert batches == [["second"]]
    assert r2.task.status == "completed"

    # the inbox drain picked the row up once more and found it working: a dead holder had it
    lost = await task_of(b, sid, LANE, "AST_1")
    assert lost is not None and lost.status == "failed"
    assert lost.data.status_message["metadata"][REASON] == "worker-lost"


async def test_a_resend_after_the_holder_died_fails_worker_lost(
    thread_storage_worker: Callable[[str], Storage],
) -> None:
    """Under ``interrupt``: the holder dies mid-turn (its turn is cancelled, so the run is
    dropped without a commit and its lease lapses); a resend of the same messageId on another
    worker fails the orphaned ``working`` row ``worker-lost``, nothing runs again, and the
    message sent under a new messageId runs as a new task."""
    a, b = (thread_storage_worker(h) for h in ("wa", "wb"))
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def stuck(ctx: A2ASessionContext) -> str:
        entered.set()
        await never.wait()
        return "late"

    dead = asyncio.create_task(
        run_turn(a, sid, LANE, a2a("hello", "m1"), "interrupt", stuck, task_id="AST_x")
    )
    await entered.wait()
    dead.cancel()
    await asyncio.gather(dead, return_exceptions=True)
    await let_the_lease_lapse(a)  # the lease lapses with no renewal

    lost = await run_turn(b, sid, LANE, a2a("hello", "m1"), "interrupt", echo)
    assert lost.task.task_id == "AST_x" and lost.task.status == "failed"
    assert lost.task.data.status_message["metadata"][REASON] == "worker-lost"
    assert lost.task.data.history == [a2a("hello", "m1")]
    assert await history_items(b, sid, LANE) == []

    again = await run_turn(b, sid, LANE, a2a("hello again", "m2"), "interrupt", echo)
    assert again.task.task_id != "AST_x" and again.task.status == "completed"
    assert again.output == "reply to: hello again"
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["hello again", "reply to: hello again"]


# -- input required --


async def test_input_required_pending_call_is_the_next_messages_output(
    thread_storage: Storage,
) -> None:
    storage = thread_storage
    sid = await new_session(storage, LANE)
    fnol = Agent(
        instructions="claims",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="my car is damaged",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(name="request_photo", arguments="{}", call_id="p1")
                    ],
                ),
                FakeLLMResponse(input="here it is", content="thanks", ttft=0, duration=0),
            ]
        ),
    )

    @fnol.tool
    async def request_photo(ctx: RunContext) -> InputRequired:
        """Ask for a photo."""
        return InputRequired("Could you send a photo?")

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await fnol.run(ctx.message, message_history=ctx.message_history)

    first = await run_turn(storage, sid, LANE, a2a("my car is damaged", "m1"), "interrupt", handler)
    assert first.task.status == "input-required"
    ask = first.task.internal.ask if first.task.internal else None
    assert ask is not None
    assert (ask.question, ask.tool, ask.call_id) == (
        "Could you send a photo?",
        "request_photo",
        "p1",
    )
    assert first.task.data.status_message["parts"][0]["text"] == "Could you send a photo?"

    second = await run_turn(storage, sid, LANE, a2a("here it is", "m2"), "interrupt", handler)
    assert second.output == "thanks"
    items = await history_items(storage, sid, LANE)
    outputs = [i for i in items if i.type == "function_call_output" and i.call_id == "p1"]
    assert outputs[0].output == "here it is"
    answered = await task_of(storage, sid, LANE, first.task.task_id)
    assert answered is not None and answered.status == "completed"
    assert answered.data.metadata[MERGED_INTO] == second.task.task_id


async def test_needs_approval_asks_then_runs_on_the_approved_call(
    thread_storage: Storage,
) -> None:
    """Plan 7.4: ``needs_approval=True`` returns the ``InputRequired`` for you. The first call
    asks yes or no and does not run; the answer is its output; the model's repeat of the same
    call after the answer runs the body. The schema has no approval flag for the model."""
    storage = thread_storage
    sid = await new_session(storage, LANE)
    desk = Agent(
        instructions="front desk",
        llm=FakeLLM(
            fake_responses=[
                FakeLLMResponse(
                    input="cancel monday",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="cancel_appointment", arguments='{"date": "monday"}', call_id="c1"
                        )
                    ],
                ),
                FakeLLMResponse(
                    input="yes",
                    content="",
                    ttft=0,
                    duration=0,
                    tool_calls=[
                        FunctionToolCall(
                            name="cancel_appointment", arguments='{"date": "monday"}', call_id="c2"
                        )
                    ],
                ),
                FakeLLMResponse(input="canceled monday", content="done", ttft=0, duration=0),
            ]
        ),
    )
    ran: list[str] = []

    @desk.tool(needs_approval=True)
    async def cancel_appointment(ctx: RunContext, date: str) -> str:
        """Cancel the appointment on a date."""
        ran.append(date)
        return f"canceled {date}"

    async def handler(ctx: A2ASessionContext) -> RunResult[str]:
        return await desk.run(ctx.message, message_history=ctx.message_history)

    assert "approved" not in json.dumps(build_legacy_openai_schema(cancel_appointment))
    first = await run_turn(storage, sid, LANE, a2a("cancel monday", "m1"), "interrupt", handler)
    assert first.task.status == "input-required" and ran == []
    assert first.task.data.status_message["parts"][0]["text"] == (
        "Please confirm: Cancel the appointment on a date (date='monday'). Yes or no?"
    )
    second = await run_turn(storage, sid, LANE, a2a("yes", "m2"), "interrupt", handler)
    assert second.output == "done" and ran == ["monday"]
    items = await history_items(storage, sid, LANE)
    outputs = {i.call_id: i.output for i in items if i.type == "function_call_output"}
    assert outputs == {"c1": "yes", "c2": "canceled monday"}


# -- concurrent and reject --


async def test_concurrent_runs_append_without_a_lease(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext) -> str:
        await gate.wait()
        return "slow"

    async def fast(ctx: A2ASessionContext) -> str:
        gate.set()
        return "fast"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "concurrent", slow))
    await asyncio.sleep(0.01)
    second = await run_turn(b, sid, LANE, a2a("two", "m2"), "concurrent", fast)
    r1 = await first
    assert r1.output == "slow" and second.output == "fast"
    items = await history_items(a, sid, LANE)
    assert sorted(i.text_content or "" for i in items) == ["fast", "one", "slow", "two"]


async def test_reject_refuses_while_a_turn_runs(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext) -> str:
        entered.set()
        await gate.wait()
        return "slow"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "reject", slow))
    await entered.wait()
    with pytest.raises(RunRejected) as excinfo:
        await run_turn(b, sid, LANE, a2a("two", "m2"), "reject", echo)
    gate.set()
    r1 = await first
    assert excinfo.value.running_task_id == r1.task.task_id, "the error names the running task"
    assert r1.output == "slow"


@pytest.mark.parametrize("policy", ["interrupt", "queue", "concurrent", "reject"])
async def test_a_submitted_row_exists_while_the_turn_runs(
    policy: str, thread_storages: Workers
) -> None:
    """Plan 3.3.4: the ``submitted`` row is written before the lease is tried, so ``tasks/get``
    finds the task before the commit; the lease holder picks it up (``working``, one attempt,
    plan 3.3.2), and the commit moves that same row to its terminal state. ``concurrent`` holds
    no lease and leaves the row ``submitted``."""
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext) -> str:
        entered.set()
        await gate.wait()
        return "done"

    turn = asyncio.create_task(
        run_turn(a, sid, LANE, a2a("hello", "m1"), policy, slow, task_id="AST_early")
    )
    await entered.wait()
    # another worker, before the commit: the row is there, with the message
    during = await task_of(b, sid, LANE, "AST_early")
    assert during is not None
    if policy == "concurrent":
        assert during.status == "submitted"
    else:
        assert during.status == "working"
    assert during.message_id == "m1" and during.data.history == [a2a("hello", "m1")]
    assert during.name is None and during.internal is None
    assert during.seq == 1

    gate.set()
    result = await turn
    after = await task_of(b, sid, LANE, "AST_early")
    assert after is not None and after.status == "completed" and after.seq == 1
    assert result.task.task_id == "AST_early" and result.output == "done"
    assert len(await tasks_of(b, sid, LANE)) == 1, "the commit moved the early row, not added one"


async def test_a_lease_timeout_fails_the_early_row(thread_storages: Workers) -> None:
    """A turn that never took the lease: its early row ends ``failed`` rather than staying
    ``submitted`` for a message the caller was told failed."""
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext) -> str:
        entered.set()
        await gate.wait()
        return "slow"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "queue", slow))
    await entered.wait()
    second = await run_turn(
        b, sid, LANE, a2a("two", "m2"), "queue", echo, task_id="AST_2", wait_timeout=0.05
    )
    assert second.task.status == "failed"
    assert second.task.data.status_message["metadata"][REASON] == "lease-timeout"
    row = await task_of(b, sid, LANE, "AST_2")
    assert row is not None and row.status == "failed", "persisted, not the unwritten stand-in"
    gate.set()
    await first
    assert sorted(t.status for t in await tasks_of(b, sid, LANE)) == ["completed", "failed"]


async def test_a_rejected_message_leaves_no_submitted_row(thread_storages: Workers) -> None:
    a, b = thread_storages
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext) -> str:
        entered.set()
        await gate.wait()
        return "slow"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "reject", slow))
    await entered.wait()
    # the lane reads as leased, so the second is refused before any row is written
    with pytest.raises(RunRejected):
        await run_turn(b, sid, LANE, a2a("two", "m2"), "reject", echo, task_id="AST_2")
    assert await task_of(b, sid, LANE, "AST_2") is None
    gate.set()
    await first
    assert [t.status for t in await tasks_of(b, sid, LANE)] == ["completed"]


# -- session resolution and the Session handle --


async def test_session_resolution_follows_the_threads_service(
    thread_storage: Storage,
) -> None:
    threads = thread_storage.threads
    fresh = await _resolve_thread(threads)
    assert fresh.created and fresh.id.startswith("AT_")
    by_id = await _resolve_thread(threads, context_id=fresh.id)
    assert by_id.id == fresh.id and not by_id.created


# -- a text during a call (plan 3.3.7) --


async def test_the_rtc_door_resolves_the_call_thread(thread_storage: Storage) -> None:
    import json
    import pickle
    from types import SimpleNamespace

    from livekit.agents.storage.resolve import _CallThreadEntrypoint as _CallThreadEntrypoint

    seen: list[Thread | None] = []

    async def entrypoint(ctx: Any) -> None:
        seen.append(ctx._thread)

    door = _CallThreadEntrypoint(entrypoint, thread_storage)
    sid = await new_session(thread_storage, "desk")
    unknown = json.dumps({"thread_id": "AT_unknown"})
    for metadata in (json.dumps({"thread_id": sid}), "", unknown):
        job = SimpleNamespace(metadata=metadata, agent_name="desk")
        await door(SimpleNamespace(worker_id="w", job=job, _thread=None))  # type: ignore[arg-type]

    named, fresh, refused = seen
    # an unknown id is refused: the call runs in memory, not on a handle to no thread
    assert refused is None
    assert named is not None and named.id == sid and named._agent_name == "desk"
    assert fresh is not None and fresh.id != sid
    assert (await fresh.info()).thread_id == fresh.id
    # a job process rebuilds the storage from its configuration
    assert pickle.loads(pickle.dumps(thread_storage))._args == thread_storage._args


async def test_live_session_reloads_when_a_text_moved_the_lane(thread_storages: Workers) -> None:
    from livekit.agents import VoiceSession

    call_worker, text_worker = thread_storages
    sid = await new_session(call_worker, "desk")
    agent = Agent(instructions="desk", name="desk")
    llm = FakeLLM(
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there", ttft=0, duration=0),
            FakeLLMResponse(
                input="still there?", content="yes, and I read your text", ttft=0, duration=0
            ),
        ]
    )
    state = Patient()
    thread = Thread(call_worker, sid, agent_name="desk")
    app = AppContext(patient=state)
    async with VoiceSession(llm=llm, context=app, thread=thread) as session:
        await session.start(agent)
        assert session.thread is thread
        await session._run(user_input="hello")
        await asyncio.sleep(0.05)  # the turn's commit is chained behind the speech
        items = await history_items(text_worker, sid, "desk")
        assert [i.text_content for i in items if i.type == "message"] == ["hello", "hi there"]
        row = await lane_row(text_worker, sid, "desk")
        assert row is not None and row.lease_until == 0, "released without draining"

        # a text lands between two voice turns and is answered by another worker
        async def by_text(ctx: A2ASessionContext) -> str:
            text_app = AppContext()
            await ctx.store.load_context(text_app)
            text_app.patient.verified = 5
            await ctx.store.save_context(text_app)
            return "texted back"

        await run_turn(
            text_worker,
            sid,
            "desk",
            a2a("a text", "m1", channel="sms"),
            None,
            by_text,
            debounce=timedelta(0),
        )
        # the next voice turn's acquire returns a newer version: the call reloads once, before
        # the reply is generated, so the turn is over the texted history by the time it returns
        result = await session._run(user_input="still there?")
        assert result.output == "yes, and I read your text"
        texts = [i.text_content for i in session.history.items if i.type == "message"]
        assert texts[:4] == ["hello", "hi there", "a text", "texted back"]
        await session.store.load_context(app)
        assert app.patient.verified == 5 and session.context is app
        seen = [
            i.text_content
            for i in session._agent_chat_ctx().items
            if i.type == "message" and i.role != "system"
        ]
        assert seen[:4] == texts[:4], "the agent's working history reads the text too"

    final = await history_items(text_worker, sid, "desk")
    assert [i.text_content for i in final if i.type == "message"][-2:] == [
        "still there?",
        "yes, and I read your text",
    ]


# -- the database behind a session --


async def test_create_thread_creates_the_database(thread_storage: Storage) -> None:
    """``CreateThread`` creates the thread's database; every worker reads it after a write."""
    storage = thread_storage
    created = await storage.threads.create_thread()
    thread_id = created.thread_id
    assert created.thread_id.startswith("AT_") and created.database_id

    assert await history_items(storage, thread_id, "front") == []
    first = await run_turn(storage, thread_id, "front", a2a("hi", "m1"), "queue", echo)
    assert first.output == "reply to: hi"
    info = await storage.threads.get_thread(thread_id)
    assert info.database_id == created.database_id
    second = await run_turn(storage, thread_id, "front", a2a("again", "m2"), "queue", echo)
    assert second.output == "reply to: again"
    core = await storage._open(info.thread_id)
    items = await core.history("front")
    assert [payload_to_item(i.payload).text_content for i in items] == [
        "hi",
        "reply to: hi",
        "again",
        "reply to: again",
    ]
    tasks = await core.tasks("front")
    assert len(tasks) == 2 and all(t.status == "completed" for t in tasks)
    core.close()


async def test_handler_transactions_join_the_turn_or_commit_on_exit(
    thread_storage: Storage,
) -> None:
    """docs/context.md#application-records-and-transactions: ``transaction()`` commits or rolls
    back with the turn, like a nested savepoint; ``independent_transaction()`` commits when its
    block exits, whatever the turn does next."""
    storage = thread_storage
    thread_id = (await storage.threads.create_thread()).thread_id

    async def handler(ctx: A2ASessionContext) -> str:
        async with ctx.store.independent_transaction() as tx:
            await tx.execute("CREATE TABLE IF NOT EXISTS notes (body TEXT NOT NULL)")
            await tx.execute("INSERT INTO notes (body) VALUES (?)", ("prepared",))
        async with ctx.store.transaction() as tx:
            await tx.execute("INSERT INTO notes (body) VALUES (?)", ("with the turn",))
        with pytest.raises(RuntimeError):
            async with ctx.store.transaction() as tx:
                await tx.execute("INSERT INTO notes (body) VALUES ('rolled back')")
                raise RuntimeError("the block failed")
        async with ctx.store.independent_transaction() as tx:
            assert await tx.fetch_one("SELECT body FROM notes WHERE body = 'none'") is None
            rows = await tx.fetch("SELECT body FROM notes")
            staged_unseen = rows.count(("with the turn",)) == rows.count(("prepared",)) - 1
            assert staged_unseen, "the turn's own write is staged until it commits"
        if ctx.message.text_content == "fail":
            raise RuntimeError("the turn failed")
        return "ok"

    async def bodies() -> list[str]:
        async def read(ctx: A2ASessionContext) -> str:
            async with ctx.store.independent_transaction() as tx:
                return ",".join(str(row[0]) for row in await tx.fetch("SELECT body FROM notes"))

        result = await run_turn(
            storage, thread_id, "probe", a2a("?", f"p{id(read)}"), "queue", read
        )
        return str(result.output).split(",")

    await run_turn(storage, thread_id, LANE, a2a("go", "m1"), "queue", handler)
    assert await bodies() == ["prepared", "with the turn"]
    await run_turn(storage, thread_id, LANE, a2a("fail", "m2"), "queue", handler)
    assert await bodies() == ["prepared", "with the turn", "prepared"], "only the independent one"


async def test_the_thread_handle_pages_items_and_tasks(thread_storage: Storage) -> None:
    """``ctx.thread`` pages committed items and task records without a lease
    (docs/internals/storage.md#reading-thread-items-and-tasks)."""
    storage = thread_storage
    thread_id = (await storage.threads.create_thread()).thread_id
    for lane, text, message_id in [("front", "a", "m1"), ("back", "b", "m2"), ("front", "c", "m3")]:
        await run_turn(storage, thread_id, lane, a2a(text, message_id), "queue", echo)
    thread = Thread(storage, thread_id, agent_name="front")

    texts: list[str] = []
    after: str | None = None
    while True:
        items = await thread.list_items(agent="front", after=after, limit=3)
        texts += [payload_to_item(i.item).text_content or "" for i in items.items]
        if not items.after:
            break
        after = items.after
    assert texts == ["a", "reply to: a", "c", "reply to: c"]

    first = await thread.list_tasks(agent="front", limit=1)
    second = await thread.list_tasks(agent="front", page_token=first.next_page_token, limit=1)
    assert first.total_size == 2 and not second.next_page_token
    ids = [first.tasks[0].task_id, second.tasks[0].task_id]
    assert [t.message_id for t in first.tasks + second.tasks] == ["m1", "m3"], "oldest first"
    assert (await thread.task(ids[1])) == second.tasks[0]
    assert await thread.task("AST_missing") is None
