"""The storage package (plan 5): the Rust turn with the door's host, the lease across workers,
resolution, the voice lane; against the local agent-db of ``agentdb_fixture.py``, every test
skipped when the data plane is not up. The run itself (a raising run, a committed run id, a
stale commit) is tested in the crate: ``rust/crates/lk-agents-storage/tests``."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from livekit.agents import Agent, AgentTask, InputRequired, RunContext, _core
from livekit.agents._generated.storage import (
    LaneRow,
    Received,
    ReceiveRequest,
    SubmittedTask,
    TaskData,
)
from livekit.agents._generated.storage_run import TaskRecord
from livekit.agents._generated.storage_turn import TurnOptions, TurnResult
from livekit.agents.a2a.endpoint import A2AEndpoint
from livekit.agents.llm import ChatItem, FunctionToolCall
from livekit.agents.llm.utils import build_legacy_openai_schema
from livekit.agents.storage import AgentDBStorage, Ref, RunRejected
from livekit.agents.storage.context import A2ASessionContext, Handler, TurnHost
from livekit.agents.storage.protocol import dump_state
from livekit.agents.storage.resolve import AgentRegistry, Thread, resolve_thread

from .agentdb_fixture import let_the_lease_lapse, new_session, past_the_fresh_acquire
from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]

LANE = "front-desk"
INTERRUPTED_KEY = "livekit/interrupted"
INTERRUPTION_REASON_KEY = "livekit/interruptionReason"
MERGED_INTO_KEY = "livekit/mergedInto"


@dataclass
class Patient:
    verified: int = 0


def a2a(
    text: str, message_id: str, *, channel: str | None = None, task_id: str | None = None
) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "messageId": message_id,
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if channel:
        msg["metadata"] = {"livekit/channel": channel}
    if task_id:
        msg["taskId"] = task_id
    return msg


async def echo(ctx: A2ASessionContext[Any]) -> str:
    return "reply to: " + ctx.message.text_content


Workers = tuple[AgentDBStorage, AgentDBStorage]


async def run_turn(
    storage: AgentDBStorage,
    thread_id: str,
    lane: str,
    message: dict[str, Any],
    policy: str | None,
    handler: Handler[Any],
    *,
    state_type: type[Any] | None = None,
    debounce: timedelta | None = None,
    task_id: str | None = None,
    wait_timeout: float | None = None,
) -> TurnResult:
    """One message through the Rust turn with the door's host, as the A2A service runs it."""
    options = TurnOptions(
        policy=policy,
        debounce=debounce.total_seconds() if debounce is not None else None,
        wait_timeout=wait_timeout,
        state_default=dump_state(state_type(), state_type) if state_type is not None else None,
    )
    host = TurnHost(
        handler,
        thread=Thread(storage, thread_id, agent_name=lane),
        state_type=state_type,
    )
    db = await storage.open(thread_id)
    core = _core.Thread(db, storage.holder)
    try:
        return await _core.Turn(core, lane, host, options).run(message, task_id)
    finally:
        core.close()


async def history_items(storage: AgentDBStorage, thread_id: str, lane: str) -> list[ChatItem]:
    return (await Thread(storage, thread_id, agent_name=lane).chat_history()).items


async def tasks_of(storage: AgentDBStorage, thread_id: str, lane: str) -> list[TaskRecord]:
    return await Thread(storage, thread_id, agent_name=lane).tasks()


async def task_of(
    storage: AgentDBStorage, thread_id: str, lane: str, task_id: str
) -> TaskRecord | None:
    return next((t for t in await tasks_of(storage, thread_id, lane) if t.id == task_id), None)


async def lane_row(storage: AgentDBStorage, thread_id: str, lane: str) -> LaneRow | None:
    db = await storage.open(thread_id)
    try:
        return await db.lane(lane)
    finally:
        db.close()


async def run_outcomes(storage: AgentDBStorage, thread_id: str) -> list[str]:
    """What ``run_ended`` recorded, oldest first."""
    db = await storage.open(thread_id)
    core = _core.Thread(db, storage.holder)
    try:
        events = await core.events()
    finally:
        core.close()
    return [str(e.payload["outcome"]) for e in events if e.type == "run_ended"]


# -- the thin slice --


async def test_two_workers_share_one_session(agentdb_workers: Workers) -> None:
    """Nothing stays in memory between runs: the second worker loads what the first committed."""
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    seen: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Patient]) -> str:
        seen.append([i.text_content or "" for i in ctx.message_history.items])
        ctx.state.verified += 1
        return f"seen {ctx.state.verified}"

    first = await run_turn(a, sid, LANE, a2a("hello", "m1"), "queue", handler, state_type=Patient)
    second = await run_turn(b, sid, LANE, a2a("again", "m2"), "queue", handler, state_type=Patient)

    assert first.output == "seen 1" and second.output == "seen 2"
    assert seen == [[], ["hello", "seen 1"]]
    assert first.task.status == "completed" and first.task.data.artifacts
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["hello", "seen 1", "again", "seen 2"]
    assert await run_outcomes(b, sid) == ["completed", "completed"]


async def test_a_raising_handler_fails_the_task_and_keeps_the_message(
    agentdb_storage: AgentDBStorage,
) -> None:
    storage = agentdb_storage
    sid = await new_session(storage, LANE)

    async def boom(ctx: A2ASessionContext[Any]) -> str:
        raise RuntimeError("boom")

    result = await run_turn(storage, sid, LANE, a2a("hello", "m1"), "queue", boom)
    assert result.task.status == "failed"
    assert result.task.status_message["parts"][0]["text"] == "the agent failed before it finished"
    assert [i.text_content for i in await history_items(storage, sid, LANE)] == ["hello"]
    assert await run_outcomes(storage, sid) == ["failed"]


# -- the journal --


async def test_the_journal_records_a_tool_call_and_its_progress(agentdb_workers: Workers) -> None:
    a, _ = agentdb_workers
    sid = await new_session(a, LANE)
    front = Agent(instructions="front desk")

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        await ctx.update("looking")
        return "found"

    front.update_options(
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
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        result = await front.run(ctx.message, message_history=ctx.message_history)
        return result.output

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


async def test_the_interrupting_turn_outlives_the_lease_ttl(agentdb_workers: Workers) -> None:
    """The waiter that took the lane keeps it for as long as its turn runs: a reply slower than
    the lease TTL is still committed (the renewal timer runs on the waiter's lease too)."""
    from .agentdb_fixture import LEASE_TTL

    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    released = asyncio.Event()

    async def stuck(ctx: A2ASessionContext[Any]) -> str:
        entered.set()
        await released.wait()
        return "never sent"

    async def slow(ctx: A2ASessionContext[Any]) -> str:
        await asyncio.sleep(LEASE_TTL + 1)
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
    # the first handler has no boundary inside it, so it completes rather than yields
    assert r1.task.status == "completed"
    assert r2.task.status == "completed" and r2.output == "late reply"
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["first", "never sent", "second", "late reply"]


async def test_interrupt_yields_and_the_newer_task_answers_both(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    tool_entered = asyncio.Event()
    waiter_set = asyncio.Event()
    front = Agent(instructions="front desk")

    @front.tool
    async def lookup(ctx: RunContext) -> str:
        """Look the caller up."""
        tool_entered.set()
        await waiter_set.wait()
        return "found"

    front.update_options(
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
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        result = await front.run(ctx.message, message_history=ctx.message_history)
        return result.output

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("first", "m1"), "interrupt", handler))
    await tool_entered.wait()
    second = asyncio.create_task(run_turn(b, sid, LANE, a2a("second", "m2"), "interrupt", handler))
    # B fails the lease and records itself as waiting; A sees it at its next boundary
    for _ in range(100):
        row = await lane_row(b, sid, LANE)
        if row is not None and (row.waiter or "").startswith("worker-b/"):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("the waiter was never set")
    await past_the_fresh_acquire()
    waiter_set.set()

    r1 = await first
    r2 = await second
    assert r1.task.status == "canceled"
    assert r1.task.data.metadata[INTERRUPTED_KEY] is True
    assert r1.task.data.metadata[INTERRUPTION_REASON_KEY] == "new-message"
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


async def test_inbox_batches_a_burst_into_one_reply(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        batches.append([m.parts[0]["text"] for m in ctx.messages])
        assert ctx.message.content == [m.parts[0]["text"] for m in ctx.messages]
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
    assert r1.task.data.metadata[MERGED_INTO_KEY] == r2.task.id
    assert r1.answered_by == r2.task.id
    assert r2.task.status == "completed" and r2.output == "one reply"
    items = await history_items(a, sid, LANE)
    assert [i.role for i in items if i.type == "message"] == ["user", "assistant"]
    assert items[0].content == ["where", "are you"]
    row = await lane_row(a, sid, LANE)
    assert row is not None and row.drained_id == 2 and row.lease_until == 0


async def test_message_id_resend_returns_the_existing_task(
    agentdb_storage: AgentDBStorage,
) -> None:
    storage = agentdb_storage
    sid = await new_session(storage, LANE)
    first = await run_turn(
        storage, sid, LANE, a2a("hi", "m1", channel="sms"), None, echo, debounce=timedelta(0)
    )
    again = await run_turn(
        storage, sid, LANE, a2a("hi", "m1", channel="sms"), None, echo, debounce=timedelta(0)
    )
    assert again.task.id == first.task.id and again.output == first.output
    # under interrupt the task row holds the message id
    first = await run_turn(storage, sid, LANE, a2a("yo", "m2"), "interrupt", echo)
    again = await run_turn(storage, sid, LANE, a2a("yo", "m2"), "interrupt", echo)
    assert again.task.id == first.task.id
    assert len(await history_items(storage, sid, LANE)) == 4


# -- the lease --


def receive(task_id: str, message_id: str, text: str, *, lease_token: str) -> ReceiveRequest:
    message = a2a(text, message_id, channel="sms")
    return ReceiveRequest(
        lane=LANE,
        holder="worker-a",
        lease_token=lease_token,
        task_id=task_id,
        message_id=message_id,
        message=message,
        data=TaskData(history=[message]),
    )


async def test_a_lapsed_lease_fails_the_orphan_worker_lost(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    db = await a.open(sid)

    # worker A receives, picks the task up, and dies before its commit
    received = await db.receive(receive("AST_1", "m1", "first", lease_token="ta"))
    assert isinstance(received, Received) and received.acquired == 0
    assert await db.pickup_task(LANE, "AST_1") == "submitted"
    await let_the_lease_lapse()

    batches: list[list[str]] = []

    async def handler(ctx: A2ASessionContext[Any]) -> str:
        batches.append([m.parts[0]["text"] for m in ctx.messages])
        return "answered"

    # worker B takes the lane for the next message: the orphan fails worker-lost, not run again
    r2 = await run_turn(
        b, sid, LANE, a2a("second", "m2", channel="sms"), None, handler, debounce=timedelta(0)
    )
    assert batches == [["second"]]
    assert r2.task.status == "completed"

    # the inbox drain picked the row up once more and found it working: a dead holder had it
    lost = await task_of(b, sid, LANE, "AST_1")
    assert lost is not None and lost.status == "failed"
    assert lost.status_message["metadata"]["livekit/reason"] == "worker-lost"
    db.close()


@pytest.mark.rule("R16", "B22")
async def test_a_resend_after_the_holder_died_fails_worker_lost(
    agentdb_worker: Callable[[str], AgentDBStorage],
) -> None:
    """Under ``interrupt``: the holder dies mid-turn (its turn is cancelled, so the run is
    dropped without a commit and its lease lapses); a resend of the same messageId on another
    worker fails the orphaned ``working`` row ``worker-lost``, nothing runs again, and the
    message sent under a new messageId runs as a new task."""
    a, b = (agentdb_worker(h) for h in ("wa", "wb"))
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    never = asyncio.Event()

    async def stuck(ctx: A2ASessionContext[Any]) -> str:
        entered.set()
        await never.wait()
        return "late"

    dead = asyncio.create_task(
        run_turn(a, sid, LANE, a2a("hello", "m1"), "interrupt", stuck, task_id="AST_x")
    )
    await entered.wait()
    dead.cancel()
    await asyncio.gather(dead, return_exceptions=True)
    await let_the_lease_lapse()  # the lease lapses with no renewal

    lost = await run_turn(b, sid, LANE, a2a("hello", "m1"), "interrupt", echo)
    assert lost.task.id == "AST_x" and lost.task.status == "failed"
    assert lost.task.status_message["metadata"]["livekit/reason"] == "worker-lost"
    assert lost.task.data.history == [a2a("hello", "m1")]
    assert await history_items(b, sid, LANE) == []

    again = await run_turn(b, sid, LANE, a2a("hello again", "m2"), "interrupt", echo)
    assert again.task.id != "AST_x" and again.task.status == "completed"
    assert again.output == "reply to: hello again"
    items = await history_items(b, sid, LANE)
    assert [i.text_content for i in items] == ["hello again", "reply to: hello again"]


# -- input required --


async def test_input_required_pending_call_is_the_next_messages_output(
    agentdb_storage: AgentDBStorage,
) -> None:
    storage = agentdb_storage
    sid = await new_session(storage, LANE)
    fnol = Agent(instructions="claims")

    @fnol.tool
    async def request_photo(ctx: RunContext) -> InputRequired:
        """Ask for a photo."""
        return InputRequired("Could you send a photo?")

    fnol.update_options(
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
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        result = await fnol.run(ctx.message, message_history=ctx.message_history)
        return result.output

    first = await run_turn(storage, sid, LANE, a2a("my car is damaged", "m1"), "interrupt", handler)
    assert first.task.status == "input-required"
    assert first.task.internal is not None and first.task.internal.ask == {
        "question": "Could you send a photo?",
        "tool": "request_photo",
        "call_id": "p1",
    }
    assert first.task.status_message["parts"][0]["text"] == "Could you send a photo?"

    second = await run_turn(storage, sid, LANE, a2a("here it is", "m2"), "interrupt", handler)
    assert second.output == "thanks"
    items = await history_items(storage, sid, LANE)
    outputs = [i for i in items if i.type == "function_call_output" and i.call_id == "p1"]
    assert outputs[0].output == "here it is"
    answered = await task_of(storage, sid, LANE, first.task.id)
    assert answered is not None and answered.status == "completed"
    assert answered.data.metadata[MERGED_INTO_KEY] == second.task.id


@pytest.mark.xfail(
    strict=True,
    reason="approval per task: the old turn engine runs the answer as a new task, whose message "
    "cannot approve the read-back; the Rust continuation replaces this test",
)
async def test_requires_approval_asks_then_runs_on_the_approved_call(
    agentdb_storage: AgentDBStorage,
) -> None:
    """Plan 7.4: ``requires_approval=True`` returns the ``InputRequired`` for you. The first call
    asks yes or no and does not run; the answer is its output; the model's repeat of the same
    call after the answer runs the body. The schema has no approval flag for the model."""
    storage = agentdb_storage
    sid = await new_session(storage, LANE)
    desk = Agent(instructions="front desk")
    ran: list[str] = []

    @desk.tool(requires_approval=True)
    async def cancel_appointment(ctx: RunContext, date: str) -> str:
        """Cancel the appointment on a date."""
        ran.append(date)
        return f"canceled {date}"

    desk.update_options(
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
                            name="cancel_appointment",
                            arguments='{"date": "monday"}',
                            call_id="c2",
                        )
                    ],
                ),
                FakeLLMResponse(input="canceled monday", content="done", ttft=0, duration=0),
            ]
        )
    )

    async def handler(ctx: A2ASessionContext[Any]) -> str | None:
        return (await desk.run(ctx.message, message_history=ctx.message_history)).output

    assert "approved" not in json.dumps(build_legacy_openai_schema(cancel_appointment))
    first = await run_turn(storage, sid, LANE, a2a("cancel monday", "m1"), "interrupt", handler)
    assert first.task.status == "input-required" and ran == []
    assert first.task.status_message["parts"][0]["text"] == (
        "Please confirm: Cancel the appointment on a date (date='monday'). Yes or no?"
    )
    second = await run_turn(storage, sid, LANE, a2a("yes", "m2"), "interrupt", handler)
    assert second.output == "done" and ran == ["monday"]
    items = await history_items(storage, sid, LANE)
    outputs = {i.call_id: i.output for i in items if i.type == "function_call_output"}
    assert outputs == {"c1": "yes", "c2": "canceled monday"}


# -- concurrent and reject --


async def test_concurrent_runs_append_without_a_lease(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext[Any]) -> str:
        await gate.wait()
        return "slow"

    async def fast(ctx: A2ASessionContext[Any]) -> str:
        gate.set()
        return "fast"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "concurrent", slow))
    await asyncio.sleep(0.01)
    second = await run_turn(b, sid, LANE, a2a("two", "m2"), "concurrent", fast)
    r1 = await first
    assert r1.output == "slow" and second.output == "fast"
    items = await history_items(a, sid, LANE)
    assert sorted(i.text_content or "" for i in items) == ["fast", "one", "slow", "two"]
    with pytest.raises(ValueError):
        A2AEndpoint(echo, endpoint="x", description="x", state=Patient, inbox="concurrent")


async def test_reject_refuses_while_a_turn_runs(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext[Any]) -> str:
        entered.set()
        await gate.wait()
        return "slow"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "reject", slow))
    await entered.wait()
    with pytest.raises(RunRejected) as excinfo:
        await run_turn(b, sid, LANE, a2a("two", "m2"), "reject", echo)
    gate.set()
    r1 = await first
    assert excinfo.value.running_task_id == r1.task.id, "the error names the running task"
    assert r1.output == "slow"


@pytest.mark.parametrize("policy", ["interrupt", "queue", "concurrent", "reject"])
async def test_a_submitted_row_exists_while_the_turn_runs(
    policy: str, agentdb_workers: Workers
) -> None:
    """Plan 3.3.4: the ``submitted`` row is written before the lease is tried, so ``tasks/get``
    finds the task before the commit; the lease holder picks it up (``working``, one attempt,
    plan 3.3.2), and the commit moves that same row to its terminal state. ``concurrent`` holds
    no lease and leaves the row ``submitted``."""
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext[Any]) -> str:
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
    assert result.task.id == "AST_early" and result.output == "done"
    assert len(await tasks_of(b, sid, LANE)) == 1, "the commit moved the early row, not added one"
    # a resend of the message inserts no second row
    db = await b.open(sid)
    assert not await db.insert_submitted_task(
        SubmittedTask(lane=LANE, task_id="AST_other", message_id="m1")
    )
    db.close()


async def test_a_lease_timeout_fails_the_early_row(agentdb_workers: Workers) -> None:
    """A turn that never took the lease: its early row ends ``failed`` rather than staying
    ``submitted`` for a message the caller was told failed."""
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext[Any]) -> str:
        entered.set()
        await gate.wait()
        return "slow"

    first = asyncio.create_task(run_turn(a, sid, LANE, a2a("one", "m1"), "queue", slow))
    await entered.wait()
    second = await run_turn(
        b, sid, LANE, a2a("two", "m2"), "queue", echo, task_id="AST_2", wait_timeout=0.05
    )
    assert second.task.status == "failed"
    assert second.task.status_message["metadata"]["livekit/reason"] == "lease-timeout"
    row = await task_of(b, sid, LANE, "AST_2")
    assert row is not None and row.status == "failed", "persisted, not the unwritten stand-in"
    gate.set()
    await first
    assert sorted(t.status for t in await tasks_of(b, sid, LANE)) == ["completed", "failed"]


async def test_a_rejected_message_leaves_no_submitted_row(agentdb_workers: Workers) -> None:
    a, b = agentdb_workers
    sid = await new_session(a, LANE)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def slow(ctx: A2ASessionContext[Any]) -> str:
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


# -- refs --


@dataclass
class Verify(AgentTask[bool]):
    code: int = 0

    def __post_init__(self) -> None:
        AgentTask.__init__(self, instructions="verify")


def test_ref_rebuild_drops_what_a_deploy_renamed(caplog: pytest.LogCaptureFixture) -> None:
    front = Agent(instructions="x", name="front")
    registry = AgentRegistry()
    registry.register(front)

    stack = [
        Ref(name="front"),
        Ref(name="Verify", args={"code": 7}),
        Ref(name="RenamedAway", args={"x": 1}),
        Ref(name="front"),
    ]
    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        agents = registry.resolve(stack)
    assert agents[0] is front
    assert isinstance(agents[1], Verify) and agents[1].code == 7
    assert len(agents) == 2, "the unknown class and everything above it are dropped"
    assert any("cannot rebuild" in r.getMessage() for r in caplog.records)
    speaking = registry.speaking(stack)
    assert isinstance(speaking, Verify) and speaking.code == 7
    assert registry.speaking([Ref(name="nobody")], default=front) is front


# -- session resolution and the Session handle --


async def test_session_resolution_follows_the_threads_service(
    agentdb_storage: AgentDBStorage,
) -> None:
    threads = agentdb_storage.threads
    fresh = await resolve_thread(threads)
    assert fresh.created and fresh.id.startswith("AT_")
    by_id = await resolve_thread(threads, context_id=fresh.id)
    assert by_id.id == fresh.id and not by_id.created


# -- a text during a call (plan 3.3.7) --


async def test_the_rtc_door_resolves_the_call_thread(agentdb_storage: AgentDBStorage) -> None:
    import json
    import pickle
    from types import SimpleNamespace

    from livekit.agents.worker import _CallThreadEntrypoint

    seen: list[Thread | None] = []

    async def entrypoint(ctx: Any) -> None:
        seen.append(ctx._thread)

    door = _CallThreadEntrypoint(entrypoint, agentdb_storage)
    sid = await new_session(agentdb_storage, "desk")
    for metadata in (json.dumps({"thread_id": sid}), ""):
        job = SimpleNamespace(metadata=metadata, agent_name="desk")
        await door(SimpleNamespace(worker_id="w", job=job, _thread=None))  # type: ignore[arg-type]

    named, fresh = seen
    assert named is not None and named.id == sid and named.agent_name == "desk"
    assert fresh is not None and fresh.id != sid
    assert (await fresh.info()).thread_id == fresh.id
    # a job process rebuilds the storage from its configuration, its project's URL in its env
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LIVEKIT_URL", agentdb_storage.url)
        assert pickle.loads(pickle.dumps(agentdb_storage))._args == agentdb_storage._args


async def test_live_session_reloads_when_a_text_moved_the_lane(agentdb_workers: Workers) -> None:
    from livekit.agents import LiveSession

    call_worker, text_worker = agentdb_workers
    sid = await new_session(call_worker, "desk")
    agent = Agent(instructions="desk", name="desk", state_type=Patient)
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
    async with LiveSession(llm=llm, state=state, thread=thread) as session:
        await session.start(agent)
        assert session.thread is thread
        await session.run(user_input="hello")
        await asyncio.sleep(0.05)  # the turn's commit is chained behind the speech
        items = await history_items(text_worker, sid, "desk")
        assert [i.text_content for i in items if i.type == "message"] == ["hello", "hi there"]
        row = await lane_row(text_worker, sid, "desk")
        assert row is not None and row.lease_until == 0, "released without draining"

        # a text lands between two voice turns and is answered by another worker
        async def by_text(ctx: A2ASessionContext[Patient]) -> str:
            ctx.state.verified = 5
            return "texted back"

        await run_turn(
            text_worker,
            sid,
            "desk",
            a2a("a text", "m1", channel="sms"),
            None,
            by_text,
            state_type=Patient,
            debounce=timedelta(0),
        )
        # the next voice turn's acquire returns a newer version: the call reloads once, before
        # the reply is generated, so the turn is over the texted history by the time it returns
        result = await session.run(user_input="still there?")
        assert result.output == "yes, and I read your text"
        texts = [i.text_content for i in session.history.items if i.type == "message"]
        assert texts[:4] == ["hello", "hi there", "a text", "texted back"]
        assert state.verified == 5 and session.state is state
        assert session._activity is not None
        seen = [
            i.text_content
            for i in session._activity.chat_ctx.items
            if i.type == "message" and i.role != "system"
        ]
        assert seen[:4] == texts[:4], "the agent's working history reads the text too"

    final = await history_items(text_worker, sid, "desk")
    assert [i.text_content for i in final if i.type == "message"][-2:] == [
        "still there?",
        "yes, and I read your text",
    ]


# -- the database behind a session --


async def test_create_thread_creates_the_database(agentdb_storage: AgentDBStorage) -> None:
    """``CreateThread`` creates the thread's database; every worker reads it after a write."""
    storage = agentdb_storage
    created = await storage.threads.create_thread()
    thread_id = created.thread_id
    assert created.database_id.startswith("DB_")

    assert await history_items(storage, thread_id, "front") == []
    first = await run_turn(storage, thread_id, "front", a2a("hi", "m1"), "queue", echo)
    assert first.output == "reply to: hi"
    info = await storage.get_thread(thread_id, fresh=True)
    assert info.database_id == created.database_id
    second = await run_turn(storage, thread_id, "front", a2a("again", "m2"), "queue", echo)
    assert second.output == "reply to: again"
    db = await storage.open_database(info.database_id)
    core = _core.Thread(db, storage.holder)
    items = await core.history("front")
    assert [i.payload["content"][0] for i in items] == [
        "hi",
        "reply to: hi",
        "again",
        "reply to: again",
    ]
    tasks = await core.tasks("front")
    assert len(tasks) == 2 and all(t.status == "completed" for t in tasks)
    core.close()
