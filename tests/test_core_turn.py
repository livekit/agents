"""The Rust turn through ``livekit.agents._core`` (plan 13.8): a ``Session`` over a database
``AgentDBStorage`` opened, a ``Turn`` whose host is a Python object, one message answered, and a
second message under ``interrupt`` that yields the first turn. Against the local agent-db of
``agentdb_fixture.py``; skipped when the data plane is not up."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import pytest

from livekit.agents import _core
from livekit.agents._core import Boundary
from livekit.agents._generated.storage_run import Item
from livekit.agents._generated.storage_turn import HandlerOutput, TurnContext, TurnOptions
from livekit.agents.storage import AgentDBStorage

from .agentdb_fixture import past_the_fresh_acquire

pytestmark = pytest.mark.unit

LANE = "front-desk"
INTERRUPTED = "livekit/interrupted"
INTERRUPTION_REASON = "livekit/interruptionReason"


def a2a(text: str, message_id: str) -> dict[str, Any]:
    return {"messageId": message_id, "role": "user", "parts": [{"kind": "text", "text": text}]}


def assistant_item(text: str) -> dict[str, Any]:
    return {
        "id": f"item_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "content": [text],
        "interrupted": False,
        "extra": {},
        "created_at": time.time(),
    }


def text_of(item: Item) -> str:
    return "\n".join(c for c in item.payload["content"] if isinstance(c, str))


class Host:
    """Echoes the message; with ``gate`` set it sleeps until told and polls the boundary."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.entered = asyncio.Event()
        self.contexts: list[tuple[TurnContext, Boundary]] = []
        self.updates: list[tuple[str, str, bool]] = []

    async def handle(self, ctx: TurnContext, boundary: Boundary) -> HandlerOutput:
        self.contexts.append((ctx, boundary))
        text = ctx.batch[-1].parts[0]["text"]
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
            if await boundary.check():
                return HandlerOutput(new_items=[ctx.message], interrupted=True)
        return HandlerOutput(
            output="reply to: " + text,
            new_items=[ctx.message, assistant_item("reply to: " + text)],
            usage={"requests": 1, "input_tokens": 3, "output_tokens": 4, "tool_calls": 0},
        )

    async def on_update(self, task_id: str, text: str) -> None:
        self.updates.append((task_id, text))

    async def on_token(self, task_id: str, delta: str) -> None:
        pass


async def database_of(storage: AgentDBStorage) -> str:
    return (await storage.threads.create_thread()).database_id


async def test_one_message_through_the_rust_turn(agentdb_storage: AgentDBStorage) -> None:
    database_id = await database_of(agentdb_storage)
    db = await agentdb_storage.open_database(database_id)
    session = _core.Thread(db, "worker-a")
    assert session.id == database_id
    assert session.holder.startswith("worker-a/")
    assert session.now_ms > 1_000_000_000_000

    host = Host()
    turn = _core.Turn(session, LANE, host, TurnOptions(policy="interrupt", debounce=0))
    assert turn.lane == LANE and turn.policy_for("sms") == "interrupt"

    result = await turn.run(a2a("hello", "m1"))
    assert result.task.status == "completed"
    assert result.task.message_id == "m1"
    assert result.output == "reply to: hello"
    assert result.thread_id == database_id
    assert result.answered_by == result.task.id
    assert result.run_id == result.task.id

    # the context the handler saw
    ((ctx, boundary),) = host.contexts
    assert ctx.thread_id == database_id and ctx.lane == LANE
    assert ctx.task_id == result.task.id
    assert ctx.history == [] and ctx.state is None and ctx.caller_history is None
    assert ctx.message["role"] == "user" and ctx.message["content"] == ["hello"]
    assert [m.message_id for m in ctx.batch] == ["m1"]
    assert [t.status for t in ctx.tasks] == ["working"]
    assert boundary.task_id == result.task.id
    assert not boundary.tripped and not boundary.client_canceled

    items = await session.history(LANE)
    assert [(i.payload["role"], text_of(i)) for i in items] == [
        ("user", "hello"),
        ("assistant", "reply to: hello"),
    ]
    tasks = await session.tasks(LANE)
    assert [t.id for t in tasks] == [result.task.id]
    lanes = await session.lanes()
    assert [lane.name for lane in lanes] == [LANE]
    assert lanes[0].lease_until <= time.time(), "the lease was released by the commit"
    events = await session.events()
    assert [e.type for e in events][:1] == ["run_started"]
    assert "run_ended" in {e.type for e in events}
    session.close()


async def test_a_second_message_under_interrupt_yields_the_first(
    agentdb_workers: tuple[AgentDBStorage, AgentDBStorage],
) -> None:
    storage_a, storage_b = agentdb_workers
    database_id = await database_of(storage_a)
    db_a = await storage_a.open_database(database_id)
    session_a = _core.Thread(db_a, "worker-a")
    gate = asyncio.Event()
    host_a = Host(gate)
    turn_a = _core.Turn(session_a, LANE, host_a, TurnOptions(policy="interrupt", debounce=0))
    # the first acquire writes the schema; the second worker dials after it
    first = asyncio.ensure_future(turn_a.run(a2a("first", "m1")))
    await asyncio.wait_for(host_a.entered.wait(), 10)

    db_b = await storage_b.open_database(database_id)
    session_b = _core.Thread(db_b, "worker-b")
    host_b = Host()
    turn_b = _core.Turn(session_b, LANE, host_b, TurnOptions(policy="interrupt", debounce=0))
    second = asyncio.ensure_future(turn_b.run(a2a("second", "m2")))

    # B fails the lease and records itself as waiting; A sees it at its next boundary
    for _ in range(500):
        lanes = await session_b.lanes()
        if any((lane.waiter or "").startswith("worker-b/") for lane in lanes):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the waiter was never set")
    await past_the_fresh_acquire()
    gate.set()

    r1 = await asyncio.wait_for(first, 30)
    r2 = await asyncio.wait_for(second, 30)
    assert r1.task.status == "canceled"
    assert r1.task.data.metadata == {INTERRUPTED: True, INTERRUPTION_REASON: "new-message"}
    assert r1.output is None
    assert r2.task.status == "completed"
    assert r2.output == "reply to: second"
    boundary_a = host_a.contexts[0][1]
    assert boundary_a.tripped and not boundary_a.client_canceled

    items = await session_a.history(LANE)
    assert [(i.payload["role"], text_of(i)) for i in items] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "reply to: second"),
    ]
    # the second handler saw the first message in its history
    assert [text_of(i) for i in host_b.contexts[0][0].history] == ["first"]
    session_a.close()
    session_b.close()
