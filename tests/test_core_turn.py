"""The Rust turn through ``livekit.agents._core``: a ``Session`` over a database
``Storage`` opened, a ``Turn`` whose host is a Python object, one message answered, and a
second message under ``interrupt`` that yields the first turn. Against the local agent-db of
``storage_fixture.py``; skipped when the data plane is not up."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from livekit.agents import _core
from livekit.agents._generated.a2a_client import TextPart
from livekit.agents._generated.storage_class import ItemRow, RunUsage
from livekit.agents._generated.storage_options import TurnOptions
from livekit.agents._generated.storage_turn import HandlerOutput, TurnContext
from livekit.agents.a2a.extension import INTERRUPTED, INTERRUPTION_REASON
from livekit.agents.llm import ChatMessage
from livekit.agents.storage import Storage
from livekit.agents.storage.context import _TurnStream
from livekit.agents.storage.protocol import item_to_payload, payload_to_item

pytestmark = pytest.mark.unit

LANE = "front-desk"


def a2a(text: str, message_id: str) -> dict[str, Any]:
    return {"messageId": message_id, "role": "ROLE_USER", "parts": [{"text": text}]}


def assistant_item(text: str) -> dict[str, Any]:
    return item_to_payload(ChatMessage(role="assistant", content=[text]))


def role_and_text(item: ItemRow) -> tuple[str, str]:
    message = payload_to_item(item.payload)
    assert isinstance(message, ChatMessage)
    return message.role, message.text_content or ""


class Host:
    """Echoes the message; with ``gate`` set it sleeps until told and asks for a checkpoint."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.entered = asyncio.Event()
        self.contexts: list[TurnContext] = []
        self.yielded: list[bool] = []

    async def _core_handle(self, ctx: TurnContext) -> _TurnStream:
        return _TurnStream(lambda checkpoint: self.run(ctx, checkpoint))

    async def run(
        self, ctx: TurnContext, checkpoint: Callable[[], Awaitable[bool]]
    ) -> HandlerOutput:
        self.contexts.append(ctx)
        part = ctx.batch[-1].parts[0]
        assert isinstance(part, TextPart)
        text = part.text
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
            self.yielded.append(await checkpoint())
            if self.yielded[-1]:
                return HandlerOutput(
                    output=None,
                    new_items=[ctx.message],
                    state=None,
                    scope_writes=[],
                    app_writes=[],
                    usage=RunUsage(),
                    input_required=None,
                    pending_call=None,
                    interrupted=True,
                    metadata={},
                    error=None,
                    tool_executions=[],
                )
        return HandlerOutput(
            output="reply to: " + text,
            new_items=[ctx.message, assistant_item("reply to: " + text)],
            usage=RunUsage(requests=1, input_tokens=3, output_tokens=4, tool_calls=0),
            state=None,
            scope_writes=[],
            app_writes=[],
            input_required=None,
            pending_call=None,
            interrupted=False,
            metadata={},
            error=None,
            tool_executions=[],
        )


async def thread_of(storage: Storage) -> str:
    return (await storage.threads.create_thread()).thread_id


async def test_one_message_through_the_rust_turn(thread_storage: Storage) -> None:
    thread_id = await thread_of(thread_storage)
    session = await thread_storage._open(thread_id)
    assert session.id == thread_id
    assert session.holder.startswith("worker-a/")

    host = Host()
    turn = _core.Turn(session, LANE, host, TurnOptions(policy="interrupt", debounce=0))
    assert turn.lane == LANE and turn.policy_for("sms") == "interrupt"

    result = await turn.run(a2a("hello", "m1"))
    assert result.task.status == "completed"
    assert result.task.message_id == "m1"
    assert result.output == "reply to: hello"
    assert result.thread_id == thread_id
    assert result.answered_by == result.task.task_id
    assert result.run_id == result.task.task_id

    # the context the handler saw
    (ctx,) = host.contexts
    assert ctx.thread_id == thread_id and ctx.lane == LANE
    assert ctx.task_id == result.task.task_id
    assert ctx.history == [] and ctx.state is None and ctx.caller_history is None
    assert payload_to_item(ctx.message).model_dump(include={"role", "content"}) == {
        "role": "user",
        "content": ["hello"],
    }
    assert [m.message_id for m in ctx.batch] == ["m1"]
    assert [t.status for t in ctx.tasks] == ["working"]

    items = await session.history(LANE)
    assert [role_and_text(i) for i in items] == [
        ("user", "hello"),
        ("assistant", "reply to: hello"),
    ]
    tasks = await session.tasks(LANE)
    assert [t.task_id for t in tasks] == [result.task.task_id]
    lanes = await session.lanes()
    assert [lane.lane for lane in lanes] == [LANE]
    assert lanes[0].lease_until <= time.time(), "the lease was released by the commit"
    events = await session.journal()
    assert [e.type for e in events][:1] == ["run_started"]
    assert "run_ended" in {e.type for e in events}
    session.close()


async def test_debounce_delays_the_turn(thread_storage: Storage) -> None:
    session = await thread_storage._open(await thread_of(thread_storage))
    turn = _core.Turn(session, LANE, Host(), TurnOptions(policy="queue", debounce=0.4))
    started = time.monotonic()
    result = await turn.run(a2a("hello", "m1"))
    assert result.task.status == "completed"
    assert time.monotonic() - started >= 0.4
    session.close()


async def test_a_waiter_past_wait_timeout_fails_lease_timeout(
    thread_storages: tuple[Storage, Storage],
) -> None:
    storage_a, storage_b = thread_storages
    thread_id = await thread_of(storage_a)
    session_a = await storage_a._open(thread_id)
    gate = asyncio.Event()
    host_a = Host(gate)
    turn_a = _core.Turn(session_a, LANE, host_a, TurnOptions(policy="queue", debounce=0))
    first = asyncio.ensure_future(turn_a.run(a2a("first", "m1")))
    await asyncio.wait_for(host_a.entered.wait(), 10)

    session_b = await storage_b._open(thread_id)
    options = TurnOptions(policy="queue", debounce=0, wait_timeout=0.5)
    turn_b = _core.Turn(session_b, LANE, Host(), options)
    second = await asyncio.wait_for(turn_b.run(a2a("second", "m2")), 30)
    assert second.task.status == "failed"
    status_message = second.task.data.status_message
    assert status_message is not None
    assert "lease-timeout" in status_message["metadata"].values()

    gate.set()
    assert (await asyncio.wait_for(first, 30)).task.status == "completed"
    session_a.close()
    session_b.close()


async def test_a_second_message_under_interrupt_yields_the_first(
    thread_storages: tuple[Storage, Storage],
) -> None:
    storage_a, storage_b = thread_storages
    thread_id = await thread_of(storage_a)
    session_a = await storage_a._open(thread_id)
    gate = asyncio.Event()
    host_a = Host(gate)
    turn_a = _core.Turn(session_a, LANE, host_a, TurnOptions(policy="interrupt", debounce=0))
    # the first acquire writes the schema; the second worker dials after it
    first = asyncio.ensure_future(turn_a.run(a2a("first", "m1")))
    await asyncio.wait_for(host_a.entered.wait(), 10)

    session_b = await storage_b._open(thread_id)
    host_b = Host()
    turn_b = _core.Turn(session_b, LANE, host_b, TurnOptions(policy="interrupt", debounce=0))
    second = asyncio.ensure_future(turn_b.run(a2a("second", "m2")))

    # B fails the lease and records itself as waiting; A sees it at its next checkpoint
    for _ in range(500):
        lanes = await session_b.lanes()
        if any((lane.waiter or "").startswith("worker-b/") for lane in lanes):
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("the waiter was never set")
    gate.set()

    r1 = await asyncio.wait_for(first, 30)
    r2 = await asyncio.wait_for(second, 30)
    assert r1.task.status == "canceled"
    assert r1.task.data.metadata == {INTERRUPTED: True, INTERRUPTION_REASON: "new-message"}
    assert r1.output is None
    assert r2.task.status == "completed"
    assert r2.output == "reply to: second"
    assert host_a.yielded == [True]

    items = await session_a.history(LANE)
    assert [role_and_text(i) for i in items] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "reply to: second"),
    ]
    # the second handler saw the first message in its history
    assert [role_and_text(i)[1] for i in host_b.contexts[0].history] == ["first"]
    session_a.close()
    session_b.close()


async def test_turn_options_apply_scalar_and_channel_policies(thread_storage: Storage) -> None:
    session = await thread_storage._open(await thread_of(thread_storage))
    try:
        all_channels = _core.Turn(
            session,
            LANE,
            Host(),
            TurnOptions(policy="interrupt", debounce=0, wait_timeout=1),
        )
        assert all_channels.policy_for("sms") == "interrupt"
        assert all_channels.policy_for("audio") == "interrupt"

        per_channel = _core.Turn(
            session,
            LANE,
            Host(),
            TurnOptions(policy={"sms": "batch"}, debounce={"sms": 2}, wait_timeout=1.5),
        )
        assert per_channel.policy_for("sms") == "batch"
        assert per_channel.policy_for("audio") == "interrupt"

        with pytest.raises(TypeError, match="TurnOptions.policy") as invalid:
            _core.Turn(
                session,
                LANE,
                Host(),
                TurnOptions(policy="later"),  # type: ignore[arg-type]
            )
        assert isinstance(invalid.value.__cause__, ValueError)
        assert "later" in str(invalid.value.__cause__)
        with pytest.raises(TypeError, match="expected TurnOptions"):
            _core.Turn(
                session,
                LANE,
                Host(),
                {"policy": "interrupt"},  # type: ignore[arg-type]
            )
        with pytest.raises(TypeError, match="TurnOptions.debounce") as negative:
            _core.Turn(session, LANE, Host(), TurnOptions(debounce=-1))
        assert isinstance(negative.value.__cause__, ValueError)
    finally:
        session.close()
