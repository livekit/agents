"""Tests for the async-tool task tracker.

These exercise the tracker against a real ``RunContext`` (standalone mode, no executor)
so the synthetic call_ids it computes are validated against the ones the framework
actually assigns to progress updates and the final return.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from async_task_ui import AsyncTaskUI

from livekit.agents.llm import FunctionCall
from livekit.agents.voice.events import RunContext


class _FakeRoom:
    def __init__(self) -> None:
        self.local_participant = Mock()
        self.remote_participants: dict = {}


class _FakeSpeech:
    """Stands in for a SpeechHandle's done lifecycle."""

    def __init__(self, *, interrupted: bool = False, chat_items: list | None = None) -> None:
        self.interrupted = interrupted
        self.chat_items = chat_items if chat_items is not None else ["spoken"]
        self._cb = None

    def add_done_callback(self, cb) -> None:
        self._cb = cb

    def fire(self) -> None:
        assert self._cb is not None
        self._cb(self)


def _ctx(call_id: str, name: str, arguments: str) -> RunContext:
    fc = FunctionCall(call_id=call_id, name=name, arguments=arguments)
    return RunContext(session=Mock(), speech_handle=Mock(num_steps=1), function_call=fc)


def _tasks(ui: AsyncTaskUI) -> list[dict]:
    return json.loads(ui._snapshot())["tasks"]


@pytest.mark.asyncio
async def test_updates_and_result_lifecycle() -> None:
    ui = AsyncTaskUI(room=_FakeRoom())
    ctx = _ctx("call_1", "book_flight", '{"origin": "NYC", "destination": "Tokyo"}')

    async def tool(ctx: RunContext) -> str:
        await ctx.update("Searching flights...")
        await ctx.update("Found 3 options...")
        return "Flight booked!"

    await ui._wrap(tool)(ctx)

    task = _tasks(ui)[0]
    assert task["status"] == "done"
    assert task["result"] == "Flight booked!"
    assert task["args"] == {"origin": "NYC", "destination": "Tokyo"}

    # ids must match the synthetic ids the framework assigned to the real update pairs
    assert [p[0].call_id for p in ctx._updates] == ["call_1", "call_1_update_1"]
    assert [(e["id"], e["kind"], e["reply"]) for e in task["timeline"]] == [
        ("call_1", "update", "inline"),  # first update is voiced inline
        ("call_1_update_1", "update", "pending"),  # later update is deferred
        ("call_1_final", "result", "pending"),  # final return is deferred
    ]

    # the deferred reply starts generating, then is voiced
    speech = _FakeSpeech()
    ui._on_async_tool_reply(
        SimpleNamespace(call_ids=["call_1_update_1", "call_1_final"], speech_handle=speech)
    )
    replies = {e["id"]: e["reply"] for e in _tasks(ui)[0]["timeline"]}
    assert replies == {
        "call_1": "inline",
        "call_1_update_1": "generating",
        "call_1_final": "generating",
    }

    speech.fire()
    replies = {e["id"]: e["reply"] for e in _tasks(ui)[0]["timeline"]}
    assert replies["call_1_update_1"] == "spoken"
    assert replies["call_1_final"] == "spoken"


@pytest.mark.asyncio
async def test_interrupted_and_covered_replies() -> None:
    ui = AsyncTaskUI(room=_FakeRoom())
    ctx = _ctx("call_2", "tour_guide", "{}")

    async def tool(ctx: RunContext) -> str:
        await ctx.update("Looking up spots...")
        return "Here are some tips."

    await ui._wrap(tool)(ctx)

    interrupted = _FakeSpeech(interrupted=True)
    ui._on_async_tool_reply(SimpleNamespace(call_ids=["call_2_final"], speech_handle=interrupted))
    interrupted.fire()
    assert _entry(ui, "call_2_final")["reply"] == "interrupted"

    # an empty reply (no chat items) means the agent considered it already covered
    covered = _FakeSpeech(chat_items=[])
    ui._on_async_tool_reply(SimpleNamespace(call_ids=["call_2_final"], speech_handle=covered))
    covered.fire()
    assert _entry(ui, "call_2_final")["reply"] == "covered"


@pytest.mark.asyncio
async def test_no_updates_has_inline_result_only() -> None:
    ui = AsyncTaskUI(room=_FakeRoom())
    ctx = _ctx("call_3", "tour_guide", "{}")

    async def tool(ctx: RunContext) -> str:
        return "Immediate answer."

    await ui._wrap(tool)(ctx)

    task = _tasks(ui)[0]
    assert task["status"] == "done"
    assert task["result"] == "Immediate answer."
    assert task["timeline"] == []  # no deferred entry — voiced as part of the normal turn


@pytest.mark.asyncio
async def test_error_and_cancellation() -> None:
    ui = AsyncTaskUI(room=_FakeRoom())

    async def boom(ctx: RunContext) -> str:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await ui._wrap(boom)(_ctx("err", "book_flight", "{}"))
    assert _tasks(ui)[0]["status"] == "error"
    assert _tasks(ui)[0]["error"] == "nope"

    import asyncio

    async def cancelled(ctx: RunContext) -> str:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await ui._wrap(cancelled)(_ctx("cxl", "book_flight", "{}"))
    assert _tasks(ui)[1]["status"] == "cancelled"


def _entry(ui: AsyncTaskUI, entry_id: str) -> dict:
    for task in _tasks(ui):
        for entry in task["timeline"]:
            if entry["id"] == entry_id:
                return entry
    raise AssertionError(f"entry {entry_id} not found")
