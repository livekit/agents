from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from livekit.plugins.phonic.realtime import RealtimeModel, RealtimeSession

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_say_cancellation_cancels_pending_send_task() -> None:
    model = RealtimeModel(api_key="fake")
    sess = RealtimeSession(model)
    sess._socket = AsyncMock()

    # Blocked on readiness
    assert not sess._ready_to_start.is_set()
    fut = sess.say("hello")
    assert not fut.done()

    # Interruption cancels the future
    fut.cancel()

    # Readiness opens after cancellation
    sess._ready_to_start.set()
    await asyncio.sleep(0.02)

    # Scripted text must never be sent
    sess._socket.send_say.assert_not_called()
    assert sess._generate_reply_task is not None
    assert sess._generate_reply_task.cancelled() or sess._generate_reply_task.done()


@pytest.mark.asyncio
async def test_say_interruption_cancels_pending_send_task() -> None:
    model = RealtimeModel(api_key="fake")
    sess = RealtimeSession(model)
    sess._socket = AsyncMock()

    assert not sess._ready_to_start.is_set()
    fut = sess.say("hello")
    assert not fut.done()

    # Interrupt session directly
    sess.interrupt()

    # Readiness opens after interrupt
    sess._ready_to_start.set()
    await asyncio.sleep(0.02)

    # Scripted text must never be sent
    sess._socket.send_say.assert_not_called()
    assert fut.cancelled()
    assert sess._generate_reply_task is not None
    assert sess._generate_reply_task.cancelled() or sess._generate_reply_task.done()


@pytest.mark.asyncio
async def test_say_back_to_back_does_not_cancel_replacement() -> None:
    model = RealtimeModel(api_key="fake")
    sess = RealtimeSession(model)
    sess._socket = AsyncMock()

    assert not sess._ready_to_start.is_set()
    fut1 = sess.say("first")
    fut2 = sess.say("second")

    # First future was cancelled by the replacement
    assert fut1.cancelled()
    assert not fut2.done()

    # Readiness opens
    sess._ready_to_start.set()
    await asyncio.sleep(0.02)

    # Replacement text should be sent
    sess._socket.send_say.assert_called_once()
    payload = sess._socket.send_say.call_args[0][0]
    assert payload.text == "second"
    assert not fut2.cancelled()
