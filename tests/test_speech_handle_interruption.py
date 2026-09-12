from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents.voice.speech_handle import INTERRUPTION_TIMEOUT, SpeechHandle

pytestmark = pytest.mark.unit


async def test_wait_for_playout_unblocks_immediately_on_interruption() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)
    assert not handle.done()
    assert not handle.interrupted

    waiter_task = asyncio.create_task(handle.wait_for_playout())
    await asyncio.sleep(0.01)
    assert not waiter_task.done()

    start = time.perf_counter()
    handle.interrupt()

    # Must unblock immediately without waiting for INTERRUPTION_TIMEOUT (5.0s)
    await asyncio.wait_for(waiter_task, timeout=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert elapsed < INTERRUPTION_TIMEOUT
    assert handle.interrupted is True
    assert handle.done() is False
    assert handle.exception() is None

    # Cleanup timer handle
    handle._mark_done()


async def test_wait_for_playout_returns_immediately_if_already_interrupted() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)
    handle.interrupt()
    assert handle.interrupted is True
    assert handle.done() is False
    assert handle.exception() is None

    start = time.perf_counter()
    await asyncio.wait_for(handle.wait_for_playout(), timeout=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1
    handle._mark_done()


async def test_await_handle_unblocks_immediately_on_interruption() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)
    callback_called = False

    def _on_done(h: SpeechHandle) -> None:
        nonlocal callback_called
        callback_called = True

    handle.add_done_callback(_on_done)

    async def _await_handle() -> SpeechHandle:
        return await handle

    waiter_task = asyncio.create_task(_await_handle())
    await asyncio.sleep(0.01)
    assert not waiter_task.done()
    assert not callback_called

    start = time.perf_counter()
    handle.interrupt()

    result = await asyncio.wait_for(waiter_task, timeout=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert elapsed < INTERRUPTION_TIMEOUT
    assert result is handle
    assert handle.interrupted is True
    assert handle.done() is False
    assert handle.exception() is None
    assert callback_called is False  # callbacks are tied to actual _done_fut finalization

    handle._mark_done()
    await asyncio.sleep(0)
    assert handle.done() is True
    assert callback_called is True


async def test_wait_for_playout_unblocks_on_done() -> None:
    handle = SpeechHandle.create(allow_interruptions=True)
    waiter_task = asyncio.create_task(handle.wait_for_playout())
    await asyncio.sleep(0.01)
    assert not waiter_task.done()

    handle._mark_done()
    await asyncio.wait_for(waiter_task, timeout=1.0)
    assert handle.done() is True
    assert handle.interrupted is False
