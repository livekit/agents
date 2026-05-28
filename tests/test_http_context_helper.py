"""Tests for the public `utils.http_context.open()` helper and for the
inference STT error surface when called outside a job context.
"""

from __future__ import annotations

import asyncio

import aiohttp
import pytest

from livekit.agents import inference
from livekit.agents.utils import http_context


async def test_open_yields_working_session_and_closes_on_exit() -> None:
    with pytest.raises(RuntimeError):
        http_context.http_session()

    async with http_context.open() as session:
        assert isinstance(session, aiohttp.ClientSession)
        assert not session.closed
        # http_session() returns the same instance inside the block
        assert http_context.http_session() is session

    assert session.closed
    with pytest.raises(RuntimeError):
        http_context.http_session()


async def test_open_is_reentrant_inner_does_not_close_outer() -> None:
    async with http_context.open() as outer:
        async with http_context.open() as inner:
            # nested open() reuses the outer session — does not create a new one
            assert inner is outer

        # outer session is untouched after inner exits
        assert not outer.closed
        assert http_context.http_session() is outer

    assert outer.closed


async def test_open_isolated_per_task() -> None:
    """Each asyncio.Task gets its own http session context — they don't share."""
    barrier = asyncio.Event()

    async def worker() -> tuple[aiohttp.ClientSession, bool]:
        async with http_context.open() as session:
            await barrier.wait()
            still_open = not session.closed
        return session, still_open

    task_a = asyncio.create_task(worker())
    task_b = asyncio.create_task(worker())
    await asyncio.sleep(0.01)
    barrier.set()

    (sess_a, a_open), (sess_b, b_open) = await asyncio.gather(task_a, task_b)
    assert sess_a is not sess_b
    assert a_open and b_open
    assert sess_a.closed and sess_b.closed


async def test_http_session_error_message_points_to_helper() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        http_context.http_session()
    msg = str(exc_info.value)
    assert "http_context.open()" in msg


async def test_inference_stt_surfaces_real_error_outside_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: previously, calling `inference.STT().stream()` outside a job
    context raised `AttributeError: 'SpeechStream' object has no attribute
    '_session'` from a background task — masking the real "no http context" error.

    After the fix, _ensure_session() runs inside _run(), so the actual
    RuntimeError surfaces through the stream's main task.
    """
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")

    stt = inference.STT(model="cartesia/sonic-3")
    stream = stt.stream()

    # SpeechStream no longer eagerly grabs `_session` in __init__.
    assert not hasattr(stream, "_session")

    with pytest.raises(RuntimeError, match="http_context.open"):
        await stream._task

    await stream.aclose()
