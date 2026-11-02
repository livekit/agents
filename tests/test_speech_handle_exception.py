"""
Tests for SpeechHandle error reporting.

When a generation fails (e.g. a realtime ``generate_reply`` timeout, see
https://github.com/livekit/agents/issues/6224), the error is recorded on the
SpeechHandle instead of being set on its done future: awaiting a handle never
raises (most handles are never awaited, so a stored exception would trigger
"Future exception was never retrieved" warnings). Users inspect failures with
``SpeechHandle.exception()`` after the handle is done, and ``session.run()``
still raises through RunResult.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import RealtimeError
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


async def test_await_does_not_raise_on_error() -> None:
    handle = SpeechHandle.create()
    handle._mark_done(error=RealtimeError("generate_reply timed out."))

    result = await handle
    assert result is handle

    await handle.wait_for_playout()

    exc = handle.exception()
    assert isinstance(exc, RealtimeError)
    assert str(exc) == "generate_reply timed out."


async def test_exception_is_none_without_error() -> None:
    handle = SpeechHandle.create()
    handle._mark_done()

    await handle
    assert handle.exception() is None


async def test_exception_raises_if_not_done() -> None:
    handle = SpeechHandle.create()

    with pytest.raises(asyncio.InvalidStateError):
        handle.exception()


async def test_error_ignored_after_done() -> None:
    handle = SpeechHandle.create()
    handle._mark_done(error=RealtimeError("first"))
    # e.g. the task-level done callback marking the handle again
    handle._mark_done()
    handle._mark_done(error=RealtimeError("second"))

    exc = handle.exception()
    assert isinstance(exc, RealtimeError)
    assert str(exc) == "first"


async def test_run_result_propagates_speech_handle_error() -> None:
    run_result = RunResult[None](output_type=None)
    handle = SpeechHandle.create()
    run_result._watch_handle(handle)

    handle._mark_done(error=RealtimeError("generate_reply timed out."))

    with pytest.raises(RealtimeError, match="generate_reply timed out"):
        await run_result
