"""Regression tests for SpeechHandle / RunContext playout-wait interruption.

Guards against the deadlock described in livekit/agents#5359, where
``SpeechHandle.wait_for_playout()`` and ``RunContext.wait_for_playout()``
awaited only on the playout-completion future and ignored the interrupt
future. When ``interrupt()`` fired, callers blocked until the
``INTERRUPTION_TIMEOUT`` (~5s) hard-killed the surrounding tasks.

Both methods now race the playout future against ``_interrupt_fut`` using
``asyncio.wait(FIRST_COMPLETED)`` (the same primitive already used by
``SpeechHandle.wait_if_not_interrupted``). The wait returns promptly on
interrupt; callers can inspect ``speech_handle.interrupted`` to decide how
to proceed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from livekit.agents.llm import FunctionCall
from livekit.agents.voice.events import RunContext
from livekit.agents.voice.speech_handle import SpeechHandle


async def test_speech_handle_wait_for_playout_returns_on_interrupt() -> None:
    """wait_for_playout must unblock when interrupted, not hang on _done_fut.

    Regression test for https://github.com/livekit/agents/issues/5359.
    Pre-fix this hung on ``_done_fut`` for INTERRUPTION_TIMEOUT (~5s); the
    1.0s deadline below would fire and the test would raise TimeoutError.
    """
    sh = SpeechHandle.create()

    async def _interrupt_after_delay() -> None:
        await asyncio.sleep(0.05)
        sh._cancel()

    interrupt_task = asyncio.create_task(_interrupt_after_delay())
    try:
        await asyncio.wait_for(sh.wait_for_playout(), timeout=1.0)
    finally:
        await interrupt_task

    assert sh.interrupted
    assert not sh.done()


async def test_run_context_wait_for_playout_returns_on_interrupt() -> None:
    """RunContext.wait_for_playout must unblock when the speech is interrupted.

    Regression test for https://github.com/livekit/agents/issues/5359.
    """
    sh = SpeechHandle.create()
    sh._authorize_generation()

    fc = FunctionCall(call_id="call_test", arguments="{}", name="noop")
    ctx = RunContext(session=MagicMock(), speech_handle=sh, function_call=fc)

    async def _interrupt_after_delay() -> None:
        await asyncio.sleep(0.05)
        sh._cancel()

    interrupt_task = asyncio.create_task(_interrupt_after_delay())
    try:
        await asyncio.wait_for(ctx.wait_for_playout(), timeout=1.0)
    finally:
        await interrupt_task
        sh._mark_done()

    assert sh.interrupted


async def test_speech_handle_wait_for_playout_returns_normally_on_completion() -> None:
    """Sanity check: a non-interrupted playout still resolves on _done_fut."""
    sh = SpeechHandle.create()

    async def _complete_after_delay() -> None:
        await asyncio.sleep(0.05)
        sh._mark_done()

    completion_task = asyncio.create_task(_complete_after_delay())
    try:
        await asyncio.wait_for(sh.wait_for_playout(), timeout=1.0)
    finally:
        await completion_task

    assert not sh.interrupted
    assert sh.done()
