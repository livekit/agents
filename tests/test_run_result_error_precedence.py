"""Regression tests for error precedence in ``RunResult``.

``session.run()`` watches every speech handle created while it is active, e.g. a
``session.say()`` fired from a function tool. The run used to take the output and
error from the last handle that completed, so an earlier watched speech that
failed was hidden when a later handle completed cleanly.
"""

from __future__ import annotations

import asyncio

import pytest

from livekit.agents.llm import RealtimeError
from livekit.agents.voice.run_result import RunResult
from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = pytest.mark.unit


async def _settle_done_callbacks() -> None:
    # SpeechHandle is a future: its done callbacks are scheduled on the loop.
    await asyncio.sleep(0)


async def test_earlier_handle_error_is_not_swallowed() -> None:
    """The failing handle completes first; a later clean handle without output
    used to let the run resolve successfully and hide the failure."""
    run = RunResult[None](output_type=None)
    failing = SpeechHandle.create()
    clean = SpeechHandle.create()

    run._watch_handle(failing)
    run._watch_handle(clean)

    failing._mark_done(error=RealtimeError("provider exploded"))
    await _settle_done_callbacks()
    clean._mark_done()
    await _settle_done_callbacks()

    with pytest.raises(RealtimeError, match="provider exploded"):
        await asyncio.wait_for(run, timeout=1.0)


async def test_failure_with_queued_done_callback_is_still_seen() -> None:
    """A failure whose done callback is still queued on the loop must not be
    missed when the run checks completion synchronously (as AgentTask does)."""
    run = RunResult[None](output_type=None)
    failing = SpeechHandle.create()
    clean = SpeechHandle.create()
    run._watch_handle(failing)
    run._watch_handle(clean)

    failing._mark_done(error=RealtimeError("queued failure"))
    clean._mark_done()
    # neither done callback has run yet
    run._mark_done_if_needed(None)

    with pytest.raises(RealtimeError, match="queued failure"):
        await asyncio.wait_for(run, timeout=1.0)


async def test_earlier_handle_error_is_not_hidden_by_later_output() -> None:
    """A later handle that carries its own output must not replace the failure
    of a previously watched speech."""
    run = RunResult[str](output_type=str)
    failing = SpeechHandle.create()
    with_output = SpeechHandle.create()
    with_output._maybe_run_final_output = "task result"
    # AgentTask.complete() records the value on the handle and the run state
    run._set_final_output("task result")

    run._watch_handle(failing)
    run._watch_handle(with_output)

    failing._mark_done(error=RealtimeError("provider exploded"))
    await _settle_done_callbacks()
    with_output._mark_done()
    await _settle_done_callbacks()

    with pytest.raises(RealtimeError, match="provider exploded"):
        await asyncio.wait_for(run, timeout=1.0)


async def test_clean_handles_still_complete_normally() -> None:
    """Without a failure, the last completed handle keeps deciding the result."""
    run = RunResult[str](output_type=str)
    first = SpeechHandle.create()
    second = SpeechHandle.create()
    # AgentTask.complete() records the value on the handle and the run state
    second._maybe_run_final_output = "task result"
    run._set_final_output("task result")

    run._watch_handle(first)
    run._watch_handle(second)

    first._mark_done()
    await _settle_done_callbacks()
    second._mark_done()
    await _settle_done_callbacks()

    await asyncio.wait_for(run, timeout=1.0)
    assert run.final_output == "task result"


async def test_error_on_the_last_handle_still_propagates() -> None:
    """The existing behavior for a failure on the last handle is unchanged."""
    run = RunResult[None](output_type=None)
    clean = SpeechHandle.create()
    failing = SpeechHandle.create()

    run._watch_handle(clean)
    run._watch_handle(failing)

    clean._mark_done()
    await _settle_done_callbacks()
    failing._mark_done(error=RealtimeError("provider exploded"))
    await _settle_done_callbacks()

    with pytest.raises(RealtimeError, match="provider exploded"):
        await asyncio.wait_for(run, timeout=1.0)
