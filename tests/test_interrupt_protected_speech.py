"""``AgentActivity.interrupt()`` and queued speeches that disallow interruptions.

``SpeechHandle.interrupt()`` raises for such a handle while it is running, and
the queue loop used to let that raise escape: the remaining queued speeches
were left playing and the returned future never resolved. Interrupting *past*
the protected speech would be wrong too — the ones behind it still play, so
skipping one in the middle would leave a gap in the conversation.
"""

import heapq
import logging
import time
from unittest.mock import Mock

import pytest

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_session import FakeActions, create_session
from .test_agent_session import MyAgent, _close_test_session

pytestmark = pytest.mark.unit


def _make_activity() -> AgentActivity:
    return AgentActivity(MyAgent(), create_session(FakeActions()))


def _enqueue(activity: AgentActivity, speech: SpeechHandle, *, priority: int = 0) -> None:
    """Queue a speech the way _schedule_speech does (a heap, not a list)."""
    heapq.heappush(activity._speech_q, (-priority, time.perf_counter_ns(), speech))


async def test_an_interrupted_protected_handle_is_left_alone() -> None:
    handle = SpeechHandle.create(allow_interruptions=False)
    handle.interrupt(force=True)

    assert handle.interrupt() is handle  # must not raise
    assert handle.interrupted
    handle._mark_done()


async def test_a_done_protected_handle_is_left_alone() -> None:
    handle = SpeechHandle.create(allow_interruptions=False)
    handle._mark_done()

    assert handle.interrupt() is handle  # must not raise
    assert not handle.interrupted


class TestInterruptQueuedSpeeches:
    async def test_queue_stops_at_the_protected_speech(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=True)
        first = SpeechHandle.create(allow_interruptions=True)
        protected = SpeechHandle.create(allow_interruptions=False)
        behind = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        for speech in (first, protected, behind):
            _enqueue(activity, speech)

        try:
            with caplog.at_level(logging.WARNING, logger="livekit.agents"):
                activity.interrupt()  # must not raise

            assert current.interrupted
            assert first.interrupted
            assert not protected.interrupted
            # no hole: what plays after the protected speech is untouched
            assert not behind.interrupted
            # the rest of the sequence still ran
            activity._rt_session.interrupt.assert_called_once()
            assert any("force=True" in record.message for record in caplog.records)
        finally:
            await _close_test_session(activity._session)

    async def test_a_protected_head_shields_the_whole_queue(self) -> None:
        # [protected, interruptible, protected]: interrupting only the middle
        # one would play the first and third with a gap between them
        activity = _make_activity()
        activity._rt_session = Mock()
        head = SpeechHandle.create(allow_interruptions=False)
        middle = SpeechHandle.create(allow_interruptions=True)
        tail = SpeechHandle.create(allow_interruptions=False)
        for speech in (head, middle, tail):
            _enqueue(activity, speech)

        try:
            activity.interrupt()

            assert not head.interrupted
            assert not middle.interrupted
            assert not tail.interrupted
        finally:
            await _close_test_session(activity._session)

    async def test_a_cancelled_protected_speech_does_not_shield_the_queue(self) -> None:
        # it is never going to play, so it cannot leave a gap behind it either
        activity = _make_activity()
        activity._rt_session = Mock()
        protected = SpeechHandle.create(allow_interruptions=False)
        behind = SpeechHandle.create(allow_interruptions=True)
        for speech in (protected, behind):
            _enqueue(activity, speech)
        # e.g. the caller cancelling the handle its own say() returned
        protected.interrupt(force=True)

        try:
            activity.interrupt()

            assert behind.interrupted
        finally:
            await _close_test_session(activity._session)

    async def test_the_queue_is_walked_in_playout_order_not_heap_order(self) -> None:
        # a higher-priority speech is queued last but plays first; the heap's
        # list order does not reflect that, the walk must
        activity = _make_activity()
        activity._rt_session = Mock()
        low = SpeechHandle.create(allow_interruptions=True)
        urgent_protected = SpeechHandle.create(allow_interruptions=False)
        _enqueue(activity, low, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
        _enqueue(activity, urgent_protected, priority=SpeechHandle.SPEECH_PRIORITY_HIGH)

        try:
            activity.interrupt()

            # the protected speech plays first, so the walk stops immediately
            assert not urgent_protected.interrupted
            assert not low.interrupted
        finally:
            await _close_test_session(activity._session)

    async def test_a_protected_playing_speech_still_raises(self) -> None:
        # unchanged behaviour: SpeechHandle.interrupt() is explicit about it
        activity = _make_activity()
        activity._current_speech = SpeechHandle.create(allow_interruptions=False)

        try:
            with pytest.raises(RuntimeError):
                activity.interrupt()
        finally:
            await _close_test_session(activity._session)

    async def test_force_interrupts_the_whole_chain(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=False)
        queued = SpeechHandle.create(allow_interruptions=False)
        behind = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        for speech in (queued, behind):
            _enqueue(activity, speech)

        try:
            activity.interrupt(force=True)

            assert current.interrupted
            assert queued.interrupted
            assert behind.interrupted
            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)

    async def test_interruptible_chain_is_unaffected(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=True)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        _enqueue(activity, queued)

        try:
            activity.interrupt()

            assert current.interrupted
            assert queued.interrupted
            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)
