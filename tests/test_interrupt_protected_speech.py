"""``AgentActivity.interrupt()`` walks the playout order and stops at the first
speech that disallows interruptions.

``SpeechHandle.interrupt()`` raises for such a handle, and that raise used to
escape from the middle of ``interrupt()``: the queued speeches were left
playing, the realtime session was never told to stop generating, and the
returned future never resolved. Interrupting *past* a protected speech is
wrong too — the speeches behind it still play, so skipping one in the middle
would leave a gap in the conversation.
"""

import heapq
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


class TestInterruptWalk:
    async def test_protected_current_speech_stops_the_walk(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        activity._preemptive_generation = Mock()
        background = SpeechHandle.create(allow_interruptions=True)
        queued = SpeechHandle.create(allow_interruptions=True)
        current = SpeechHandle.create(allow_interruptions=False)
        activity._current_speech = current
        activity._background_speeches.add(background)
        _enqueue(activity, queued)

        try:
            activity.interrupt()  # must not raise

            # the protected speech keeps playing, and so does everything queued
            # behind it
            assert not current.interrupted
            assert not queued.interrupted
            # the streaming realtime response belongs to the speech we kept
            activity._rt_session.interrupt.assert_not_called()
            # neither of these depends on the playing speech
            assert background.interrupted
            assert activity._preemptive_generation is None
        finally:
            await _close_test_session(activity._session)

    async def test_walk_stops_at_the_first_protected_queued_speech(self) -> None:
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
            activity.interrupt()

            assert current.interrupted
            assert first.interrupted
            assert not protected.interrupted
            # no hole: what plays after the protected speech is untouched
            assert not behind.interrupted
            # the playing speech was interrupted, so the server must stop too
            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)

    async def test_a_protected_head_is_not_skipped_over(self) -> None:
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

    async def test_the_walk_follows_playout_order_not_heap_order(self) -> None:
        # a higher-priority speech is queued last but plays first; the heap's
        # list order does not reflect that, the walk must
        activity = _make_activity()
        activity._rt_session = Mock()
        low = SpeechHandle.create(allow_interruptions=True)
        urgent_protected = SpeechHandle.create(allow_interruptions=False)
        _enqueue(activity, low, priority=SpeechHandle.SPEECH_PRIORITY_NORMAL)
        _enqueue(activity, urgent_protected, priority=SpeechHandle.SPEECH_PRIORITY_HIGH)

        try:
            assert activity._playout_ordered_speeches() == [urgent_protected, low]

            activity.interrupt()

            # the protected speech plays first, so the walk stops immediately
            assert not urgent_protected.interrupted
            assert not low.interrupted
        finally:
            await _close_test_session(activity._session)

    async def test_realtime_is_cancelled_when_nothing_is_playing(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()

        try:
            activity.interrupt()

            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)

    async def test_force_interrupts_the_whole_chain(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=False)
        queued = SpeechHandle.create(allow_interruptions=False)
        activity._current_speech = current
        _enqueue(activity, queued)

        try:
            activity.interrupt(force=True)

            assert current.interrupted
            assert queued.interrupted
            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)

    async def test_interruptible_chain_is_unaffected(self) -> None:
        activity = _make_activity()
        current = SpeechHandle.create(allow_interruptions=True)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        _enqueue(activity, queued)

        try:
            activity.interrupt()

            assert current.interrupted
            assert queued.interrupted
        finally:
            await _close_test_session(activity._session)
