"""``AgentActivity.interrupt()`` must not half-apply an interruption.

``SpeechHandle.interrupt()`` raises when the handle disallows interruptions.
That raise used to escape from the middle of ``interrupt()``: the preemptive
generation was already cancelled and the background speeches already
interrupted, while the queue kept playing, the realtime session was never told
and the returned future never resolved. Background speeches were filtered but
queued ones were not, so a queued ``say(allow_interruptions=False)`` aborted the
sequence too.
"""

from unittest.mock import Mock

import pytest

from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_session import FakeActions, create_session
from .test_agent_session import MyAgent, _close_test_session

pytestmark = pytest.mark.unit


def _make_activity() -> AgentActivity:
    return AgentActivity(MyAgent(), create_session(FakeActions()))


class TestInterruptProtectedSpeech:
    async def test_protected_current_speech_leaves_everything_untouched(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        background = SpeechHandle.create(allow_interruptions=True)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = SpeechHandle.create(allow_interruptions=False)
        activity._background_speeches.add(background)
        activity._speech_q.append((0, 0.0, queued))

        try:
            with pytest.raises(RuntimeError, match="force=True"):
                activity.interrupt()

            assert not background.interrupted
            assert not queued.interrupted
            activity._rt_session.interrupt.assert_not_called()
        finally:
            await _close_test_session(activity._session)

    async def test_protected_queued_speech_is_skipped(self) -> None:
        activity = _make_activity()
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=True)
        protected = SpeechHandle.create(allow_interruptions=False)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        activity._speech_q.append((0, 0.0, protected))
        activity._speech_q.append((0, 1.0, queued))

        try:
            activity.interrupt()  # must not raise

            assert current.interrupted
            assert queued.interrupted
            assert not protected.interrupted
            # the rest of the sequence still runs
            activity._rt_session.interrupt.assert_called_once()
        finally:
            await _close_test_session(activity._session)

    async def test_force_interrupts_protected_speeches(self) -> None:
        activity = _make_activity()
        current = SpeechHandle.create(allow_interruptions=False)
        queued = SpeechHandle.create(allow_interruptions=False)
        activity._current_speech = current
        activity._speech_q.append((0, 0.0, queued))

        try:
            activity.interrupt(force=True)

            assert current.interrupted
            assert queued.interrupted
        finally:
            await _close_test_session(activity._session)

    async def test_interruptible_speeches_are_unaffected(self) -> None:
        activity = _make_activity()
        current = SpeechHandle.create(allow_interruptions=True)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        activity._speech_q.append((0, 0.0, queued))

        try:
            activity.interrupt()

            assert current.interrupted
            assert queued.interrupted
        finally:
            await _close_test_session(activity._session)
