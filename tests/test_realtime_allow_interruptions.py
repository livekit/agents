"""Regression tests for #6635: allow_interruptions=False with server-side turn detection.

The activity used to raise at construction when a RealtimeModel with
server-side turn detection was combined with ``allow_interruptions=False``,
and unconditionally sent ``response.cancel`` (via ``interrupt()``) on every
``input_speech_started`` event — cutting the agent off on any overlap even in
manual/button-style turn-taking setups where the provider itself is configured
with ``interrupt_response=False``.
"""

import logging
from unittest.mock import Mock

import pytest

from livekit.agents import AgentSession
from livekit.agents.llm import InputSpeechStartedEvent
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .test_agent_session import MyAgent

pytestmark = pytest.mark.unit


def _make_session(*, allow_interruptions: bool) -> AgentSession:
    return AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=True)),
        allow_interruptions=allow_interruptions,
    )


class TestAllowInterruptionsFalse:
    async def test_construction_no_longer_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        session = _make_session(allow_interruptions=False)
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            activity = AgentActivity(MyAgent(), session)
        assert activity.allow_interruptions is False
        # the provider-consistency warning is emitted instead of a ValueError
        assert any("interrupt_response" in r.message for r in caplog.records)

    async def test_construction_with_interruptions_stays_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        session = _make_session(allow_interruptions=True)
        with caplog.at_level(logging.WARNING, logger="livekit.agents"):
            AgentActivity(MyAgent(), session)
        assert not any("interrupt_response" in r.message for r in caplog.records)

    async def test_input_speech_started_does_not_interrupt(self) -> None:
        session = _make_session(allow_interruptions=False)
        activity = AgentActivity(MyAgent(), session)
        activity.interrupt = Mock()  # type: ignore[method-assign]

        activity._on_input_speech_started(InputSpeechStartedEvent())

        activity.interrupt.assert_not_called()

    async def test_input_speech_started_interrupts_when_allowed(self) -> None:
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity.interrupt = Mock()  # type: ignore[method-assign]

        activity._on_input_speech_started(InputSpeechStartedEvent())

        activity.interrupt.assert_called_once()

    async def test_protected_speech_wins_over_permissive_default(self) -> None:
        # say(..., allow_interruptions=False) while the session default is True:
        # the playing speech must not be interrupted (and no spurious
        # "this should never happen" RuntimeError must be logged)
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity._current_speech = SpeechHandle.create(allow_interruptions=False)
        activity.interrupt = Mock()  # type: ignore[method-assign]

        activity._on_input_speech_started(InputSpeechStartedEvent())

        activity.interrupt.assert_not_called()

    async def test_interruptible_speech_wins_over_protective_default(self) -> None:
        # say(..., allow_interruptions=True) while the session default is False:
        # the explicitly interruptible speech must be interrupted locally
        session = _make_session(allow_interruptions=False)
        activity = AgentActivity(MyAgent(), session)
        activity._current_speech = SpeechHandle.create(allow_interruptions=True)
        activity.interrupt = Mock()  # type: ignore[method-assign]

        activity._on_input_speech_started(InputSpeechStartedEvent())

        activity.interrupt.assert_called_once()

    async def test_interrupt_skips_protected_queued_speech(self) -> None:
        # a protected handle can now sit in the queue while an interruptible
        # one plays; interrupt() must skip it (like background speeches)
        # instead of raising mid-loop and aborting the rest of the sequence
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        current = SpeechHandle.create(allow_interruptions=True)
        protected = SpeechHandle.create(allow_interruptions=False)
        queued = SpeechHandle.create(allow_interruptions=True)
        activity._current_speech = current
        activity._speech_q.append((0, 0.0, protected))
        activity._speech_q.append((0, 1.0, queued))

        activity.interrupt()  # must not raise

        assert current.interrupted
        assert queued.interrupted
        assert not protected.interrupted

    async def test_interrupt_preserves_the_protected_realtime_generation(self) -> None:
        # the protected handle owns the response the provider is streaming:
        # response.cancel would truncate exactly the speech we preserved
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity._rt_session = Mock()
        protected = SpeechHandle.create(allow_interruptions=False)
        activity._rt_generation_handle = protected
        activity._current_speech = SpeechHandle.create(allow_interruptions=True)
        activity._speech_q.append((0, 0.0, protected))

        activity.interrupt()

        activity._rt_session.interrupt.assert_not_called()

    async def test_interrupt_cancels_when_the_generation_owner_is_interrupted(self) -> None:
        # the usual flow: the streaming response belongs to the playing speech
        # while a protected say() merely waits in the queue. Stopping playback
        # locally must still stop the provider, or it keeps generating a reply
        # nobody will hear and its state diverges from the local one
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity._rt_session = Mock()
        current = SpeechHandle.create(allow_interruptions=True)
        activity._rt_generation_handle = current
        activity._current_speech = current
        activity._speech_q.append((0, 0.0, SpeechHandle.create(allow_interruptions=False)))

        activity.interrupt()

        activity._rt_session.interrupt.assert_called_once()

    async def test_interrupt_cancels_when_the_protected_generation_is_done(self) -> None:
        # a protected handle lingers in the queue until the scheduling task
        # pops it; once done it owns nothing and must not suppress the cancel
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity._rt_session = Mock()
        protected = SpeechHandle.create(allow_interruptions=False)
        protected._mark_done()
        activity._rt_generation_handle = protected
        activity._speech_q.append((0, 0.0, protected))

        activity.interrupt()

        activity._rt_session.interrupt.assert_called_once()

    async def test_forced_interrupt_still_cancels_the_realtime_generation(self) -> None:
        session = _make_session(allow_interruptions=True)
        activity = AgentActivity(MyAgent(), session)
        activity._rt_session = Mock()
        protected = SpeechHandle.create(allow_interruptions=False)
        activity._rt_generation_handle = protected
        activity._speech_q.append((0, 0.0, protected))

        activity.interrupt(force=True)

        assert protected.interrupted
        activity._rt_session.interrupt.assert_called_once()
