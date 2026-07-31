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
