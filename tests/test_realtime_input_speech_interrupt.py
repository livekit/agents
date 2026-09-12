from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.speech_handle import SpeechHandle

from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _livekit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # the eager TurnDetector() default reads these; keep construction hermetic
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")


def _activity(*, server_turn_detection: bool, allow_interruptions: bool = True) -> AgentActivity:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=server_turn_detection, can_disable_turn_detection=False
            )
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"enabled": allow_interruptions}),
    )
    return AgentActivity(Agent(instructions="test"), session)


def _speech_started(activity: AgentActivity, *, allow_interruptions: bool) -> SpeechHandle:
    handle = SpeechHandle.create(allow_interruptions=allow_interruptions)
    activity._current_speech = handle
    activity._rt_session = MagicMock()
    # a bare MagicMock reports every capability as true; report the model's own instead
    activity._rt_session.capabilities = activity.llm.capabilities
    activity._on_input_speech_started(llm.InputSpeechStartedEvent())
    return handle


def test_allow_interruptions_false_rejected_with_server_turn_detection() -> None:
    # the server cancels its own response on user speech, so the local speech can't opt out
    with pytest.raises(ValueError, match="allow_interruptions cannot be False"):
        _activity(server_turn_detection=True, allow_interruptions=False)


def test_allow_interruptions_false_allowed_with_client_turn_taking() -> None:
    # server VAD with create_response=False reports no server-side turn detection, so a
    # turn-taking agent can disable interruptions (issue #6635)
    activity = _activity(server_turn_detection=False, allow_interruptions=False)

    assert activity._rt_turn_detection_enabled is False


def test_input_speech_started_keeps_uninterruptible_speech(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the model still reports user speech (server VAD is on to commit and transcribe audio), but
    # neither side interrupts: no response.cancel, no local interruption, and no error log since
    # this is a valid configuration
    activity = _activity(server_turn_detection=False, allow_interruptions=False)

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        handle = _speech_started(activity, allow_interruptions=False)

    assert handle.interrupted is False
    activity._rt_session.interrupt.assert_not_called()
    assert not caplog.records


def test_input_speech_started_interrupts_interruptible_speech() -> None:
    activity = _activity(server_turn_detection=True)

    handle = _speech_started(activity, allow_interruptions=True)

    assert handle.interrupted is True
    activity._rt_session.interrupt.assert_called_once()


def test_input_speech_started_keeps_a_held_speech(caplog: pytest.LogCaptureFixture) -> None:
    # the point of a hold: server-side turn detection stays on, the model still reports user
    # speech, and the speech someone deliberately held keeps playing through it. No
    # response.cancel is sent, because AgentActivity.interrupt() raises before reaching it.
    activity = _activity(server_turn_detection=True)
    handle = SpeechHandle.create(allow_interruptions=True)
    activity._current_speech = handle
    activity._rt_session = MagicMock()

    with handle.hold_interruptions():
        with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
            activity._on_input_speech_started(llm.InputSpeechStartedEvent())

        assert handle.interrupted is False
        activity._rt_session.interrupt.assert_not_called()

    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert any("held" in record.message for record in caplog.records)
    handle._mark_done()


def test_input_speech_started_still_reports_an_unheld_uninterruptible_speech(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # the loud log is narrowed, not removed. A speech that refuses interruption *without*
    # anyone holding it is the desync the log was added for -- the server cancelled its own
    # response while this speech still believed it could not be interrupted.
    activity = _activity(server_turn_detection=True)

    with caplog.at_level(logging.ERROR, logger="livekit.agents"):
        handle = _speech_started(activity, allow_interruptions=False)

    assert handle.interrupted is False
    assert [record for record in caplog.records if record.levelno >= logging.ERROR], (
        "an uninterruptible speech nobody held is still reported loudly"
    )
    handle._mark_done()
