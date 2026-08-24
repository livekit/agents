from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import _EndOfTurnInfo, _EndOfTurnMetrics

from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _livekit_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # the eager TurnDetector() default and the adaptive detector read these
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")


def _activity(*, server_segments: bool, server_replies: bool, **handling: object) -> AgentActivity:
    session = AgentSession(
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=server_segments,
                auto_turn_reply_generation=server_replies,
                can_disable_turn_detection=False,
            )
        ),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(**handling),  # type: ignore[typeddict-item]
    )
    activity = AgentActivity(Agent(instructions="test"), session)
    activity._rt_session = MagicMock()
    return activity


def _end_of_turn_info() -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=False,
        new_transcript="hello",
        transcript_confidence=0.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
        backchannel_over_agent=False,
    )


async def test_server_segmented_turn_is_not_committed_by_the_client() -> None:
    # server VAD with create_response=False segments and commits each turn itself, so a client
    # commit lands on an emptied buffer (input_audio_buffer_commit_empty). The client still owns
    # the reply.
    activity = _activity(server_segments=True, server_replies=False)
    assert activity._rt_turn_detection_enabled is True
    assert activity._rt_server_reply_enabled is False

    await activity._user_turn_completed_task(None, _end_of_turn_info())

    activity._rt_session.commit_audio.assert_not_called()


async def test_client_segmented_turn_is_committed() -> None:
    # nothing segments server-side: the client owns the input buffer and must commit it
    activity = _activity(server_segments=False, server_replies=False)
    assert activity._rt_turn_detection_enabled is False

    await activity._user_turn_completed_task(None, _end_of_turn_info())

    activity._rt_session.commit_audio.assert_called_once()


async def test_server_answered_turn_is_left_to_the_server() -> None:
    # the server both segments and answers, so the client neither commits nor replies
    activity = _activity(server_segments=True, server_replies=True)
    assert activity._rt_server_reply_enabled is True

    await activity._user_turn_completed_task(None, _end_of_turn_info())

    activity._rt_session.commit_audio.assert_not_called()


def test_commit_user_turn_skips_the_commit_when_the_server_segments() -> None:
    # the manual turn API has the same buffer owner as the automatic path
    activity = _activity(server_segments=True, server_replies=False)
    activity._audio_recognition = MagicMock()

    activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.0, skip_reply=True)

    activity._rt_session.commit_audio.assert_not_called()


def test_commit_user_turn_commits_when_the_client_segments() -> None:
    activity = _activity(server_segments=False, server_replies=False)
    activity._audio_recognition = MagicMock()

    activity.commit_user_turn(transcript_timeout=1.0, stt_flush_duration=0.0, skip_reply=True)

    activity._rt_session.commit_audio.assert_called_once()


def test_barge_in_is_disabled_when_the_server_segments() -> None:
    # adaptive barge-in gatekeeps by withholding the commit, which is impossible once the
    # server commits every segment on its own
    activity = _activity(
        server_segments=True, server_replies=False, interruption={"mode": "adaptive"}
    )

    assert activity._interruption_detector is None
    assert activity._interruption_detection_enabled is False
