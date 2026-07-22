from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
)

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


def _make_activity(session: AgentSession) -> AgentActivity:
    return AgentActivity(Agent(instructions="test"), session)


def _end_of_turn_info(
    transcript: str = "", *, backchannel_over_agent: bool = False
) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=False,
        new_transcript=transcript,
        transcript_confidence=0.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
        backchannel_over_agent=backchannel_over_agent,
    )


def _realtime_barge_in_session() -> AgentSession:
    return AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )


async def test_adaptive_interruption_enabled_for_realtime_without_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a realtime model with server-side turn detection off transcribes internally and
    # commits turns manually, so barge-in gatekeeps by withholding commit rather than
    # holding STT transcripts — no separate STT is required
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())

    assert activity._interruption_detection_enabled is True
    assert activity._interruption_detector is not None


async def test_adaptive_interruption_still_requires_stt_for_non_realtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the STT pipeline path is unchanged: without an aligned streaming STT there is
    # nothing to gatekeep, so adaptive interruption stays disabled
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeLLM(fake_responses=[]),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )

    activity = _make_activity(session)

    assert activity._interruption_detection_enabled is False
    assert activity._interruption_detector is None


async def test_adaptive_interruption_disabled_for_realtime_with_server_turn_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # server-side turn detection creates turns automatically, so client-side barge-in
    # cannot take over — it stays disabled even for a realtime model
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=True)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(interruption={"mode": "adaptive"}),
    )

    activity = _make_activity(session)

    assert activity._interruption_detection_enabled is False
    assert activity._interruption_detector is None


async def test_backchannel_does_not_commit_while_agent_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the turn overlapped agent speech and no interruption was flagged, so it's a
    # backchannel and must not commit a user turn
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())
    activity._scheduling_paused = False  # simulate a running session
    assert activity._interruption_detection_enabled is True

    current_speech = MagicMock()
    current_speech.done.return_value = False
    current_speech.interrupted = False
    activity._current_speech = current_speech
    activity._interruption_detected = False

    # verdict pending, agent still speaking — dropped via the live-speech branch
    assert activity.on_end_of_turn(_end_of_turn_info(backchannel_over_agent=False)) is False


async def test_backchannel_dropped_after_agent_finishes_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the user finished the backchannel first (verdict latched), then the agent finished —
    # the backchannel verdict survives to end of turn, so the turn is still dropped with no
    # live speech to key off
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())
    activity._scheduling_paused = False  # simulate a running session

    activity._current_speech = None  # agent has finished speaking
    activity._interruption_detected = False

    # backchannel verdict for this turn survives the agent stopping
    assert activity.on_end_of_turn(_end_of_turn_info(backchannel_over_agent=True)) is False


def _recognition_for_overlap() -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._backchannel_boundary_timer = None
    ar._overlap_in_current_turn = True
    ar._turn_backchannel_over_agent = False
    ar._hooks = MagicMock()
    return ar


def _overlap_event(*, is_interruption: bool, agent_ended: bool) -> OverlappingSpeechEvent:
    return OverlappingSpeechEvent(is_interruption=is_interruption, agent_ended=agent_ended)


async def test_user_ended_overlap_latches_backchannel() -> None:
    # the user's overlap ended on its own with no interruption flagged — a real backchannel
    ar = _recognition_for_overlap()
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    assert ar._turn_backchannel_over_agent is True


async def test_agent_ended_overlap_is_not_a_backchannel() -> None:
    # the overlap ended because the agent finished, not the user — the user may still be
    # mid-turn, so this inconclusive verdict must not mark the turn a backchannel
    ar = _recognition_for_overlap()
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=True))
    assert ar._turn_backchannel_over_agent is False


async def test_agent_ended_overlap_preserves_prior_backchannel() -> None:
    # a real backchannel was already latched this turn; the later agent-ended overlap is a
    # no-op and must not clear it
    ar = _recognition_for_overlap()
    ar._turn_backchannel_over_agent = True
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=True))
    assert ar._turn_backchannel_over_agent is True


async def test_interruption_clears_backchannel() -> None:
    # a confirmed interruption supersedes any prior backchannel verdict for the turn
    ar = _recognition_for_overlap()
    ar._turn_backchannel_over_agent = True
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=True, agent_ended=False))
    assert ar._turn_backchannel_over_agent is False
    ar._hooks.on_interruption.assert_called_once()
