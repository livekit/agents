from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from livekit.agents import NOT_GIVEN, Agent, AgentSession, TurnHandlingOptions
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
)

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT
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


async def test_unjudged_overlap_over_a_paused_speech_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a pause silences the agent and ends the overlap, so no verdict is coming: the turn must
    # commit rather than be discarded on the assumption that one might still arrive
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())
    activity._scheduling_paused = False  # simulate a running session
    assert activity._interruption_detection_enabled is True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]

    current_speech = MagicMock()
    current_speech.done.return_value = False
    current_speech.interrupted = False
    activity._current_speech = current_speech
    activity._interruption_detected = False
    activity._update_paused_speech(current_speech, timeout=2.0)

    assert activity.on_end_of_turn(_end_of_turn_info(backchannel_over_agent=False)) is True


async def test_confirmed_backchannel_drops_while_agent_speaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the detector's verdict is the only thing that suppresses a turn
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())
    activity._scheduling_paused = False

    current_speech = MagicMock()
    current_speech.done.return_value = False
    current_speech.interrupted = False
    activity._current_speech = current_speech

    assert activity.on_end_of_turn(_end_of_turn_info(backchannel_over_agent=True)) is False


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


async def test_backchannel_confirmed_clears_rt_audio_even_with_stt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a realtime session buffers input even with an STT attached, so the clear isn't stt-gated
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False)),
        stt=FakeSTT(),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )
    activity = _make_activity(session)
    assert activity.stt is not None
    activity._rt_session = MagicMock()

    activity.on_backchannel_confirmed()

    activity._rt_session.clear_audio.assert_called_once()


async def test_backchannel_confirmed_noop_when_barge_in_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a late event after barge-in was disabled must not clear the buffer
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    activity = _make_activity(_realtime_barge_in_session())
    activity._interruption_detection_enabled = False
    activity._rt_session = MagicMock()

    activity.on_backchannel_confirmed()

    activity._rt_session.clear_audio.assert_not_called()


def _recognition_for_overlap(*, speaking: bool = False) -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._backchannel_boundary_timer = None
    ar._overlap_in_current_turn = True
    ar._turn_backchannel_over_agent = False
    ar._user_silence_ev = asyncio.Event()
    if not speaking:
        ar._user_silence_ev.set()  # silent (between segments) unless told otherwise
    ar._hooks = MagicMock()
    return ar


def _swallow_task(coro: object, **kwargs: object) -> MagicMock:
    """Stand in for _create_speech_task: the reply pipeline isn't under test here."""
    coro.close()  # type: ignore[attr-defined]
    return MagicMock()


def _overlap_event(*, is_interruption: bool, agent_ended: bool) -> OverlappingSpeechEvent:
    return OverlappingSpeechEvent(is_interruption=is_interruption, agent_ended=agent_ended)


async def test_user_ended_overlap_latches_backchannel() -> None:
    # the user's overlap ended on its own with no interruption flagged — a real backchannel
    ar = _recognition_for_overlap()
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    assert ar._turn_backchannel_over_agent is True


async def test_confirmed_backchannel_between_segments_clears_audio() -> None:
    # confirmed between segments (user silent) — cleared so it can't prefix the next turn
    ar = _recognition_for_overlap(speaking=False)
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    ar._hooks.on_backchannel_confirmed.assert_called_once()


async def test_confirmed_backchannel_while_speaking_defers_clear() -> None:
    # user already mid next-segment — latch the verdict but defer the clear (else we'd clip it)
    ar = _recognition_for_overlap(speaking=True)
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    assert ar._turn_backchannel_over_agent is True
    ar._hooks.on_backchannel_confirmed.assert_not_called()


async def test_agent_ended_overlap_is_not_a_backchannel() -> None:
    # the overlap ended because the agent finished, not the user — the user may still be
    # mid-turn, so this inconclusive verdict must not mark the turn a backchannel
    ar = _recognition_for_overlap()
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=True))
    assert ar._turn_backchannel_over_agent is False
    ar._hooks.on_backchannel_confirmed.assert_not_called()


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
    ar._hooks.on_backchannel_confirmed.assert_not_called()


class _RecordingChan:
    """Stands in for the interruption channel, capturing the sentinels sent to it."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    def send_nowait(self, item: object) -> None:
        self.sent.append(item)


def _recognition_with_interruption_ch() -> tuple[AudioRecognition, _RecordingChan]:
    ar = AudioRecognition.__new__(AudioRecognition)
    ch = _RecordingChan()
    ar._interruption_enabled = True
    ar._interruption_ch = ch  # type: ignore[assignment]
    ar._agent_speaking = False
    ar._agent_speech_started_at = None
    ar._endpointing = MagicMock()
    ar._backchannel_boundary = None
    ar._backchannel_boundary_timer = None
    ar._backchannel_boundary_callback = None
    ar._ignore_user_transcript_until = NOT_GIVEN
    ar._overlap_in_current_turn = False
    ar._overlap_open = False
    ar._turn_backchannel_over_agent = False
    ar._transcript_buffer = []
    ar._tasks = set()
    ar._user_silence_ev = asyncio.Event()
    ar._user_silence_ev.set()
    ar._hooks = MagicMock()
    return ar, ch


def _sentinel_names(ch: _RecordingChan) -> list[str]:
    return [type(item).__name__ for item in ch.sent]


async def test_pause_ends_overlap_inference() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ch.sent.clear()

    ar._on_end_of_agent_speech(ignore_user_transcript_until=time.time(), paused=True)

    assert _sentinel_names(ch) == ["_AgentSpeechEndedSentinel"]
    assert ar._overlap_open is False


async def test_user_speech_ending_after_pause_does_not_close_overlap_again() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ar._on_end_of_agent_speech(ignore_user_transcript_until=time.time(), paused=True)
    ch.sent.clear()

    ar._on_end_of_speech(ended_at=time.time())

    assert _sentinel_names(ch) == []


async def test_resume_restarts_the_detector() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ar._on_end_of_agent_speech(ignore_user_transcript_until=time.time(), paused=True)
    ch.sent.clear()

    ar._on_start_of_agent_speech(started_at=time.time())

    assert _sentinel_names(ch) == ["_AgentSpeechStartedSentinel"]


async def test_a_resolved_overlap_is_not_closed_again() -> None:
    # a turn can hold several overlaps, so closing keys off the open one rather than the turn
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    await ar._on_overlap_speech_event(_overlap_event(is_interruption=True, agent_ended=False))
    ch.sent.clear()

    ar._on_end_of_speech(ended_at=time.time())

    assert _sentinel_names(ch) == []


async def test_real_end_of_agent_speech_still_tears_down() -> None:
    # the agent turn genuinely ending must stop the inference
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ch.sent.clear()

    ar._on_end_of_agent_speech(ignore_user_transcript_until=time.time())

    assert _sentinel_names(ch) == [
        "_OverlapSpeechEndedSentinel",
        "_AgentSpeechEndedSentinel",
    ]
    assert ch.sent[0]._agent_ended is True  # type: ignore[attr-defined]
