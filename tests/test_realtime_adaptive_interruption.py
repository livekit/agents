from __future__ import annotations

import asyncio
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions
from livekit.agents.inference import OverlappingSpeechEvent
from livekit.agents.types import NOT_GIVEN, NotGivenOr
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


def test_adaptive_verdict_enables_audio_activity_after_releasing_gate() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._interruption_detected = False
    activity._interruption_by_audio_activity_enabled = False
    activity._default_interruption_by_audio_activity_enabled = True
    activity._audio_recognition = MagicMock()
    activity._session = MagicMock()
    calls: list[str] = []

    event = OverlappingSpeechEvent(is_interruption=True)

    def _apply_verdict(ev: OverlappingSpeechEvent) -> None:
        assert ev is event
        assert activity._interruption_detected
        assert not activity._interruption_by_audio_activity_enabled
        calls.append("apply")

    def _interrupt() -> None:
        assert activity._interruption_by_audio_activity_enabled
        calls.append("interrupt")

    activity._audio_recognition._on_overlap_speech_event.side_effect = _apply_verdict
    activity._audio_recognition._cancel_backchannel_boundary.side_effect = lambda: calls.append(
        "enable"
    )
    activity._interrupt_by_audio_activity = MagicMock(  # type: ignore[method-assign]
        side_effect=_interrupt
    )

    activity.on_overlap_speech(event)

    assert calls == ["apply", "enable", "interrupt"]
    assert activity._interruption_detected
    assert activity._interruption_by_audio_activity_enabled
    activity._session.emit.assert_called_once_with("overlapping_speech", event)
    activity._interrupt_by_audio_activity.assert_called_once_with()  # type: ignore[attr-defined]


def test_audio_activity_waits_for_min_words() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._interruption_by_audio_activity_enabled = True
    activity._rt_turn_detection_enabled = False
    activity._rt_session = None
    activity._agent = MagicMock()
    activity._session = MagicMock()
    activity._session._text_only = False
    activity._session._aec_warmup_remaining = 0
    activity._session._aec_warmup_timer = None
    activity._session.options = SimpleNamespace(interruption={"min_words": 2})
    activity._session.agent_state = "speaking"
    activity._audio_recognition = MagicMock()
    activity._audio_recognition._current_transcript = "short"
    activity._audio_recognition._endpointing.overlapping = True
    activity._current_speech = MagicMock()
    activity._current_speech.interrupted = False
    activity._current_speech.allow_interruptions = True
    activity._cancel_false_interruption_timer = MagicMock()  # type: ignore[method-assign]
    activity._pause_enabled = MagicMock(return_value=False)  # type: ignore[method-assign]

    activity._interrupt_by_audio_activity()

    activity._current_speech.interrupt.assert_not_called()

    activity._audio_recognition._current_transcript = "now enough words"
    activity._interrupt_by_audio_activity()

    activity._current_speech.interrupt.assert_called_once_with()


def test_rejected_audio_interruption_clears_confirmed_verdict() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._interruption_by_audio_activity_enabled = True
    activity._rt_turn_detection_enabled = False
    activity._rt_session = None
    activity._agent = MagicMock()
    activity._session = MagicMock()
    activity._session._aec_warmup_remaining = 0
    activity._session._aec_warmup_timer = None
    activity._session.options = SimpleNamespace(interruption={"min_words": 0})
    activity._audio_recognition = MagicMock()
    activity._current_speech = None
    activity._interruption_detected = True

    activity._interrupt_by_audio_activity()

    assert not activity._interruption_detected


def test_boundary_expiry_disables_audio_activity_interruption() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._session = MagicMock()
    activity._session.agent_state = "speaking"
    activity._audio_recognition = MagicMock()
    activity._audio_recognition._backchannel_boundary_active = True
    activity._interruption_by_audio_activity_enabled = True

    activity._disable_vad_interruption_soon()
    activity._audio_recognition._backchannel_boundary_callback()

    assert not activity._interruption_by_audio_activity_enabled


def test_active_user_speech_keeps_audio_activity_for_zero_boundary() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._audio_recognition = MagicMock()
    activity._audio_recognition._backchannel_boundary_active = False
    activity._audio_recognition._speaking = True
    activity._interruption_by_audio_activity_enabled = True

    activity._disable_vad_interruption_soon()

    assert activity._interruption_by_audio_activity_enabled


def test_replayed_start_preserves_confirmed_interruption() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._session = MagicMock()
    activity._session.agent_state = "speaking"
    activity._audio_recognition = MagicMock()
    activity._interruption_detected = True
    activity._user_silence_event = asyncio.Event()
    activity._stt_eos_received = True
    activity._cancel_false_interruption_timer = MagicMock()  # type: ignore[method-assign]

    speech_start_time = time.time()
    activity.on_start_of_speech(None, speech_start_time=speech_start_time)

    activity._audio_recognition._on_start_of_speech.assert_called_once_with(
        started_at=speech_start_time,
        speech_duration=0.0,
        user_speaking_span=activity._session._user_speaking_span,
        skip_adaptive_interruption=True,
    )


def test_agent_speech_end_clears_confirmed_interruption_after_recognition_teardown() -> None:
    activity = AgentActivity.__new__(AgentActivity)
    activity._interruption_detected = True
    activity._audio_recognition = MagicMock()

    def _end_agent_speech(*, ended_at: float) -> None:
        assert ended_at == 10.0
        assert activity._interruption_detected

    activity._audio_recognition._on_end_of_agent_speech.side_effect = _end_agent_speech

    activity._on_end_of_agent_speech(ended_at=10.0)

    assert not activity._interruption_detected


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
    # the non-realtime gate still needs STT because VAD alone cannot provide text
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
    ar._agent_speaking = False
    ar._transcript_gate_active = False
    ar._backchannel_boundary_timer = None
    ar._overlap_in_current_turn = True
    ar._turn_backchannel_over_agent = False
    ar._user_silence_ev = asyncio.Event()
    if not speaking:
        ar._user_silence_ev.set()  # silent (between segments) unless told otherwise
    ar._hooks = MagicMock()
    ar._hooks.interruption_by_audio_activity_enabled = False
    return ar


def _swallow_task(coro: object, **kwargs: object) -> MagicMock:
    """Stand in for _create_speech_task: the reply pipeline isn't under test here."""
    coro.close()  # type: ignore[attr-defined]
    return MagicMock()


def _overlap_event(*, is_interruption: bool, agent_ended: bool) -> OverlappingSpeechEvent:
    return OverlappingSpeechEvent(is_interruption=is_interruption, agent_ended=agent_ended)


def test_user_ended_overlap_latches_backchannel() -> None:
    # the user's overlap ended on its own with no interruption flagged — a real backchannel
    ar = _recognition_for_overlap()
    ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    assert ar._turn_backchannel_over_agent is True


def test_false_verdict_trims_finished_backchannel() -> None:
    ar = _recognition_for_overlap()
    old_event = MagicMock(created_at=8.0, speech_end_time=None)
    recent_event = MagicMock(created_at=9.5, speech_end_time=None)
    ar._agent_speaking = True
    ar._transcript_gate_active = True
    ar._agent_speech_started_at = None
    ar._active_vad_speech_started_at = None
    ar._backchannel_boundary = (0.0, 1.0)
    ar._transcript_buffer = deque([old_event, recent_event])

    ar._on_overlap_speech_event(
        OverlappingSpeechEvent(
            is_interruption=False,
            agent_ended=False,
            detected_at=10.0,
        )
    )

    assert list(ar._transcript_buffer) == [recent_event]


def test_boundary_fallback_preserves_held_transcripts() -> None:
    ar = _recognition_for_overlap()
    early_event = MagicMock(created_at=9.0, speech_end_time=None)
    recent_event = MagicMock(created_at=9.75, speech_end_time=None)
    ar._agent_speaking = True
    ar._transcript_gate_active = True
    ar._agent_speech_started_at = 9.0
    ar._active_vad_speech_started_at = None
    ar._backchannel_boundary = (3.0, 0.5)
    ar._backchannel_boundary_timer = MagicMock()
    ar._transcript_buffer = deque([early_event, recent_event])

    ar._on_overlap_speech_event(
        OverlappingSpeechEvent(
            is_interruption=False,
            agent_ended=False,
            detected_at=10.0,
        )
    )

    assert list(ar._transcript_buffer) == [early_event, recent_event]


def test_confirmed_backchannel_between_segments_clears_audio() -> None:
    # confirmed between segments (user silent) — cleared so it can't prefix the next turn
    ar = _recognition_for_overlap(speaking=False)
    ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    ar._hooks.on_backchannel_confirmed.assert_called_once()


def test_confirmed_backchannel_while_speaking_defers_clear() -> None:
    # user already mid next-segment — latch the verdict but defer the clear (else we'd clip it)
    ar = _recognition_for_overlap(speaking=True)
    ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=False))
    assert ar._turn_backchannel_over_agent is True
    ar._hooks.on_backchannel_confirmed.assert_not_called()


def test_agent_ended_overlap_is_not_a_backchannel() -> None:
    # the overlap ended because the agent finished, not the user — the user may still be
    # mid-turn, so this inconclusive verdict must not mark the turn a backchannel
    ar = _recognition_for_overlap()
    ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=True))
    assert ar._turn_backchannel_over_agent is False
    ar._hooks.on_backchannel_confirmed.assert_not_called()


def test_agent_ended_overlap_preserves_prior_backchannel() -> None:
    # a real backchannel was already latched this turn; the later agent-ended overlap is a
    # no-op and must not clear it
    ar = _recognition_for_overlap()
    ar._turn_backchannel_over_agent = True
    ar._on_overlap_speech_event(_overlap_event(is_interruption=False, agent_ended=True))
    assert ar._turn_backchannel_over_agent is True


def test_interruption_clears_backchannel() -> None:
    # a confirmed interruption supersedes any prior backchannel verdict for the turn
    ar = _recognition_for_overlap()
    ar._turn_backchannel_over_agent = True
    ar._on_overlap_speech_event(_overlap_event(is_interruption=True, agent_ended=False))
    assert ar._turn_backchannel_over_agent is False
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
    ar._transcript_gate_active = False
    ar._active_vad_speech_started_at = None
    ar._endpointing = MagicMock()
    ar._backchannel_boundary = None
    ar._backchannel_boundary_timer = None
    ar._backchannel_boundary_callback = None
    ar._overlap_in_current_turn = False
    ar._overlap_open = False
    ar._turn_backchannel_over_agent = False
    ar._transcript_buffer = deque()
    ar._tasks = set()
    ar._user_silence_ev = asyncio.Event()
    ar._user_silence_ev.set()
    ar._hooks = MagicMock()
    ar._hooks.interruption_by_audio_activity_enabled = False
    ar._session = MagicMock()
    return ar, ch


def _sentinel_names(ch: _RecordingChan) -> list[str]:
    return [type(item).__name__ for item in ch.sent]


def test_positive_verdict_does_not_reopen_overlap_during_transcript_replay() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._agent_speaking = True
    ar._agent_speech_started_at = 9.0
    ar._overlap_in_current_turn = True
    ar._overlap_open = True
    ar._transcript_gate_active = True
    ar._transcript_buffer.append(MagicMock(created_at=9.5, speech_end_time=None))
    ar._process_stt_event = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda _: ar._on_start_of_speech(
            started_at=9.5, skip_adaptive_interruption=True
        )
    )

    event = OverlappingSpeechEvent(
        is_interruption=True,
        detected_at=10.0,
        overlap_started_at=9.0,
    )
    ar._on_overlap_speech_event(event)

    assert ar._overlap_open is False
    assert ar._transcript_gate_active is False
    assert _sentinel_names(ch) == []


@pytest.mark.parametrize("interruption", [NOT_GIVEN, True, False])
def test_end_of_speech_passes_interruption_verdict_to_endpointing(
    interruption: NotGivenOr[bool],
) -> None:
    ar, _ = _recognition_with_interruption_ch()
    ar._speaking = True

    ar._on_end_of_speech(ended_at=10.0, interruption=interruption)

    ar._endpointing.on_end_of_speech.assert_called_once_with(
        ended_at=10.0,
        interruption=interruption,
    )


async def test_agent_speech_end_closes_overlap_before_reset() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ch.sent.clear()

    ar._on_end_of_agent_speech(ended_at=time.time())

    assert _sentinel_names(ch) == [
        "_OverlapSpeechEndedSentinel",
        "_AgentSpeechEndedSentinel",
    ]
    assert ch.sent[0]._agent_ended is True  # type: ignore[attr-defined]
    assert ar._overlap_open is False


async def test_user_speech_ending_after_agent_end_does_not_close_overlap_again() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ar._on_end_of_agent_speech(ended_at=time.time())
    ch.sent.clear()

    ar._on_end_of_speech(ended_at=time.time())

    assert _sentinel_names(ch) == []


async def test_resume_with_active_user_speech_stays_with_audio_activity() -> None:
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    user_started_at = time.time()
    ar._speaking = True
    ar._on_start_of_speech(started_at=user_started_at)
    ar._on_end_of_agent_speech(ended_at=time.time())
    ch.sent.clear()

    ar._hooks.interruption_by_audio_activity_enabled = True
    resumed_at = time.time()
    ar._on_start_of_agent_speech(started_at=resumed_at)

    assert _sentinel_names(ch) == ["_AgentSpeechStartedSentinel"]
    assert ar._hooks.interruption_by_audio_activity_enabled
    assert not ar._transcript_gate_active
    ar._endpointing.on_start_of_speech.assert_called_once_with(
        started_at=user_started_at, overlapping=True
    )


async def test_a_resolved_overlap_is_not_closed_again() -> None:
    # a turn can hold several overlaps, so closing keys off the open one rather than the turn
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ar._on_overlap_speech_event(_overlap_event(is_interruption=True, agent_ended=False))
    ch.sent.clear()

    ar._on_end_of_speech(ended_at=time.time())

    assert _sentinel_names(ch) == []


async def test_real_end_of_agent_speech_still_tears_down() -> None:
    # the agent turn genuinely ending must stop the inference
    ar, ch = _recognition_with_interruption_ch()
    ar._on_start_of_agent_speech(started_at=time.time())
    ar._on_start_of_speech(started_at=time.time())
    ch.sent.clear()

    ar._on_end_of_agent_speech(ended_at=time.time())

    assert _sentinel_names(ch) == [
        "_OverlapSpeechEndedSentinel",
        "_AgentSpeechEndedSentinel",
    ]
    assert ch.sent[0]._agent_ended is True  # type: ignore[attr-defined]
