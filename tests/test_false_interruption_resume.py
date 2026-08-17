"""The false-interruption resume must follow the end-of-turn decision, never race it.

Both are armed from the same VAD ``END_OF_SPEECH``, but the turn commits at
``last_speaking_time + endpointing_delay`` — measured from before the VAD silence window —
so a ``max_delay`` decision lands after the resume timer.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions
from livekit.agents.voice.agent_activity import AgentActivity, _PausedSpeechInfo
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _EndOfTurnInfo,
    _EndOfTurnMetrics,
)
from livekit.agents.voice.turn import (
    TurnDetectionEvent,
    _StreamingTurnDetector,
    _StreamingTurnDetectorStream,
)

from .fake_io import FakeAudioOutput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit

# scaled-down shipped defaults, keeping max_delay - vad_min_silence > timeout
VAD_MIN_SILENCE = 0.05
FALSE_INTERRUPTION_TIMEOUT = 0.3
MAX_DELAY = 0.5


def _recognition(hooks: AgentActivity, last_speaking_time: float) -> AudioRecognition:
    """AudioRecognition wired to drive one real eou bounce against ``hooks``."""
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._session = MagicMock()
    ar._hooks = hooks
    ar._stt = None  # realtime model, no STT
    ar._audio_transcript = ""
    ar._turn_detection_mode = None
    ar._turn_detector = MagicMock(spec=_StreamingTurnDetector)

    stream_mock = MagicMock(spec=_StreamingTurnDetectorStream)
    stream_mock.supports_language = AsyncMock(return_value=True)
    stream_mock.unlikely_threshold = AsyncMock(return_value=0.5)
    stream_mock.backchannel_threshold = AsyncMock(return_value=None)
    stream_mock.flush = MagicMock()
    stream_mock.cancel_inference = MagicMock()
    stream_mock.aclose = AsyncMock()
    stream_mock.prediction_timeout = 0.5
    ar._turn_detector_stream = stream_mock

    # the detector already answered: mid-utterance, so the bounce waits out max_delay
    fut: asyncio.Future[TurnDetectionEvent] = asyncio.Future()
    fut.set_result(
        TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=0.1,
            last_speaking_time=last_speaking_time,
        )
    )
    ar._turn_detector_prediction_fut = fut
    ar._turn_detector_flushed = False
    ar._turn_detector_late_prediction_warned = False
    ar._agent_speaking = False
    ar._interruption_enabled = False
    ar._interruption_ch = None
    ar._vad_base_turn_detection = True
    ar._backchannel_boundary = None
    ar._backchannel_boundary_timer = None
    ar._backchannel_boundary_callback = None

    endpointing = MagicMock()
    endpointing.min_delay = 0.05
    endpointing.max_delay = MAX_DELAY
    ar._endpointing = endpointing

    ar._ensure_user_turn_span = MagicMock(  # type: ignore[method-assign]
        return_value=MagicMock(is_recording=MagicMock(return_value=False))
    )
    ar._user_turn_span = None
    ar._user_turn_start = None
    ar._user_silence_ev = asyncio.Event()
    ar._speaking = False
    ar._final_transcript_confidence = []
    ar._stt_request_ids = []
    ar._last_speaking_time = last_speaking_time
    ar._last_final_transcript_time = None
    ar._speech_start_time = None
    ar._vad_speech_started = False
    ar._end_of_turn_task = None
    ar._user_turn_committed = False
    ar._vad = None
    ar._last_language = None
    ar._last_emitted_prediction = None
    ar._turn_backchannel_over_agent = False
    ar._overlap_in_current_turn = False
    ar._turn_tracker = MagicMock()
    ar._closing = asyncio.Event()
    # only touched by _aclose
    ar._commit_user_turn_atask = None
    ar._stt_pipeline = None
    ar._stt_consumer_atask = None
    ar._vad_atask = None
    ar._interruption_atask = None
    ar._tasks = set()
    return ar


def _paused_activity(session: AgentSession) -> tuple[AgentActivity, MagicMock]:
    activity = AgentActivity(Agent(instructions="test"), session)
    activity._scheduling_paused = False
    session.output.audio = FakeAudioOutput(can_pause=True)

    handle = MagicMock()
    handle.done.return_value = False
    handle.interrupted = False
    handle.allow_interruptions = True
    handle._agent_turn_context = None
    activity._current_speech = handle
    activity._paused_speech = _PausedSpeechInfo(
        handle=handle, agent_state="speaking", timeout=FALSE_INTERRUPTION_TIMEOUT
    )
    return activity, handle


def _eot_info(*, skip_reply: bool = False) -> _EndOfTurnInfo:
    return _EndOfTurnInfo(
        skip_reply=skip_reply,
        new_transcript="",
        transcript_confidence=0.0,
        metrics=_EndOfTurnMetrics(
            started_speaking_at=None,
            stopped_speaking_at=None,
            transcription_delay=None,
            end_of_turn_delay=None,
        ),
        backchannel_over_agent=False,
    )


def _swallow_task(coro: object, **kwargs: object) -> MagicMock:
    """Stand in for _create_speech_task: the reply pipeline isn't under test here."""
    coro.close()  # type: ignore[attr-defined]
    return MagicMock()


def _session() -> AgentSession:
    return AgentSession(
        llm=FakeRealtimeModel(capabilities=fake_capabilities(turn_detection=False)),
        vad=FakeVAD(fake_user_speeches=[]),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "adaptive"},
        ),
    )


async def test_resume_waits_for_a_dropped_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)

    events: list[tuple[str, float]] = []
    session.on("agent_false_interruption", lambda _: events.append(("resume", time.time())))

    # t0: VAD END_OF_SPEECH — the resume timer is armed, then the bounce is scheduled
    t0 = time.time()
    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(activity, last_speaking_time=t0 - VAD_MIN_SILENCE)
    # a confirmed backchannel is what makes the turn drop rather than commit
    activity._audio_recognition._turn_backchannel_over_agent = True
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    await asyncio.sleep(MAX_DELAY + 0.3)
    await session.aclose()

    assert [name for name, _ in events] == ["resume"]
    resumed_at = events[0][1] - t0
    # the turn is dropped at max_delay - vad silence; resuming any earlier is the race
    assert resumed_at == pytest.approx(MAX_DELAY - VAD_MIN_SILENCE, abs=0.1)
    assert activity._paused_speech is None


async def test_committed_turn_suppresses_the_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)
    # a confirmed interruption: on_end_of_turn commits and the reply task interrupts the pause
    activity._interruption_detected = True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(
        activity, last_speaking_time=time.time() - VAD_MIN_SILENCE
    )
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    await asyncio.sleep(MAX_DELAY + 0.3)
    await session.aclose()

    assert events == []
    assert activity._false_interruption_pending is False
    assert activity._paused_speech is not None  # left for _cancel_speech_pause to interrupt


async def test_teardown_does_not_resume_a_deferred_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    # closing settles the open decision without deciding anything; the close path releases the
    # pause itself, so the resume must not fire mid-teardown
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    activity.on_end_of_speech(None)
    activity._audio_recognition = _recognition(
        activity, last_speaking_time=time.time() - VAD_MIN_SILENCE
    )
    activity._audio_recognition._run_eou_detection(MagicMock(), trigger="vad")

    # let the timeout elapse while the decision is still open
    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.05)
    assert activity._false_interruption_pending is True

    await activity._audio_recognition._aclose()
    await asyncio.sleep(0.2)
    await session.aclose()

    assert events == []


async def test_skipped_reply_keeps_the_resume_armed(monkeypatch: pytest.MonkeyPatch) -> None:
    # a skipped reply (AMD verdict, or commit_user_turn on a realtime session) returns before
    # _cancel_speech_pause, so the resume is the only thing that can un-pause the speech
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)
    activity._interruption_detected = True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    activity.on_end_of_speech(None)
    assert activity.on_end_of_turn(_eot_info(skip_reply=True)) is True

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.2)
    await session.aclose()

    assert events == ["resume"]
    assert activity._paused_speech is None


async def test_server_side_turn_detection_keeps_the_resume_armed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the reply task returns before _cancel_speech_pause when the server owns turn taking, so
    # only the resume can release the pre-pause taken while the agent was still generating
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)
    activity._rt_turn_detection_enabled = True
    activity._create_speech_task = _swallow_task  # type: ignore[method-assign, assignment]

    events: list[str] = []
    session.on("agent_false_interruption", lambda _: events.append("resume"))

    activity.on_end_of_speech(None)
    assert activity.on_end_of_turn(_eot_info()) is True

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.2)
    await session.aclose()

    assert events == ["resume"]
    assert activity._paused_speech is None


async def test_interrupting_paused_speech_does_not_end_agent_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, handle = _paused_activity(session)
    handle._generations = []
    activity._audio_recognition = MagicMock()

    await activity._cancel_speech_pause()
    await session.aclose()

    handle.interrupt.assert_called_once()
    activity._audio_recognition._on_end_of_agent_speech.assert_not_called()


async def test_handing_over_a_paused_speech_does_not_end_the_agent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a handoff keeps the speech for the next activity, and the recognition is closed by then
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, handle = _paused_activity(session)
    activity._audio_recognition = MagicMock()

    await activity._cancel_speech_pause(interrupt=False)
    await session.aclose()

    handle.interrupt.assert_not_called()
    activity._audio_recognition._on_end_of_agent_speech.assert_not_called()


async def test_resume_is_immediate_when_no_turn_decision_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # VAD-only noise on an stt pipeline never starts a bounce, so the timeout still rules
    monkeypatch.setenv("LIVEKIT_API_KEY", "k")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "s")

    session = _session()
    activity, _ = _paused_activity(session)

    events: list[tuple[str, float]] = []
    session.on("agent_false_interruption", lambda _: events.append(("resume", time.time())))

    t0 = time.time()
    activity.on_end_of_speech(None)

    await asyncio.sleep(FALSE_INTERRUPTION_TIMEOUT + 0.2)
    await session.aclose()

    assert [name for name, _ in events] == ["resume"]
    assert events[0][1] - t0 == pytest.approx(FALSE_INTERRUPTION_TIMEOUT, abs=0.1)
