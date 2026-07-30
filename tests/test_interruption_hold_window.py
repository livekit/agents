"""Unit tests for the bounded ignore-user-transcript window in ``AudioRecognition``.

The adaptive-interruption hold logic decides whether an STT transcript falls inside
the window during which user transcripts are ignored while overlapping speech is
adjudicated. It must bound the event's wall-clock time so a *mis-anchored* timestamp
— e.g. from a fallback STT leg that reset its timeline and reports near-zero relative
timestamps — cannot be mistaken for a transcript inside the window and held (which
would wedge the user's turn). The valid window is
``agent_start < event_time < min(now, ignore_until)``.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import AudioRecognition

pytestmark = pytest.mark.unit


def _make_recognition(
    *,
    ignore_until: float,
    agent_started: float | None,
    input_started: float,
) -> AudioRecognition:
    ar = AudioRecognition.__new__(AudioRecognition)
    ar._interruption_enabled = True
    ar._agent_speaking = False
    ar._ignore_user_transcript_until = ignore_until
    ar._agent_speech_started_at = agent_started
    pipeline = MagicMock()
    pipeline.input_started_at = input_started
    ar._stt_pipeline = pipeline
    return ar


def _final(start: float, end: float) -> SpeechEvent:
    return SpeechEvent(
        type=SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[SpeechData(language="", text="hi", start_time=start, end_time=end)],
    )


def test_holds_in_window_transcript() -> None:
    # input started at t=1000; agent spoke from t=1005; ignore window until t=1010.
    ar = _make_recognition(ignore_until=1010.0, agent_started=1005.0, input_started=1000.0)
    # event audio at +7s -> wall 1007, inside [1005, 1010) -> ignored, held
    assert ar._should_hold_stt_event(_final(7.0, 8.0)) is True


def test_does_not_hold_timestamp_anchored_before_agent_speech() -> None:
    # The regression: a mis-anchored leg reports a near-zero relative timestamp,
    # computing to a wall-clock *before* the agent started speaking. That cannot belong
    # to this agent's overlap window and must NOT be held.
    ar = _make_recognition(ignore_until=1010.0, agent_started=1005.0, input_started=1000.0)
    # event audio at +2s -> wall 1002, before agent_started 1005 -> not held
    assert ar._should_hold_stt_event(_final(2.0, 3.0)) is False


def test_does_not_hold_after_ignore_cutoff() -> None:
    ar = _make_recognition(ignore_until=1010.0, agent_started=1005.0, input_started=1000.0)
    # event audio at +15s -> wall 1015, after ignore cutoff 1010 -> not held
    assert ar._should_hold_stt_event(_final(15.0, 16.0)) is False


def test_does_not_hold_timestamp_in_the_future() -> None:
    # Upper bound is clamped to now: a timestamp computed into the future (the
    # over-anchored variant) is outside the window and not held.
    now = time.time()
    ar = _make_recognition(
        ignore_until=now + 100.0,
        agent_started=now - 1.0,
        input_started=now,
    )
    # event audio at +200s -> wall now+200, beyond now -> not held
    assert ar._should_hold_stt_event(_final(200.0, 201.0)) is False


def test_lower_bound_is_agent_start_across_multiple_overlaps() -> None:
    # A turn may contain several overlap episodes. The lower bound stays at the agent
    # speech start, so a transcript from an *earlier* overlap is still held even after a
    # later overlap occurs (the bound must not advance to the latest overlap start).
    ar = _make_recognition(ignore_until=1010.0, agent_started=1005.0, input_started=1000.0)
    # first overlap transcript at +6s -> wall 1006, still inside [1005, 1010) -> held
    assert ar._should_hold_stt_event(_final(6.0, 6.5)) is True
    # later overlap transcript at +8s -> wall 1008 -> also held
    assert ar._should_hold_stt_event(_final(8.0, 8.5)) is True
