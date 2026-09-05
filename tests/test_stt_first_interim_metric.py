"""Unit coverage for first-interim STT responsiveness metrics."""

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents.stt import SpeechData, SpeechEvent, SpeechEventType
from livekit.agents.voice.audio_recognition import (
    AudioRecognition,
    _compute_end_of_turn_metrics,
)

pytestmark = pytest.mark.unit


def test_first_interim_delay_uses_speech_onset() -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=10.0,
        last_speaking_time=14.0,
        last_final_transcript_time=14.5,
        first_interim_time=10.25,
        now=15.0,
    )
    assert metrics.first_interim_delay == pytest.approx(0.25)
    assert metrics.first_interim_status == "received"


def test_first_interim_delay_is_unavailable_without_interim() -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=10.0,
        last_speaking_time=14.0,
        last_final_transcript_time=14.5,
        first_interim_time=None,
        now=15.0,
    )
    assert metrics.first_interim_delay is None
    assert metrics.first_interim_status == "absent"


async def test_blank_interim_does_not_capture_first_interim_time() -> None:
    recognition = AudioRecognition.__new__(AudioRecognition)
    recognition._turn_detection_mode = "vad"
    recognition._interruption_enabled = False
    recognition._first_interim_time = None
    recognition._speech_start_time = 10.0
    recognition._hooks = MagicMock()
    recognition._vad = None

    blank_interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="", text="")],
    )
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=11.0):
        await recognition._on_stt_event(blank_interim)

    assert recognition._first_interim_time is None

    populated_interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="", text="hello")],
    )
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=12.0):
        await recognition._on_stt_event(populated_interim)

    assert recognition._first_interim_time == 12.0
