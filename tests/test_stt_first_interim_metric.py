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


@pytest.mark.parametrize("blank_text", ["", "  \t\n"])
async def test_blank_interim_does_not_capture_first_interim_time(blank_text: str) -> None:
    session = MagicMock()
    session.options.interruption = {}
    recognition = AudioRecognition(
        session,
        hooks=MagicMock(),
        endpointing=MagicMock(),
        stt=None,
        vad=None,
        interruption_detection=None,
        turn_detection="vad",
    )
    recognition._speech_start_time = 10.0

    blank_interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="", text=blank_text)],
        created_at=11.0,
    )
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=11.0):
        await recognition._on_stt_event(blank_interim)

    assert recognition._first_interim_time is None

    populated_interim = SpeechEvent(
        type=SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[SpeechData(language="", text="hello")],
        created_at=12.0,
    )
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=12.0):
        await recognition._on_stt_event(populated_interim)

    assert recognition._first_interim_time == 12.0

    # Processing a buffered first update uses receipt time, not replay time.
    recognition._first_interim_time = None
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=20.0):
        recognition._process_stt_event(populated_interim)
    assert recognition._first_interim_time == 12.0

    # A later update cannot replace the first receipt.
    populated_interim.created_at = 13.0
    with patch("livekit.agents.voice.audio_recognition.time.time", return_value=20.0):
        recognition._process_stt_event(populated_interim)
    assert recognition._first_interim_time == 12.0


def test_existing_eou_payload_remains_valid() -> None:
    from livekit.agents.metrics import EOUMetrics

    metrics = EOUMetrics(
        timestamp=1.0,
        end_of_utterance_delay=0.1,
        transcription_delay=0.2,
        on_user_turn_completed_delay=0.3,
    )
    assert metrics.first_interim_delay is None
    assert metrics.first_interim_status is None
