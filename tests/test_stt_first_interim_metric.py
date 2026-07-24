"""Unit coverage for first-interim STT responsiveness metrics."""

import pytest

from livekit.agents.voice.audio_recognition import _compute_end_of_turn_metrics

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


def test_first_interim_delay_is_unavailable_without_interim() -> None:
    metrics = _compute_end_of_turn_metrics(
        speech_start_time=10.0,
        last_speaking_time=14.0,
        last_final_transcript_time=14.5,
        first_interim_time=None,
        now=15.0,
    )
    assert metrics.first_interim_delay is None
