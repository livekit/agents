from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import LanguageCode, stt
from livekit.plugins.speechmatics.stt import SpeechStream, STTOptions, _extract_confidence


def _stream() -> tuple[SpeechStream, list[stt.SpeechEvent]]:
    stream = SpeechStream.__new__(SpeechStream)
    events: list[stt.SpeechEvent] = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)  # type: ignore[attr-defined]
    stream._stt = SimpleNamespace(  # type: ignore[attr-defined]
        _stt_options=STTOptions(language=LanguageCode("en")),
    )
    stream.start_time_offset = 0.25  # type: ignore[attr-defined]
    return stream, events


def _segment(**overrides: Any) -> dict[str, Any]:
    segment: dict[str, Any] = {
        "text": "hello",
        "language": "en",
        "speaker_id": "S1",
        "metadata": {"start_time": 1.0, "end_time": 2.0},
    }
    segment.update(overrides)
    return segment


def test_send_frames_uses_reported_confidence() -> None:
    stream, events = _stream()

    stream._send_frames([_segment(confidence=0.42)], is_final=True)

    assert events[0].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    data = events[0].alternatives[0]
    assert data.confidence == pytest.approx(0.42)
    assert data.start_time == pytest.approx(1.25)
    assert data.end_time == pytest.approx(2.25)


def test_send_frames_defaults_unknown_confidence_to_zero() -> None:
    stream, events = _stream()

    stream._send_frames([_segment()], is_final=False)

    assert events[0].type == stt.SpeechEventType.INTERIM_TRANSCRIPT
    assert events[0].alternatives[0].confidence == 0.0


def test_extract_confidence_averages_fragment_confidence() -> None:
    assert _extract_confidence(
        {
            "fragments": [
                {"confidence": 0.2},
                {"confidence": 0.8},
                {"confidence": True},
                {"confidence": "0.5"},
            ]
        }
    ) == pytest.approx(0.5)
