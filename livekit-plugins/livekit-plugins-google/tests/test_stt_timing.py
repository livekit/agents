"""Regression tests for Google STT result timing.

``chirp_3`` forces ``enable_word_time_offsets=False``, so its results carry text
but no words. Those results must report their timing as unknown (0) instead of
the stream offset: the SDK reads any ``end_time > 0`` as the provider's end of
speech, which freezes the end-of-turn anchor in ``turn_detection="stt"``.
"""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.plugins.google.stt import _streaming_recognize_response_to_speech_data

pytestmark = pytest.mark.unit


def _word(text: str, start: float, end: float) -> Any:
    return SimpleNamespace(
        word=text,
        start_offset=timedelta(seconds=start),
        end_offset=timedelta(seconds=end),
        confidence=0.9,
    )


def _result(*, text: str, words: list[Any], is_final: bool = True, confidence: float = 0.8) -> Any:
    return SimpleNamespace(
        alternatives=[SimpleNamespace(transcript=text, confidence=confidence, words=words)],
        is_final=is_final,
        language_code="en-US",
    )


def _response(results: list[Any]) -> Any:
    return SimpleNamespace(results=results)


def test_final_without_word_offsets_reports_unknown_timing() -> None:
    """chirp_3 finals have no words: timing stays 0 rather than the stream offset."""
    resp = _response([_result(text="hello world", words=[])])

    data = _streaming_recognize_response_to_speech_data(
        resp, min_confidence_threshold=0.0, start_time_offset=12.5
    )

    assert data is not None
    assert data.text == "hello world"
    assert data.start_time == 0.0
    assert data.end_time == 0.0
    assert data.words is None


def test_interim_without_word_offsets_reports_unknown_timing() -> None:
    resp = _response([_result(text="hello", words=[], is_final=False, confidence=0.5)])

    data = _streaming_recognize_response_to_speech_data(
        resp, min_confidence_threshold=0.0, start_time_offset=12.5
    )

    assert data is not None
    assert data.text == "hello"
    assert data.start_time == 0.0
    assert data.end_time == 0.0


def test_final_with_words_keeps_the_stream_timeline() -> None:
    """Word-based results keep the stream offset applied and expose word timing."""
    words = [_word("hello", 1.0, 1.4), _word("world", 1.4, 1.9)]
    resp = _response([_result(text="hello world", words=words)])

    data = _streaming_recognize_response_to_speech_data(
        resp, min_confidence_threshold=0.0, start_time_offset=10.0
    )

    assert data is not None
    assert data.start_time == pytest.approx(11.0)
    assert data.end_time == pytest.approx(11.9)
    assert data.words is not None
    assert [str(word) for word in data.words] == ["hello", "world"]
    assert data.words[0].start_time == pytest.approx(11.0)
    assert data.words[0].end_time == pytest.approx(11.4)


def test_low_confidence_interim_is_dropped() -> None:
    resp = _response([_result(text="hmm", words=[], is_final=False, confidence=0.1)])

    data = _streaming_recognize_response_to_speech_data(
        resp, min_confidence_threshold=0.5, start_time_offset=3.0
    )

    assert data is None
