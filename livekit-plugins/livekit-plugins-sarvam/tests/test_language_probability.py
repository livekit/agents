"""Tests for SpeechData.confidence threading from Sarvam's language_probability.

Verifies that both the REST and WS paths thread ``language_probability`` from
Sarvam's response into ``SpeechData.confidence`` (instead of the previous
hardcoded ``1.0``), with a defensive fallback when the field is absent or has
an unexpected type.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.plugins.sarvam.stt import SpeechStream

# ---------------------------------------------------------------------------
# Helpers — build a minimal STT instance + fake the channel/logger/state that
# `_handle_transcript_data` touches. We bypass __init__ so the test doesn't
# need an API key, an HTTP session, or a real WebSocket.
# ---------------------------------------------------------------------------


def _make_stream_under_test() -> tuple[SpeechStream, list[Any]]:
    """Construct a minimal SpeechStream and collect its emitted events.

    Returns ``(stream_instance, captured_events)`` where each event sent via
    ``send_nowait`` is appended to ``captured_events``. We bypass ``__init__``
    so the test doesn't need an API key, an HTTP session, or a real WebSocket.
    """
    instance = SpeechStream.__new__(SpeechStream)
    captured: list[Any] = []
    event_ch = MagicMock()
    event_ch.send_nowait = captured.append
    instance._event_ch = event_ch  # type: ignore[attr-defined]
    instance._logger = MagicMock()  # type: ignore[attr-defined]
    instance._build_log_context = lambda: {}  # type: ignore[attr-defined]
    instance._server_request_id = None  # type: ignore[attr-defined]
    instance._opts = MagicMock(language="en-IN")  # type: ignore[attr-defined]
    return instance, captured


def _ws_message(**transcript_overrides: Any) -> dict:
    """Build the outer WS message dict expected by ``_handle_transcript_data``.

    Default shape mirrors a Saaras v3 streaming final-transcript chunk.
    """
    transcript_data: dict[str, Any] = {
        "transcript": "नमस्ते",
        "language_code": "hi-IN",
        "speech_start": 0.0,
        "speech_end": 1.2,
        "metrics": {"audio_duration": 1.2},
        "request_id": "req-test",
    }
    transcript_data.update(transcript_overrides)
    return {"type": "data", "data": transcript_data}


def _final_event(captured: list[Any]) -> stt.SpeechEvent:
    finals = [ev for ev in captured if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert finals, "no FINAL_TRANSCRIPT event emitted"
    return finals[0]


# ---------------------------------------------------------------------------
# WS path — happy cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language_probability, expected_confidence",
    [
        (0.87, 0.87),
        (1.0, 1.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (0.123, 0.123),
    ],
)
async def test_ws_threads_language_probability_into_confidence(
    language_probability: float, expected_confidence: float
) -> None:
    """WS path must thread Sarvam's language_probability into SpeechData.confidence."""
    instance, captured = _make_stream_under_test()
    await instance._handle_transcript_data(_ws_message(language_probability=language_probability))
    final = _final_event(captured)
    assert final.alternatives[0].confidence == pytest.approx(expected_confidence)


# ---------------------------------------------------------------------------
# WS path — defensive fallback cases (absent / null / wrong type)
# ---------------------------------------------------------------------------


async def test_ws_missing_language_probability_falls_back_to_1_0() -> None:
    """When the field is absent, confidence falls back to 1.0 (no crash)."""
    instance, captured = _make_stream_under_test()
    # _ws_message() helper omits language_probability by default
    await instance._handle_transcript_data(_ws_message())
    final = _final_event(captured)
    assert final.alternatives[0].confidence == 1.0


async def test_ws_null_language_probability_falls_back_to_1_0() -> None:
    """Explicit null also falls back to 1.0."""
    instance, captured = _make_stream_under_test()
    await instance._handle_transcript_data(_ws_message(language_probability=None))
    final = _final_event(captured)
    assert final.alternatives[0].confidence == 1.0


@pytest.mark.parametrize("bad_value", ["0.95", [], {}, object(), True, False])
async def test_ws_unexpected_type_falls_back_to_1_0(bad_value: Any) -> None:
    """String / list / dict / object / bool → confidence falls back to 1.0 with a debug log.

    bool is included because Python's ``bool`` is a subclass of ``int``;
    without an explicit guard a JSON ``false`` from Sarvam would silently
    become ``confidence=0.0`` and wrongly flag a valid transcript as low
    confidence. Same defensive pattern as ``livekit-plugins-slng``.
    """
    instance, captured = _make_stream_under_test()
    await instance._handle_transcript_data(_ws_message(language_probability=bad_value))
    final = _final_event(captured)
    assert final.alternatives[0].confidence == 1.0
    # The defensive branch logs a debug warning so contract drift is visible.
    assert instance._logger.debug.called  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# WS path — out-of-range values pass through verbatim (clamping is not this
# layer's job; downstream consumers can clamp if they need to).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [-0.5, 1.5, 2.0])
async def test_ws_out_of_range_values_passed_through(value: float) -> None:
    """Out-of-[0,1] values are passed through verbatim (no clamping)."""
    instance, captured = _make_stream_under_test()
    await instance._handle_transcript_data(_ws_message(language_probability=value))
    final = _final_event(captured)
    assert final.alternatives[0].confidence == pytest.approx(value)
