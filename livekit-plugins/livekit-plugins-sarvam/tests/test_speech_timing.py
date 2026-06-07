from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit.agents import stt
from livekit.plugins.sarvam.stt import SpeechStream

pytestmark = pytest.mark.unit


def _make_stream_under_test(*, audio_position: float = 1.25) -> tuple[SpeechStream, list[Any]]:
    instance = SpeechStream.__new__(SpeechStream)
    captured: list[Any] = []
    event_ch = MagicMock()
    event_ch.send_nowait = captured.append

    instance._event_ch = event_ch  # type: ignore[attr-defined]
    instance._logger = MagicMock()  # type: ignore[attr-defined]
    instance._build_log_context = lambda: {}  # type: ignore[attr-defined]
    instance._server_request_id = "req-session"  # type: ignore[attr-defined]
    instance._opts = SimpleNamespace(language="en-IN", sample_rate=16000)  # type: ignore[attr-defined]
    instance._speaking = False  # type: ignore[attr-defined]
    instance._should_flush = False  # type: ignore[attr-defined]
    instance._audio_position = audio_position  # type: ignore[attr-defined]
    instance._utterance_start_audio_pos = 0.0  # type: ignore[attr-defined]
    instance._utterance_speech_end_audio_pos = None  # type: ignore[attr-defined]
    instance._utterance_speech_start_wall = None  # type: ignore[attr-defined]
    instance._utterance_speech_end_wall = None  # type: ignore[attr-defined]
    instance._pending_final_data = None  # type: ignore[attr-defined]
    instance._pending_eos = False  # type: ignore[attr-defined]
    instance._eos_fallback_task = None  # type: ignore[attr-defined]
    instance._final_received_for_utterance = False  # type: ignore[attr-defined]
    instance._eos_emitted_for_utterance = False  # type: ignore[attr-defined]
    return instance, captured


def _event(signal_type: str) -> dict[str, Any]:
    return {"type": "events", "data": {"signal_type": signal_type, "request_id": "req-test"}}


def _ws_message(**transcript_overrides: Any) -> dict[str, Any]:
    transcript_data: dict[str, Any] = {
        "transcript": "नमस्ते",
        "language_code": "hi-IN",
        "speech_start": 0.0,
        "speech_end": 0.0,
        "metrics": {"audio_duration": 1.2},
        "request_id": "req-test",
    }
    transcript_data.update(transcript_overrides)
    return {"type": "data", "data": transcript_data}


def _events_of_type(captured: list[Any], event_type: stt.SpeechEventType) -> list[stt.SpeechEvent]:
    return [ev for ev in captured if ev.type == event_type]


async def test_start_speech_sets_speech_start_time() -> None:
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))

    start_events = _events_of_type(captured, stt.SpeechEventType.START_OF_SPEECH)
    assert len(start_events) == 1
    assert start_events[0].speech_start_time is not None
    assert start_events[0].request_id == "req-session"


async def test_end_speech_emits_end_time_alternative() -> None:
    instance, captured = _make_stream_under_test(audio_position=1.4)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.15)

    end_events = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)
    assert len(end_events) == 1
    assert end_events[0].alternatives[0].end_time == pytest.approx(1.4)
    assert end_events[0].request_id == "req-session"


async def test_final_uses_fallback_when_api_timing_zero() -> None:
    instance, captured = _make_stream_under_test(audio_position=1.25)

    await instance._handle_events(_event("START_SPEECH"))
    instance._audio_position = 1.4  # type: ignore[attr-defined]
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.0, speech_end=0.0))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].start_time == pytest.approx(1.25)
    assert final.alternatives[0].end_time == pytest.approx(1.4)


async def test_commit_order_final_before_eos() -> None:
    instance, captured = _make_stream_under_test(audio_position=1.4)

    await instance._handle_events(_event("START_SPEECH"))
    instance._audio_position = 1.4  # type: ignore[attr-defined]
    await instance._handle_transcript_data(_ws_message())
    await instance._handle_events(_event("END_SPEECH"))

    event_order = [
        ev.type
        for ev in captured
        if ev.type
        in {
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.END_OF_SPEECH,
        }
    ]
    assert event_order == [
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_commit_order_final_before_eos_when_speech_end_arrives_first() -> None:
    instance, captured = _make_stream_under_test(audio_position=1.25)

    await instance._handle_events(_event("START_SPEECH"))
    instance._audio_position = 1.4  # type: ignore[attr-defined]
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message())

    event_order = [
        ev.type
        for ev in captured
        if ev.type
        in {
            stt.SpeechEventType.FINAL_TRANSCRIPT,
            stt.SpeechEventType.END_OF_SPEECH,
        }
    ]
    assert event_order == [
        stt.SpeechEventType.FINAL_TRANSCRIPT,
        stt.SpeechEventType.END_OF_SPEECH,
    ]


async def test_final_emits_immediately_without_speech_end() -> None:
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_transcript_data(_ws_message())

    final_events = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)
    assert len(final_events) == 1
    assert final_events[0].alternatives[0].end_time == 0.0


async def test_multiple_transcripts_emit_multiple_finals() -> None:
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_transcript_data(_ws_message(transcript="first"))
    await instance._handle_transcript_data(_ws_message(transcript="second"))

    final_events = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)
    assert [ev.alternatives[0].text for ev in final_events] == ["first", "second"]
