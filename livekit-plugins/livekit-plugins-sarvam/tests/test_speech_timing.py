from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins.sarvam.stt import SpeechStream

pytestmark = pytest.mark.unit


def _make_stream_under_test(
    *, eos_fallback_timeout: float = 0.01
) -> tuple[SpeechStream, list[Any]]:
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
    instance._start_time_offset = 0.0  # type: ignore[attr-defined]
    instance._utterance_speech_start_wall = None  # type: ignore[attr-defined]
    instance._utterance_speech_end_wall = None  # type: ignore[attr-defined]
    instance._audio_sent_duration = 0.0  # type: ignore[attr-defined]
    instance._utterance_speech_end_audio_pos = None  # type: ignore[attr-defined]
    instance._pending_final_data = None  # type: ignore[attr-defined]
    instance._pending_eos = False  # type: ignore[attr-defined]
    instance._eos_fallback_task = None  # type: ignore[attr-defined]
    instance._eos_fallback_timeout = eos_fallback_timeout  # type: ignore[attr-defined]
    instance._final_received_for_utterance = False  # type: ignore[attr-defined]
    instance._eos_emitted_for_utterance = False  # type: ignore[attr-defined]
    return instance, captured


def _event(signal_type: str, *, occured_at: float | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"signal_type": signal_type, "request_id": "req-test"}
    if occured_at is not None:
        data["occured_at"] = occured_at
    return {"type": "events", "data": data}


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


async def test_end_speech_emits_bare_end_of_speech() -> None:
    # Like every other STT plugin, END_OF_SPEECH carries no alternatives/timing.
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.05)

    end_events = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)
    assert len(end_events) == 1
    assert end_events[0].alternatives == []
    assert end_events[0].request_id == "req-session"


async def test_end_of_speech_carries_speech_end_wall_clock() -> None:
    # The voice pipeline anchors `_last_speaking_time` from END_OF_SPEECH's
    # speech_end_time (stt turn-detection mode), which makes transcription_delay
    # measure Sarvam VAD speech end -> final transcript received. The anchor is
    # the wall-clock moment the plugin received END_SPEECH, not the later moment
    # the EOS event is actually emitted.
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.05)

    end_events = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)
    assert len(end_events) == 1
    speech_end = end_events[0].speech_end_time
    assert speech_end is not None
    assert speech_end <= time.time()
    # The anchor is captured at END_SPEECH receipt, so it must not be later than
    # the moment the EOS event was created by the fallback task.
    assert speech_end <= end_events[0].created_at


async def test_speech_end_wall_clock_resets_on_new_utterance() -> None:
    # A stale anchor from the previous utterance must not leak into the next
    # turn: _reset_utterance_state clears it on START_SPEECH.
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.05)  # let the EOS fallback task fire
    first_end = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)[0]
    assert first_end.speech_end_time is not None

    await instance._handle_events(_event("START_SPEECH"))
    assert instance._utterance_speech_end_wall is None  # type: ignore[attr-defined]
    assert instance._utterance_speech_end_audio_pos is None  # type: ignore[attr-defined]


async def test_final_end_time_falls_back_to_audio_sent_at_end_speech() -> None:
    # Sarvam streaming exposes no word timestamps and a null speech_end, so a final
    # that follows END_SPEECH takes the audio sent when END_SPEECH arrived, not the
    # audio sent by the time the final lands.
    instance, captured = _make_stream_under_test()
    instance._audio_sent_duration = 3.2  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    instance._audio_sent_duration = 3.4  # type: ignore[attr-defined]
    await instance._handle_transcript_data(_ws_message(speech_start=0.0, speech_end=0.0))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].start_time == 0.0
    assert final.alternatives[0].end_time == pytest.approx(3.2)


async def test_late_final_after_eos_fallback_keeps_the_end_speech_position() -> None:
    # The final missed the EOS grace period, so it reaches the pipeline after
    # END_OF_SPEECH. Its end_time is what keeps the pipeline's speech-end anchor
    # (with no local VAD a final without one re-anchors the turn at its arrival,
    # and transcription_delay collapses to ~0).
    instance, captured = _make_stream_under_test()
    instance._audio_sent_duration = 3.2  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.05)  # EOS fallback fires
    instance._audio_sent_duration = 4.6  # type: ignore[attr-defined]  # audio kept flowing
    await instance._handle_transcript_data(_ws_message())

    event_order = [
        ev.type
        for ev in captured
        if ev.type in {stt.SpeechEventType.FINAL_TRANSCRIPT, stt.SpeechEventType.END_OF_SPEECH}
    ]
    assert event_order == [
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
    ]
    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].end_time == pytest.approx(3.2)


async def test_audio_position_fallback_includes_start_time_offset() -> None:
    # Same stream timeline as the provider times (see test_final_timing_includes_start_time_offset).
    instance, captured = _make_stream_under_test()
    instance._start_time_offset = 2.0  # type: ignore[attr-defined]
    instance._audio_sent_duration = 3.2  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message())

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].end_time == pytest.approx(5.2)


async def test_provider_speech_end_wins_over_the_audio_position() -> None:
    instance, captured = _make_stream_under_test()
    instance._audio_sent_duration = 3.2  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.5, speech_end=1.7))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].end_time == pytest.approx(1.7)


async def test_process_audio_counts_only_audio_actually_sent() -> None:
    instance, _ = _make_stream_under_test()
    instance._chunk_size = 800  # type: ignore[attr-defined]  # 50 ms at 16 kHz
    instance._audio_encoding = "audio/wav"  # type: ignore[attr-defined]
    instance._end_of_stream_msg = "{}"  # type: ignore[attr-defined]
    ws = SimpleNamespace(send_str=AsyncMock())

    async def _frames() -> Any:
        for _ in range(2):
            yield rtc.AudioFrame(
                data=b"\x00\x00" * 1000, sample_rate=16000, num_channels=1, samples_per_channel=1000
            )

    instance._input_ch = _frames()  # type: ignore[attr-defined]
    await instance._process_audio(ws)  # type: ignore[arg-type]

    # 2000 samples pushed: two full 800-sample chunks went out, 400 are still buffered
    assert ws.send_str.await_count == 2
    assert instance._audio_sent_duration == pytest.approx(0.1)  # type: ignore[attr-defined]


async def test_final_end_time_uses_speech_end_when_present() -> None:
    # If Sarvam ever populates the documented speech_start/speech_end fields over
    # the socket, they flow through to the final transcript's timing.
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.5, speech_end=1.7))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].start_time == pytest.approx(0.5)
    assert final.alternatives[0].end_time == pytest.approx(1.7)


async def test_final_timing_includes_start_time_offset() -> None:
    # Provider times are stream-relative; the base class accrues start_time_offset
    # across reconnects, so present speech_start/speech_end must be shifted by it.
    instance, captured = _make_stream_under_test()
    instance._start_time_offset = 2.0  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.5, speech_end=1.7))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].start_time == pytest.approx(2.5)
    assert final.alternatives[0].end_time == pytest.approx(3.7)


async def test_commit_order_final_before_eos() -> None:
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
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
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
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


async def test_late_transcript_after_eos_fallback_is_emitted_after_eos() -> None:
    instance, captured = _make_stream_under_test()

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.05)
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
        stt.SpeechEventType.END_OF_SPEECH,
        stt.SpeechEventType.FINAL_TRANSCRIPT,
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


async def test_aclose_cancels_pending_eos_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop_parent_aclose(self: object) -> None:
        return None

    monkeypatch.setattr(stt.SpeechStream, "aclose", _noop_parent_aclose)
    instance, captured = _make_stream_under_test(eos_fallback_timeout=0.05)
    instance._connection_lock = asyncio.Lock()  # type: ignore[attr-defined]
    instance._audio_task = None  # type: ignore[attr-defined]
    instance._message_task = None  # type: ignore[attr-defined]
    instance._ws = None  # type: ignore[attr-defined]
    instance._session = SimpleNamespace(closed=True)  # type: ignore[attr-defined]
    instance._client_request_id = "client"  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    fallback_task = instance._eos_fallback_task  # type: ignore[attr-defined]

    assert fallback_task is not None
    await instance.aclose()
    await asyncio.sleep(0.1)

    assert fallback_task.cancelled()
    assert not _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)


async def test_occured_at_is_ignored() -> None:
    # `occured_at` is a wall-clock epoch, not an audio-relative offset, so it is
    # never used: EOS stays bare and the final's end_time falls back to the audio
    # sent at END_SPEECH, even when END_SPEECH arrives before the final transcript.
    instance, captured = _make_stream_under_test()
    instance._audio_sent_duration = 2.0  # type: ignore[attr-defined]

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH", occured_at=1_700_000_000.0))
    await instance._handle_transcript_data(_ws_message(speech_start=0.0, speech_end=0.0))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    end = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)[0]
    assert final.alternatives[0].end_time == pytest.approx(2.0)
    assert end.alternatives == []
    # The wall-clock anchor still rides on the EOS event itself.
    assert end.speech_end_time is not None
