from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import stt
from livekit.plugins import sarvam
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
    instance._input_started_wall = None  # type: ignore[attr-defined]
    instance._input_start_offset = 0.0  # type: ignore[attr-defined]
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


def _input_started(instance: SpeechStream, seconds_ago: float, *, offset: float = 0.0) -> None:
    # what push_frame stamps on the first frame
    instance._input_started_wall = time.time() - seconds_ago  # type: ignore[attr-defined]
    instance._input_start_offset = offset  # type: ignore[attr-defined]


def _frame() -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=b"\x00\x00" * 320, sample_rate=16000, num_channels=1, samples_per_channel=320
    )


async def _final_from(stream: SpeechStream) -> stt.SpeechEvent:
    while True:
        ev = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            return ev


@pytest.fixture
async def stalled_stream(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[SpeechStream]:
    """A real stream whose connection never sends: no audio reaches Sarvam."""

    async def _stalled_connection(self: SpeechStream) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(SpeechStream, "_run", _stalled_connection)
    stream = sarvam.STT(api_key="sk_test", language="hi-IN").stream()
    yield stream
    await stream.aclose()


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


async def test_final_end_time_falls_back_to_end_speech_receipt_on_input_clock() -> None:
    # Sarvam streaming exposes no word timestamps and a null speech_end, so a final
    # that follows END_SPEECH takes the END_SPEECH receipt on the input clock: the
    # instant EOS carries as speech_end_time, as an offset from the first input frame.
    instance, captured = _make_stream_under_test()
    _input_started(instance, 3.2)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.0, speech_end=0.0))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    end = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)[0]
    assert final.alternatives[0].start_time == 0.0
    assert final.alternatives[0].end_time == pytest.approx(3.2, abs=0.05)
    assert end.speech_end_time is not None
    assert final.alternatives[0].end_time == pytest.approx(
        end.speech_end_time - instance._input_started_wall  # type: ignore[attr-defined]
    )


async def test_late_final_after_eos_fallback_keeps_the_end_speech_position() -> None:
    # The final missed the EOS grace period, so it reaches the pipeline after
    # END_OF_SPEECH. Its end_time is what keeps the pipeline's speech-end anchor
    # (with no local VAD a final without one re-anchors the turn at its arrival,
    # and transcription_delay collapses to ~0).
    instance, captured = _make_stream_under_test()
    _input_started(instance, 3.2)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await asyncio.sleep(0.2)  # EOS fallback fires; the final lands 0.2 s after END_SPEECH
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
    assert final.alternatives[0].end_time == pytest.approx(3.2, abs=0.05)


async def test_speech_end_fallback_includes_start_time_offset() -> None:
    # Same stream timeline as the provider times (see test_final_timing_includes_start_time_offset).
    instance, captured = _make_stream_under_test()
    _input_started(instance, 3.2, offset=2.0)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message())

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].end_time == pytest.approx(5.2, abs=0.05)


async def test_provider_speech_end_wins_over_the_fallback() -> None:
    instance, captured = _make_stream_under_test()
    _input_started(instance, 3.2)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH"))
    await instance._handle_transcript_data(_ws_message(speech_start=0.5, speech_end=1.7))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    assert final.alternatives[0].end_time == pytest.approx(1.7)


async def test_input_ahead_of_real_time_does_not_push_the_speech_end_ahead(
    stalled_stream: SpeechStream,
) -> None:
    # 5 s of audio pushed at once (a pre-connect buffer, a flush) while none of it
    # has reached Sarvam: an audio count would put the speech end ~5 s into the
    # future, where the pipeline clamps it to the final's arrival (~0 delay).
    stream = stalled_stream
    for _ in range(250):
        stream.push_frame(_frame())
    await stream._handle_events(_event("START_SPEECH"))
    await stream._handle_events(_event("END_SPEECH"))
    await stream._handle_transcript_data(_ws_message())

    final = await _final_from(stream)
    assert final.alternatives[0].end_time < 1.0
    expected = (
        stream._utterance_speech_end_wall  # type: ignore[attr-defined]
        - stream._input_started_wall  # type: ignore[attr-defined]
        + stream._input_start_offset  # type: ignore[attr-defined]
    )
    assert final.alternatives[0].end_time == pytest.approx(expected)


async def test_retry_offset_does_not_shift_the_speech_end(stalled_stream: SpeechStream) -> None:
    # The base retry loop advances start_time_offset for provider-relative times; the
    # input clock started with the first frame and must not move with it.
    stream = stalled_stream
    stream.push_frame(_frame())
    initial_offset = stream.start_time_offset
    stream.start_time_offset = initial_offset + 3.0  # what a retry would add

    await stream._handle_events(_event("START_SPEECH"))
    await stream._handle_events(_event("END_SPEECH"))
    await stream._handle_transcript_data(_ws_message())

    final = await _final_from(stream)
    assert final.alternatives[0].end_time < 1.0
    expected = (
        stream._utterance_speech_end_wall  # type: ignore[attr-defined]
        - stream._input_started_wall  # type: ignore[attr-defined]
        + initial_offset
    )
    assert final.alternatives[0].end_time == pytest.approx(expected)


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
    # `occured_at` is a wall-clock epoch from the server's clock, so it is never
    # used: EOS stays bare and the final's end_time falls back to the END_SPEECH
    # receipt on the input clock, even when END_SPEECH arrives before the final.
    instance, captured = _make_stream_under_test()
    _input_started(instance, 2.0)

    await instance._handle_events(_event("START_SPEECH"))
    await instance._handle_events(_event("END_SPEECH", occured_at=1_700_000_000.0))
    await instance._handle_transcript_data(_ws_message(speech_start=0.0, speech_end=0.0))

    final = _events_of_type(captured, stt.SpeechEventType.FINAL_TRANSCRIPT)[0]
    end = _events_of_type(captured, stt.SpeechEventType.END_OF_SPEECH)[0]
    assert final.alternatives[0].end_time == pytest.approx(2.0, abs=0.05)
    assert end.alternatives == []
    # The wall-clock anchor still rides on the EOS event itself.
    assert end.speech_end_time is not None
