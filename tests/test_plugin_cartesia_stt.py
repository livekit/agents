from __future__ import annotations

from types import SimpleNamespace

import pytest

from livekit.agents import stt
from livekit.plugins.cartesia._recognize_streams.legacy_recognize_stream import (
    LegacyRecognizeStream,
)

pytestmark = pytest.mark.plugin("cartesia")


def test_auto_endpoint_does_not_use_billed_duration_as_speech_end():
    from livekit.plugins.cartesia._recognize_streams.auto_finalize_recognize_stream import (
        AutoFinalizeRecognizeStream,
    )

    stream = object.__new__(AutoFinalizeRecognizeStream)
    stream._speaking = False
    stream._language = "en"
    stream._request_id = "test"
    stream._speech_duration = 100.0
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append, closed=False)
    stream._process_stream_event({"type": "turn.start", "request_id": "test"})
    stream._process_stream_event({"type": "turn.end", "transcript": "hello", "request_id": "test"})
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    assert events[-1].alternatives == []
    assert events[-1].speech_end_time is None


@pytest.mark.parametrize("end_time", [0.4, 0.0, None])
def test_endpoint_uses_words_and_preserves_transcript_spans(end_time):
    stream = object.__new__(LegacyRecognizeStream)
    stream._speaking = False
    stream._language = None
    stream._request_id = "test"
    stream._speech_duration = 0.0
    stream._last_speech_end_time = 0.0
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append, closed=False)
    for turn in range(2):
        for is_final, start, end in [(False, 7.0, 7.5), (True, 7.5, 8.0)]:
            message = {
                "type": "transcript",
                "is_final": is_final,
                "text": "hello",
                "duration": 0.5,
            }
            if end_time is not None:
                message["words"] = [
                    {
                        "word": "hello",
                        "start": 0.1 + turn,
                        "end": end_time + turn if end_time else 0.0,
                    }
                ]
            stream._process_stream_event(message)
            transcript = events[-2] if is_final else events[-1]
            assert transcript.type == (
                stt.SpeechEventType.FINAL_TRANSCRIPT
                if is_final
                else stt.SpeechEventType.INTERIM_TRANSCRIPT
            )
            assert transcript.alternatives[0].start_time == pytest.approx(start + turn)
            assert transcript.alternatives[0].end_time == pytest.approx(end + turn)
        assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
        if end_time:
            assert events[-1].alternatives[0].start_time == pytest.approx(7.1 + turn)
            assert events[-1].alternatives[0].end_time == pytest.approx(7.4 + turn)
        else:
            assert events[-1].alternatives == []


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_transcript_and_audio_clock(monkeypatch):
    import asyncio

    from livekit.agents import APIConnectOptions
    from livekit.plugins.cartesia.stt import STT

    from .stt_reconnect import ReconnectClock, wait_until

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", model="ink-whisper", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.start_time_offset = 7.0
    try:
        for index, captured_at, expected_start in [(0, 1000.0, 7.0), (1, 1010.0, 17.0)]:
            if index:
                clock.now = captured_at
                stream.update_options(language="fr")
                await wait_until(lambda: len(clock.sockets) == 2)
            await clock.push(stream, captured_at=captured_at)
            assert stream.start_time_offset == pytest.approx(expected_start)
            clock.sockets[index].respond(
                {"type": "transcript", "is_final": True, "text": "hello", "duration": 0.5}
            )
            async with asyncio.timeout(2):
                async for event in stream:
                    if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                        assert event.alternatives[0].start_time == pytest.approx(expected_start)
                        assert event.alternatives[0].end_time == pytest.approx(expected_start + 0.5)
                        break
                else:
                    pytest.fail("stream ended without a final transcript")
    finally:
        await stream.aclose()
        await provider.aclose()
