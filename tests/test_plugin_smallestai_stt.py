from __future__ import annotations

import pytest

pytestmark = pytest.mark.plugin("smallestai")


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.smallestai.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(1.0, 1.0, (1.0, 1.0)), (1.0, 1.003, (6.0, 6.003)), (0.997, 0.999, (0.997, 0.999))],
)
async def test_utterance_metadata_stays_on_the_provider_clock(start, end, expected, monkeypatch):
    import time
    from types import SimpleNamespace

    from livekit import rtc
    from livekit.plugins.smallestai.stt import SpeechStream

    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    # the input clock measures elapsed time on the monotonic clock
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    stream = object.__new__(SpeechStream)
    stream._init_channels(translate_timestamps=True)
    stream.start_time_offset = 0.0
    stream._opts = SimpleNamespace(language="en", diarize=False, vad_events=True)
    stream._session_id = ""
    stream._speaking = True
    frame = rtc.AudioFrame.create(16000, 1, 16000)
    for captured_end in (1001.0, 1007.0):
        now = captured_end
        stream._input_ch.send_nowait(frame)
        stream._input_ch.recv_nowait()

    utterances = [
        {"start": start, "end": end, "speaker": 1, "text": "hello"},
        {"text": "no timing"},
    ]
    stream._process_stream_event(
        {
            "type": "transcription",
            "transcript": "hello",
            "is_final": True,
            "words": [{"word": "hello", "start": start, "end": end}],
            "utterances": [dict(u) for u in utterances],
        }
    )
    placed = stream._event_ch.recv_nowait().alternatives[0]
    assert (placed.start_time, placed.end_time) == pytest.approx(expected)
    assert placed.metadata == {"utterances": utterances}
