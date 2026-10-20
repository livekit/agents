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
def test_endpoint_uses_words_instead_of_billed_duration(end_time):
    stream = object.__new__(LegacyRecognizeStream)
    stream._speaking = False
    stream._language = None
    stream._request_id = "test"
    stream._speech_duration = 0.0
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append, closed=False)
    for turn in range(2):
        message = {"type": "transcript", "is_final": True, "text": "hello", "duration": 100.0}
        if end_time is not None:
            message["words"] = [
                {"word": "hello", "start": 0.1 + turn, "end": end_time + turn if end_time else 0.0}
            ]
        stream._process_stream_event(message)
        assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
        if end_time:
            assert events[-1].alternatives[0].end_time == pytest.approx(7.4 + turn)
        else:
            assert events[-1].alternatives == []
            assert events[-2].alternatives[0].end_time == 0.0


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.cartesia.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", model="ink-whisper", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()
