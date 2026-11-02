"""Tests for the xAI STT plugin: half-open socket detection.

Runs a real `SpeechStream` (its real `_run` loop, driven by `_main_task`) against
fake sockets handed out by a fake aiohttp session, so the connect kwargs and the
reconnect behaviour are assertable without a network. Same shape as the Deepgram
tests added in #7206 and the Telnyx tests in #7359.
"""

from __future__ import annotations

import pytest

from livekit.agents import APIConnectOptions

pytestmark = pytest.mark.plugin("xai")

# Retry immediately so a reconnect shows up within the test timeout.
_FAST_RETRY = APIConnectOptions(max_retry=3, retry_interval=0.01, timeout=1.0)


@pytest.mark.parametrize(
    "endpoint_type",
    ["speech_final", "done", "empty_done", "wordless_done", "wordless_summary_done"],
)
@pytest.mark.parametrize("end_time", [0.4, 0.0])
@pytest.mark.parametrize("prior_timed_chunk", [False, True])
def test_endpoint_preserves_word_timing(endpoint_type, end_time, prior_timed_chunk):
    from types import SimpleNamespace

    from livekit.agents import stt
    from livekit.plugins.xai import stt as xai_stt

    stream = object.__new__(xai_stt.SpeechStream)
    stream._opts = SimpleNamespace(enable_diarization=False, enable_interim_results=True)
    stream._speaking = False
    stream._emitted_chunk_final = False
    stream._final_alternatives = []
    stream._input_ch = SimpleNamespace(start_time_offset=7.0)
    stream._request_id = "test"
    events = []
    stream._event_ch = SimpleNamespace(send_nowait=events.append)
    final = {
        "type": "transcript.partial",
        "text": "hello",
        "is_final": True,
        "words": [{"text": "hello", "start": 0.1, "end": end_time}],
    }
    if prior_timed_chunk:
        stream._process_stream_event(
            {**final, "text": "earlier", "words": [{"text": "earlier", "start": 0.0, "end": 0.2}]}
        )
    stream._process_stream_event(final)
    endpoint = dict(final)
    if endpoint_type == "speech_final":
        endpoint["speech_final"] = True
    else:
        endpoint["type"] = "transcript.done"
        if endpoint_type == "empty_done":
            endpoint["text"] = ""
            endpoint["words"] = []
        elif endpoint_type in ("wordless_done", "wordless_summary_done"):
            endpoint["words"] = []
            if endpoint_type == "wordless_summary_done":
                endpoint["text"] = "earlier hello"
    stream._process_stream_event(endpoint)
    assert events[-1].type == stt.SpeechEventType.END_OF_SPEECH
    if end_time:
        assert events[-1].alternatives[0].end_time == pytest.approx(7.4)
    else:
        assert events[-1].alternatives == []
    stream._process_stream_event({"type": "transcript.partial", "text": "next"})
    stream._process_stream_event({"type": "transcript.done"})
    assert events[-1].alternatives == []


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_audio_clock(monkeypatch):
    from livekit.agents import APIConnectOptions
    from livekit.plugins.xai.stt import STT

    from .stt_reconnect import ReconnectClock

    clock = ReconnectClock(monkeypatch, ready={"type": "transcript.created"})
    provider = STT(api_key="test-key", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
    finally:
        await stream.aclose()
        await provider.aclose()
