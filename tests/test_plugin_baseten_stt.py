from __future__ import annotations

import asyncio

import pytest

from livekit.agents import APIConnectOptions, stt
from livekit.plugins.baseten import STT

from .stt_reconnect import ReconnectClock

pytestmark = pytest.mark.plugin("baseten")


@pytest.mark.no_concurrent
async def test_internal_reconnect_reanchors_transcript_timing(monkeypatch):
    clock = ReconnectClock(monkeypatch)
    provider = STT(api_key="test-key", model_endpoint="wss://test.invalid/stt", http_session=clock)
    stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
    try:
        await clock.check(stream, lambda: stream.update_options(language="fr"))
        clock.sockets[-1].respond(
            {
                "is_final": True,
                "segments": [{"text": "hello", "start_time": 0.1, "end_time": 0.4}],
            }
        )
        async with asyncio.timeout(2):
            async for event in stream:
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    assert event.alternatives[0].end_time == pytest.approx(17.4)
                    break
            else:
                pytest.fail("No FINAL_TRANSCRIPT from the restarted stream")
    finally:
        await stream.aclose()
        await provider.aclose()
