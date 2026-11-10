"""An acknowledged Whisper flush can take longer than the former idle poll.

The common STT contracts cover final audio, ignored end_audio, missing completion,
and remote close through the same provider-owned Baseten protocol.
"""

import asyncio
from dataclasses import replace

import pytest

from livekit.agents import APIConnectOptions, stt
from livekit.plugins.baseten import STT
from tests.stt_harness.discovery import provider_by_name
from tests.stt_harness.drivers import make_audio_segments, open_stream, push_audio
from tests.stt_harness.server import STTServer

from .stt_reconnect import ReconnectClock

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent, pytest.mark.usefixtures("job_process")]


async def test_slow_flush_still_delivers_the_final_transcript(monkeypatch):
    provider = replace(provider_by_name("baseten"), final_response_timeout=7)
    server = STTServer(provider, segments=make_audio_segments(provider))
    server.release_results.clear()
    async with open_stream(provider, monkeypatch, server=server) as (
        stream,
        collector,
        _,
        events,
        errors,
    ):
        for segment in make_audio_segments(provider):
            push_audio(stream, provider, segment)
        stream.end_input()
        await asyncio.wait_for(server.final_requested.wait(), 1)
        await asyncio.sleep(5.5)
        assert not collector.done()
        server.release_results.set()
        await asyncio.wait_for(collector, 1)
        assert (
            " ".join(
                event.alternatives[0].text
                for event in events
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            )
            == "first tail"
        )
        assert not errors
        assert bytes(server.received_audio) == server.expected_audio


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
