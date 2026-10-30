"""An acknowledged Whisper flush can take longer than the former idle poll.

The common STT contracts cover final audio, ignored end_audio, missing completion,
and remote close through the same provider-owned Baseten protocol.
"""

import asyncio
from dataclasses import replace

import pytest

from livekit.agents import stt
from tests.stt_harness.discovery import provider_by_name
from tests.stt_harness.drivers import make_audio_segments, open_stream, push_audio
from tests.stt_harness.server import STTServer

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
