"""Public STT client driver and local input fixtures; SDK networking stays real."""

import asyncio
from contextlib import asynccontextmanager

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils, vad
from tests.fake_stt import FakeSTT

from .server import Segment, STTServer


class NoSpeechVAD(vad.VAD):
    def __init__(self):
        super().__init__(capabilities=vad.VADCapabilities(update_interval=0.1))

    def stream(self):
        return NoSpeechVADStream(self)


class NoSpeechVADStream(vad.VADStream):
    async def _main_task(self):
        async for _ in self._input_ch:
            pass


def make_audio_segments(provider):
    # Both turns satisfy provider minimum durations; the last frame stays short.
    return (
        Segment(b"\x00\x10" * (provider.sample_rate * 2 // 5), "first"),
        Segment(b"\x00\x20" * (provider.sample_rate * 3 // 10 + 317), "tail"),
    )


def push_audio(stream, provider, segment):
    chunk_size = provider.sample_rate // 10 * 2
    for offset in range(0, len(segment.audio), chunk_size):
        chunk = segment.audio[offset : offset + chunk_size]
        stream.push_frame(rtc.AudioFrame(chunk, provider.sample_rate, 1, len(chunk) // 2))


@asynccontextmanager
async def open_stream(provider, monkeypatch, *, server=None, max_retry=0, fallback=False):
    server = server or STTServer(provider, segments=make_audio_segments(provider))
    errors = []
    events = []
    async with server:
        primary = provider.create_stt(server, monkeypatch)
        primary.on("error", errors.append)
        instance = primary
        backup = None
        if fallback:
            backup = FakeSTT(fake_transcript="backup")
            instance = stt.FallbackAdapter([primary, backup], max_retry_per_stt=1, retry_interval=0)
        stream = instance.stream(
            conn_options=APIConnectOptions(
                max_retry=max_retry,
                retry_interval=0,
                timeout=2,
                final_response_timeout=provider.final_response_timeout,
                liveness_timeout=0.4,
            )
        )

        collector = asyncio.create_task(collect_events(stream, events))
        try:
            yield stream, collector, server, events, errors
        finally:
            # Abort may discard audio that the scenario has not pushed yet.
            server.allow_partial_input = True
            await stream.aclose()
            await utils.aio.gracefully_cancel(collector)
            await instance.aclose()
            if fallback:
                await primary.aclose()
                await backup.aclose()
            for connection in server.connections:
                await asyncio.wait_for(connection.disconnected.wait(), 2)


async def collect_events(stream, events):
    """Observe terminal errors through the public iterator, including after the last event."""
    async for event in stream:
        events.append(event)
