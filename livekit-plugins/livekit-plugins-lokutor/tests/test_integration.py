"""Live integration test against the real Lokutor API.

Skipped unless LOKUTOR_API_KEY is set, so the default `pytest` run stays
offline and hermetic. To run it:

    LOKUTOR_API_KEY="sk_your_key" pytest tests/test_integration.py -v

It exercises the full path — WebSocket connect, a request carrying the
canonical `language` field, binary audio frames, and the `EOS` end signal —
so it will catch a wire-protocol regression that the unit tests (which only
build the request dict) cannot.

Written as sync functions driving ``asyncio.run`` so the suite needs no
``pytest-asyncio`` plugin or event-loop configuration.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("LOKUTOR_API_KEY"),
    reason="LOKUTOR_API_KEY not set; skipping live Lokutor API test",
)


async def _stream_synthesis() -> tuple[int, int]:
    import aiohttp

    from livekit.plugins import lokutor

    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(voice="F1", language="en", steps=8, http_session=session)
        async with tts:
            stream = tts.stream()
            stream.push_text("Testing the Lokutor LiveKit integration end to end.")
            stream.end_input()

            total_bytes = 0
            frames = 0
            async for audio in stream:
                total_bytes += len(audio.frame.data)
                frames += 1
    return frames, total_bytes


async def _chunked_synthesis() -> int:
    import aiohttp

    from livekit.plugins import lokutor

    async with aiohttp.ClientSession() as session:
        tts = lokutor.TTS(voice="M1", language="en", steps=8, http_session=session)
        async with tts:
            total_bytes = 0
            async for audio in tts.synthesize("A short one-shot synthesis."):
                total_bytes += len(audio.frame.data)
    return total_bytes


def test_live_streaming_synthesis_returns_audio():
    frames, total_bytes = asyncio.run(_stream_synthesis())
    assert frames > 0, "no audio frames were emitted"
    assert total_bytes > 0, "audio frames were empty"


def test_live_chunked_synthesis_returns_audio():
    total_bytes = asyncio.run(_chunked_synthesis())
    assert total_bytes > 0, "chunked synthesis produced no audio"
