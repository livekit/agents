"""Hume TTS: utterance options whose legal value is falsy.

``trailing_silence`` is documented as a per-utterance silence duration in seconds,
0 to 5 with a 0.35 s default, so ``0.0`` is a valid request for "append no silence".
The plugin adds the field to the utterance behind a truthiness check, which drops
that request and lets the server default win: the caller asks for no trailing
silence and still gets 350 ms of it after every utterance.

These tests serve the request from a local aiohttp app, so they exercise the real
serializer path without reaching the Hume API.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from livekit.agents import APIConnectOptions
from livekit.plugins.hume import TTS, AudioFormat
from livekit.plugins.hume.tts import STREAM_PATH

pytestmark = pytest.mark.unit

# 2400 mono 16-bit samples, i.e. 50 ms at the plugin's 48 kHz output rate
PCM = b"\x01\x00" * 2400


@asynccontextmanager
async def _local_api(payload: dict[str, Any]) -> AsyncIterator[str]:
    """Serve Hume's streaming endpoint locally and record the request body."""

    async def handler(request: web.Request) -> web.Response:
        payload.update(await request.json())
        chunk = json.dumps({"type": "audio_chunk", "audio": base64.b64encode(PCM).decode()})
        return web.Response(body=(chunk + "\n").encode(), content_type="application/x-ndjson")

    app = web.Application()
    app.router.add_post(STREAM_PATH, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}"
    finally:
        await runner.cleanup()


async def _utterance(**options: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    async with _local_api(payload) as base_url:
        async with aiohttp.ClientSession() as session:
            client = TTS(
                api_key="test-key",
                base_url=base_url,
                audio_format=AudioFormat.pcm,
                http_session=session,
                **options,
            )
            stream = client.synthesize(
                "hello", conn_options=APIConnectOptions(max_retry=0, timeout=5)
            )
            try:
                frames = [event.frame async for event in stream]
            finally:
                await stream.aclose()
                await client.aclose()

    # audio came back, so the request reached the serializer
    assert frames
    assert {frame.sample_rate for frame in frames} == {48000}
    utterances = payload["utterances"]
    assert len(utterances) == 1
    return utterances[0]


async def test_positive_trailing_silence_is_sent() -> None:
    utterance = await _utterance(trailing_silence=1.5)
    assert utterance["trailing_silence"] == 1.5


async def test_zero_trailing_silence_is_sent() -> None:
    """0.0 is inside the documented range and means "no silence after the utterance"."""
    utterance = await _utterance(trailing_silence=0.0)
    assert utterance.get("trailing_silence") == 0.0


async def test_unset_trailing_silence_is_left_to_the_server() -> None:
    utterance = await _utterance()
    assert "trailing_silence" not in utterance
