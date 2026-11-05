"""Telnyx malformed audio fails at the provider boundary without retry."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectionError, APIConnectOptions, utils
from livekit.plugins.telnyx import TTS

pytestmark = pytest.mark.plugin("telnyx")


async def test_invalid_audio_field_is_a_terminal_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoder_type = utils.media.AudioStreamDecoder

    def decoder(
        *, sample_rate: int, num_channels: int, format: str
    ) -> utils.media.AudioStreamDecoder:
        # No audio reaches decoding; WAV keeps this wire test independent of optional FFmpeg.
        return decoder_type(sample_rate=sample_rate, num_channels=num_channels, format="audio/wav")

    monkeypatch.setattr(utils.media, "AudioStreamDecoder", decoder)

    async def handle(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"audio": ["invalid"]})
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/tts", handle)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        provider = TTS(api_key="test", base_url=str(server.make_url("/tts")), http_session=session)
        try:
            async with asyncio.timeout(3):
                with pytest.raises(APIConnectionError) as caught:
                    async with provider.stream(
                        conn_options=APIConnectOptions(max_retry=0)
                    ) as stream:
                        stream.push_text("hello")
                        stream.end_input()
                        async for _ in stream:
                            pass
                assert caught.value.retryable is False
                assert isinstance(caught.value.__cause__, ValidationError)
        finally:
            await provider.aclose()
