"""Inworld plugin: streaming STT frame validation and HTTP TTS usage billing."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

from livekit.agents import APIConnectionError, APIConnectOptions, http
from livekit.plugins import inworld


@pytest.mark.parametrize("body", ["{", "[]", '{"result":{"transcription":{"transcript":42}}}'])
async def test_malformed_stream_frame_is_terminal_provider_error(body: str) -> None:
    """A malformed transcription frame fails the stream for good, caused by its validation."""

    async def recognize(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.receive()
        await ws.send_str(body)
        async for _ in ws:
            pass
        return ws

    app = web.Application()
    app.router.add_get("/stt/v1/transcribe:streamBidirectional", recognize)
    session = http.HttpSession()
    async with TestServer(app) as server:
        model = inworld.STT(
            api_key="test", base_url=str(server.make_url("/")), http_session=session
        )
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            with pytest.raises(APIConnectionError) as failure:
                await asyncio.wait_for(anext(stream), 0.5)
            assert not failure.value.retryable
            assert isinstance(failure.value.__cause__, ValidationError)
        await model.aclose()
    await session.aclose()


async def test_http_stream_bills_the_request_count_once() -> None:
    """Only the first streamed message's character count is billed, once per request."""
    audio = base64.b64encode(bytes(4800)).decode()

    async def voice_stream(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse()
        await response.prepare(request)
        for characters in (11, 0, 0):
            usage = {"processedCharactersCount": characters, "modelId": "inworld-tts-2"}
            line = {"result": {"audioContent": audio, "usage": usage}}
            await response.write(json.dumps(line).encode() + b"\n")
        return response

    app = web.Application()
    app.router.add_post("/tts/v1/voice:stream", voice_stream)
    async with TestServer(app) as server, http.open():
        tts = inworld.TTS(
            api_key="key", model="inworld-tts-2", encoding="PCM", base_url=str(server.make_url("/"))
        )

        frames = [audio async for audio in tts.synthesize("Hello world")]
        await tts.aclose()

    assert [f.characters_count for f in frames if f.characters_count is not None] == [11]
    assert frames[-1].is_final and frames[-1].characters_count == 11
