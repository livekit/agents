"""Murf TTS against a local peer speaking Murf's stream-input wire."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, http
from livekit.plugins import murf


# Catches: an error the provider reports for a context being dropped, leaving the turn waiting
# for audio that never comes instead of failing with the provider's reason.
async def test_a_context_error_fails_the_stream_with_its_reason() -> None:
    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            packet = json.loads(message.data)
            if packet.get("text", "").strip():
                error = {"error": "voice not found", "error_code": "VOICE_NOT_FOUND"}
                await ws.send_json({"context_id": packet["context_id"], **error})
        return ws

    app = web.Application()
    app.router.add_get("/v1/speech/stream-input", peer)
    async with TestServer(app) as server, http.open():
        model = murf.TTS(api_key="key", base_url=str(server.make_url("")).rstrip("/"))
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_text("Hello there.")
            stream.end_input()
            with pytest.raises(APIError) as failure:
                async with asyncio.timeout(2):
                    _ = [audio async for audio in stream]
        await model.aclose()

    assert failure.value.detail == "voice not found"
