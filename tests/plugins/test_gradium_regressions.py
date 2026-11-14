"""Gradium synthesis failure and voice selection contracts."""

import asyncio
import base64
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http, tts
from livekit.plugins.gradium import TTS


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["error", "close"])
async def test_synthesis_surfaces_provider_failure(streaming: bool, failure: str) -> None:
    """Synthesis preserves server error details and abnormal close codes. https://github.com/livekit/agents/pull/7672"""

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if json.loads(msg.data)["type"] != "end_of_stream":
                continue
            await ws.send_json(
                {"type": "audio", "audio": base64.b64encode(b"\x01\x00" * 4800).decode()}
            )
            if failure == "error":
                await ws.send_json({"type": "error", "message": "quota exhausted", "code": 429})
                await ws.send_json({"type": "end_of_stream"})
            else:
                await ws.close(code=1011, message=b"provider failed")
        return ws

    app = web.Application()
    app.router.add_get("/tts", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key", model_endpoint=str(server.make_url("/tts")), http_session=session
        )
        stream: tts.SynthesizeStream | tts.ChunkedStream
        if streaming:
            stream = provider.stream(conn_options=APIConnectOptions(max_retry=0))
            stream.push_text("Hello.")
            stream.end_input()
        else:
            stream = provider.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0))
        async with stream:
            with pytest.raises(APIError) as error:
                async with asyncio.timeout(2):
                    _ = [ev async for ev in stream]
            if failure == "close":
                assert error.value.kind == "server"  # close code 1011
            else:
                assert isinstance(error.value, APIStatusError)
                assert error.value.status_code == 429
                assert error.value.detail == "quota exhausted"


@pytest.mark.parametrize("updated", [False, True])
async def test_explicit_voice_does_not_send_competing_default(updated: bool) -> None:
    """An explicit speaker never competes with the default voice ID. https://github.com/livekit/agents/pull/7813"""
    setups = []

    async def speak(request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            packet = json.loads(msg.data)
            if packet["type"] == "setup":
                setups.append(packet)
            if packet["type"] == "end_of_stream":
                await ws.send_json(
                    {"type": "audio", "audio": base64.b64encode(bytes(9600)).decode()}
                )
                await ws.send_json({"type": "end_of_stream"})
        return ws

    app = web.Application()
    app.router.add_get("/tts", speak)
    async with TestServer(app) as server, http.open() as session:
        provider = TTS(
            api_key="key",
            voice=None if updated else "speaker",
            model_endpoint=str(server.make_url("/tts")),
            http_session=session,
        )
        if updated:
            provider.update_options(voice="speaker")
        async with provider.synthesize(
            "Hello.", conn_options=APIConnectOptions(max_retry=0)
        ) as stream:
            assert [ev async for ev in stream]
    assert setups[0]["voice"] == "speaker"
    assert "voice_id" not in setups[0]
