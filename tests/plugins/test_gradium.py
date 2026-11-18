"""Gradium plugin: streaming STT setup message, against a local peer."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit import rtc
from livekit.agents import APIConnectOptions, http, stt
from livekit.plugins.gradium import STT

# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


async def test_eof_preserves_all_text_chunks_without_a_vad_boundary() -> None:
    async def asr(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive()
        await socket.send_json({"type": "ready", "frame_size": 1920, "delay_in_tokens": 4})
        async for raw in socket:
            if json.loads(raw.data)["type"] == "end_of_stream":
                for text in ("hello", "world"):
                    await socket.send_json({"type": "text", "text": text, "start_s": 0.0})
                await socket.send_json({"type": "end_of_stream"})
        return socket

    app = web.Application()
    app.router.add_get("/asr", asr)
    async with TestServer(app) as server, http.open():
        model = STT(api_key="key", model_endpoint=str(server.make_url("/asr").with_scheme("ws")))
        async with model.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
            stream.push_frame(rtc.AudioFrame(bytes(3840), 24000, 1, 1920))
            stream.end_input()
            finals = [
                event.alternatives[0].text
                async for event in stream
                if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
            ]
        await model.aclose()
    assert finals == ["hello world"]


async def test_speech_stream_sends_language_in_json_config() -> None:
    """The setup message carries the language and temperature. https://github.com/livekit/agents/pull/5878"""
    setups: asyncio.Queue[Wire] = asyncio.Queue()

    async def asr(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        setups.put_nowait(json.loads((await socket.receive()).data))
        async for _ in socket:
            pass
        return socket

    app = web.Application()
    app.router.add_get("/asr", asr)
    async with TestServer(app) as server, http.open():
        endpoint = str(server.make_url("/asr").with_scheme("ws"))
        provider = STT(api_key="key", model_endpoint=endpoint, language="pt", temperature=0.3)
        stream = provider.stream()

        setup = await setups.get()
        await stream.aclose()

    assert setup["json_config"] == {"language": "pt", "temp": 0.3}
