"""Deepgram TTS (v1 and Flux) against a local peer speaking Deepgram's speak wire."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, http, tts
from livekit.plugins import deepgram

NO_RETRY = APIConnectOptions(max_retry=0)
PCM = b"\x34\x12" * 2400


def _server(
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]], method: str = "GET"
) -> TestServer:
    app = web.Application()
    app.router.add_route(method, "/speak", handler)
    return TestServer(app)


async def _speak(model: tts.TTS, text: str) -> bytes:
    async with model.stream(conn_options=NO_RETRY) as stream:
        stream.push_text(text)
        stream.end_input()
        return b"".join([event.frame.data.tobytes() async for event in stream])


async def test_batch_returns_headerless_pcm_and_sends_lowercase_booleans() -> None:
    """https://github.com/livekit/agents/pull/2225 https://github.com/livekit/agents/issues/2463"""
    received = []

    async def synthesize(request: web.Request) -> web.Response:
        received.append(dict(request.query))
        await request.read()
        return web.Response(body=PCM, content_type="audio/pcm")

    async with _server(synthesize, method="POST") as server, http.open():
        model = deepgram.TTS(
            api_key="test", mip_opt_out=True, base_url=str(server.make_url("/speak"))
        )
        async with model.synthesize("hello") as stream:
            audio = b"".join([event.frame.data.tobytes() async for event in stream])
        await model.aclose()

    assert audio[: len(PCM)] == PCM
    assert not any(audio[len(PCM) :])  # only padding of the last frame follows
    assert received[0]["encoding"] == "linear16"
    assert received[0]["container"] == "none"
    assert received[0]["mip_opt_out"] == "true"


@pytest.mark.parametrize("error", [False, True])
async def test_stream_drains_the_final_audio_or_raises_the_error_before_closing(
    error: bool,
) -> None:
    """https://github.com/livekit/agents/pull/3608 https://github.com/livekit/agents/issues/5719"""
    wire = []

    async def peer(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            wire.append(message.json()["type"])
            if wire[-1] == "Flush" and error:
                await ws.send_json({"type": "Error", "description": "quota exceeded"})
            elif wire[-1] == "Flush":
                await ws.send_bytes(PCM)
                await ws.send_json({"type": "Flushed"})
            elif wire[-1] == "Close":
                await ws.close()
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.TTS(api_key="test", base_url=str(server.make_url("/speak")))
        if error:
            with pytest.raises(APIError) as caught:
                await _speak(model, "Hello world.")
            assert isinstance(caught.value.body, dict)
            assert caught.value.body["description"] == "quota exceeded"
        else:
            assert await _speak(model, "Hello world.") == PCM
        await model.aclose()

    assert wire[-2:] == ["Flush", "Close"]


async def test_flux_turn_ends_on_speech_metadata_and_the_pooled_socket_speaks_again() -> None:
    """SpeechMetadata ends the turn; the trailing Flushed does not leak into the next one.

    https://github.com/livekit/agents/pull/6511
    """
    connections = 0

    async def peer(request: web.Request) -> web.WebSocketResponse:
        nonlocal connections
        connections += 1
        turns = 0
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for message in ws:
            if message.json()["type"] != "Flush":
                continue
            if turns:  # the previous turn's Flushed arrives late, at the start of this one
                await ws.send_json({"type": "Flushed"})
            turns += 1
            await ws.send_json({"type": "SpeechStarted"})
            await ws.send_bytes(PCM)
            await ws.send_json(
                {
                    "type": "SpeechMetadata",
                    "audio_duration_ms": 100,
                    "input_character_count": 12,
                    "billable_character_count": 11,
                }
            )
        return ws

    async with _server(peer) as server, http.open():
        model = deepgram.TTSv2(api_key="test", base_url=str(server.make_url("/speak")))
        first = await _speak(model, "hello world")
        second = await _speak(model, "hello again")
        await model.aclose()

    assert first[: len(PCM)] == PCM
    assert second[: len(PCM)] == PCM
    assert connections == 1
