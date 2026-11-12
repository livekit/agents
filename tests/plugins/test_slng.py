"""SLNG TTS plugin: streaming text privacy and per-connection init, against a local peer."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, http
from livekit.plugins.slng import TTS

PCM = b"\x00\x01" * 2400
BRIDGE = "/v1/bridges/unmute/tts/deepgram/aura:2"
DOWN_BRIDGE = "/v1/bridges/unmute/tts/cartesia/sonic:3"


@dataclass
class Slng:
    """The local SLNG bridge's endpoint and each socket's init config."""

    endpoint: str = ""
    down_endpoint: str = ""  # a bridge that answers 503
    configs: list[dict[str, object]] = field(default_factory=list)


@pytest.fixture
async def slng() -> AsyncIterator[Slng]:
    """A local SLNG bridge: answers each flush with audio; ``configs`` holds each socket's init config config."""
    peer = Slng()

    async def speak(request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        peer.configs.append(json.loads((await socket.receive()).data)["config"])
        await socket.send_json({"type": "ready"})
        async for message in socket:
            if json.loads(message.data).get("type") == "flush":
                await socket.send_bytes(PCM)
                await socket.send_json({"type": "audio_end"})
        return socket

    async def unavailable(request: web.Request) -> web.Response:
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get(BRIDGE, speak)
    app.router.add_get(DOWN_BRIDGE, unavailable)
    async with TestServer(app) as server, http.open():
        peer.endpoint = str(server.make_url(BRIDGE).with_scheme("ws"))
        peer.down_endpoint = str(server.make_url(DOWN_BRIDGE).with_scheme("ws"))
        yield peer


async def _say(tts: TTS, text: str) -> bytes:
    async with tts.stream(conn_options=APIConnectOptions(max_retry=0)) as stream:
        stream.push_text(text)
        stream.end_input()
        return b"".join([bytes(event.frame.data) async for event in stream])


async def test_streaming_text_is_kept_out_of_log_messages(
    slng: Slng, caplog: pytest.LogCaptureFixture
) -> None:
    """Text being synthesized never appears in the plugin's log lines."""
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice")
    caplog.set_level(logging.DEBUG, logger="livekit.plugins.slng")

    audio = await _say(tts, "CONFIDENTIALACCOUNT")
    await tts.aclose()

    assert audio == PCM
    assert all("CONFIDENTIALACCOUNT" not in record.getMessage() for record in caplog.records)


async def test_speed_update_opens_a_connection_with_current_init(slng: Slng) -> None:
    """A speed change opens a socket whose init carries it. https://github.com/livekit/agents/issues/6174"""
    tts = TTS(api_key="key", model=slng.endpoint, voice="voice")

    for speed in (1.0, 1.2):
        tts.update_options(speed=speed)
        assert await _say(tts, "Hello.") == PCM
    await tts.aclose()

    assert [config["speed"] for config in slng.configs] == [1.0, 1.2]


async def test_connection_chain_fails_over_without_losing_audio(slng: Slng) -> None:
    """An unavailable first bridge hands the turn to the next one, which plays all of it."""
    # Catches: the chain giving up on the first bridge's 503, or the fallback leg dropping
    # the audio of the turn it took over.
    tts = TTS(api_key="key", connections=[slng.down_endpoint, slng.endpoint], voice="voice")

    options = APIConnectOptions(max_retry=0, retry_interval=0)
    async with tts.stream(conn_options=options) as stream:
        stream.push_text("Hello.")
        stream.end_input()
        audio = b"".join([bytes(event.frame.data) async for event in stream])
    await tts.aclose()

    assert audio == PCM
    assert len(slng.configs) == 1
