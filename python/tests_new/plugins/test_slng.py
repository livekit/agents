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


@dataclass
class Slng:
    """The local SLNG bridge's endpoint and each socket's init config."""

    endpoint: str = ""
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

    app = web.Application()
    app.router.add_get(BRIDGE, speak)
    async with TestServer(app) as server, http.open():
        peer.endpoint = str(server.make_url(BRIDGE).with_scheme("ws"))
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
