"""AsyncAI TTS: settings sent in the websocket init payload, against a local peer."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import http
from livekit.plugins.asyncai import TTS

PCM = base64.b64encode(b"\x00\x00" * 320).decode()


# Any: decoded wire JSON, indexed by the provider's documented shape
Wire = dict[str, Any]


class _Peer:
    def __init__(self) -> None:
        self.inits: list[Wire] = []

    async def speak(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.inits.append(json.loads((await socket.receive()).data))
        async for message in socket:
            packet = json.loads(message.data)
            if packet["transcript"]:
                await socket.send_json({"context_id": packet["context_id"], "audio": PCM})
            else:
                await socket.send_json({"context_id": packet["context_id"], "final": True})
        return socket


async def _say(tts: TTS, text: str) -> None:
    async with tts.stream() as stream:
        stream.push_text(text)
        stream.end_input()
        async for _ in stream:
            pass


@pytest.mark.parametrize(
    ("option", "value", "sent"),
    [
        ("voice", "other-voice", lambda init: init["voice"]["id"]),
        ("model", "async_flash_v1.0", lambda init: init["model_id"]),
        ("language", "es", lambda init: init["language"]),
    ],
)
async def test_changed_connection_setting_reaches_the_next_synthesis(
    option: str,
    value: str,
    sent: Callable[[Wire], object],
) -> None:
    """A voice, model or language change opens a fresh socket that sends it.

    https://github.com/livekit/agents/pull/7132
    """
    peer = _Peer()
    app = web.Application()
    app.router.add_get("/text_to_speech/websocket/ws", peer.speak)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key="key", model="asyncflow_v2.0", base_url=str(server.make_url("")))
        await _say(tts, "Hello there.")

        tts.update_options(**{option: value})
        await _say(tts, "Hello again.")
        await tts.aclose()

    assert len(peer.inits) == 2
    assert sent(peer.inits[-1]) == value
