"""Respeecher TTS plugin: the model is part of the websocket URL, against a local peer."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDictProxy

from livekit.agents import http
from livekit.plugins.respeecher import TTS
from tests_new.fakes import until

EN = "/public/tts/en-rt"
UA = "/public/tts/ua-rt"
PCM = base64.b64encode(b"\x00\x00" * 320).decode()


@dataclass
class Respeecher:
    """The local Respeecher's record of sockets and requests."""

    base_url: str = ""
    opened: list[str] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)
    requests: list[tuple[str, CIMultiDictProxy[str]]] = field(default_factory=list)


@pytest.fixture
async def respeecher(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Respeecher]:
    """A local Respeecher behind a TLS terminator: answers each sentence with audio, then done.

    ``opened``/``closed`` list the model path of each socket; ``requests`` the path and headers.
    """
    peer = Respeecher()

    async def speak(request: web.Request) -> web.WebSocketResponse:
        model = request.path.removesuffix("/tts/websocket")
        peer.requests.append((request.raw_path, request.headers))
        peer.opened.append(model)
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        async for message in socket:
            packet: dict[str, object] = json.loads(message.data)
            reply = {"type": "chunk", "data": PCM} if packet["continue"] else {"type": "done"}
            await socket.send_json({"context_id": packet["context_id"], **reply})
        peer.closed.append(model)
        return socket

    app = web.Application()
    app.router.add_get("/{model:.*}/tts/websocket", speak)
    async with TestServer(app) as server, http.open() as session:
        connect = session.ws_connect

        # Any: ws_connect's keyword options, forwarded untouched
        async def local_tls_terminator(url: str, **kwargs: Any) -> http.WebSocket:
            return await connect(url.replace("wss://", "ws://", 1), **kwargs)

        monkeypatch.setattr(session, "ws_connect", local_tls_terminator)
        peer.base_url = str(server.make_url("")).replace("http://", "https://").rstrip("/")
        yield peer


async def _say(tts: TTS, text: str) -> None:
    async with tts.stream() as stream:
        stream.push_text(text)
        stream.end_input()
        async for _ in stream:
            pass


async def test_changing_model_closes_the_idle_pooled_connection(respeecher: Respeecher) -> None:
    """After a model change the next reply opens the new model, and the old socket is closed.

    https://github.com/livekit/agents/pull/7237
    """
    tts = TTS(api_key="key", model=EN, base_url=respeecher.base_url)
    await _say(tts, "Hello.")

    tts.update_options(model=UA)
    await _say(tts, "Hello again.")

    assert respeecher.opened == [EN, UA]
    await until(lambda: respeecher.closed == [EN], "the English socket is closed")
    await tts.aclose()


async def test_changing_model_lets_an_in_use_connection_finish(respeecher: Respeecher) -> None:
    """A reply in flight finishes on its socket; that socket closes once it is returned.

    https://github.com/livekit/agents/pull/7237
    """
    tts = TTS(api_key="key", model=EN, base_url=respeecher.base_url)
    async with tts.stream() as stream:
        stream.push_text("A reply already playing.")
        await until(lambda: respeecher.opened == [EN], "the reply holds an English socket")

        tts.update_options(model=UA)
        stream.end_input()
        frames = [audio async for audio in stream]

    assert frames, "changing the model severed a stream in flight"
    await _say(tts, "Hello again.")
    await until(lambda: respeecher.closed == [EN], "the returned English socket is closed")
    await tts.aclose()


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/8286 reproduces: websocket URL contains API key",
)
async def test_websocket_credentials_are_headers_only(respeecher: Respeecher) -> None:
    """WebSocket authentication never places credentials in a URL. https://github.com/livekit/agents/pull/8286"""
    tts = TTS(api_key="private-key", base_url=respeecher.base_url)
    await _say(tts, "Hello.")
    await tts.aclose()

    path, headers = respeecher.requests[0]
    assert "private-key" not in path
    assert headers.get("X-API-Key") == "private-key"
    assert "source=livekit" in path
