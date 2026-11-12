"""Vakyam TTS: HTTP error envelopes and the WebSocket streaming session, against a local peer."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http
from livekit.plugins.vakyam import TTS

PCM = b"\x00\x00" * 2400
NO_RETRY = APIConnectOptions(max_retry=0)


@dataclass
class Vakyam:
    """The local Vakyam: its script and what it received."""

    base_url: str = ""
    connections: int = 0
    texts: list[dict[str, object]] = field(default_factory=list)
    utterance: list[bytes | dict[str, object] | None] = field(default_factory=list)


@pytest.fixture
async def vakyam() -> AsyncIterator[Vakyam]:
    """A local Vakyam: ``utterance`` is what its socket sends per text (None drops the
    connection); ``texts`` is what it received."""
    peer = Vakyam()

    async def websocket(request: web.Request) -> web.WebSocketResponse:
        peer.connections += 1
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.send_json({"type": "connected"})
        await socket.receive()  # the session config
        await socket.send_json({"type": "configured"})
        async for message in socket:
            peer.texts.append(json.loads(message.data))
            for reply in peer.utterance:
                if reply is None:
                    assert request.transport is not None
                    request.transport.close()  # the connection drops
                elif isinstance(reply, bytes):
                    await socket.send_bytes(reply)
                else:
                    await socket.send_json(reply)
        return socket

    async def stream(_: web.Request) -> web.Response:
        envelope = {"error": {"code": "rate_limit_exceeded", "message": "Too many requests."}}
        return web.json_response(envelope, status=429)

    app = web.Application()
    app.router.add_get("/v1/tts/websocket", websocket)
    app.router.add_post("/v1/tts/stream", stream)
    async with TestServer(app) as server, http.open():
        peer.base_url = str(server.make_url(""))
        yield peer


async def test_http_error_envelope_keeps_the_status_and_code_without_the_message(
    vakyam: Vakyam,
) -> None:
    """A 429 envelope is a retryable status error whose body omits the provider's message.

    https://github.com/livekit/agents/pull/7009
    """
    tts = TTS(api_key="key", base_url=vakyam.base_url)

    with pytest.raises(APIStatusError, match="status 429") as caught:
        async for _ in tts.synthesize("Hello.", conn_options=NO_RETRY):
            pass

    assert caught.value.status_code == 429
    assert caught.value.retryable is True
    assert caught.value.body == {"status_code": 429, "error_code": "rate_limit_exceeded"}
    assert "Too many requests" not in str(caught.value)


async def test_stream_does_not_retry_after_partial_audio(vakyam: Vakyam) -> None:
    """A socket lost after some audio fails the stream instead of replaying the utterance.

    https://github.com/livekit/agents/pull/7009
    """
    vakyam.utterance = [PCM, None]
    tts = TTS(api_key="key", base_url=vakyam.base_url)
    options = APIConnectOptions(max_retry=3, retry_interval=0.0, read_timeout=1)

    with pytest.raises(APIError):
        async with tts.stream(conn_options=options) as stream:
            stream.push_text("Hello.")
            stream.end_input()
            async for _ in stream:
                pass

    assert vakyam.connections == 1
    await tts.aclose()


async def test_synthesize_stream_waits_for_end_of_utterance(vakyam: Vakyam) -> None:
    """All audio before end_of_utterance plays, billed with the utterance's character count.

    https://github.com/livekit/agents/pull/7009
    """
    end = {"type": "end_of_utterance", "characters_used": 4, "duration_seconds": 0.5}
    vakyam.utterance = [PCM, PCM, end]
    tts = TTS(api_key="key", base_url=vakyam.base_url)

    async with tts.stream() as stream:
        stream.push_text("Hello.")
        stream.end_input()
        frames = [audio async for audio in stream]
    await tts.aclose()

    assert vakyam.texts[0] == {"type": "text", "text": "Hello."}
    assert sum(len(f.frame.data) for f in frames) * 2 == 2 * len(PCM)
    assert [f.characters_count for f in frames if f.characters_count] == [4]
