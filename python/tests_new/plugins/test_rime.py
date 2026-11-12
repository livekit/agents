"""Rime TTS: the v1 WebSocket turn in both wire encodings, raw PCM over HTTP, and errors."""

from __future__ import annotations

import base64
import logging
import traceback
from typing import Literal

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from google.protobuf import json_format
from rime_api import text_to_speech_pb2 as proto

from livekit.agents import APIConnectOptions, APIError, APIStatusError, http
from livekit.agents.tts import ChunkedStream, SynthesizeStream
from livekit.plugins.rime import TTS

SECRET = "customer-secret-marker"
PCM = b"\x01\x00" * 2205


def rime_v1_peer(
    *, fail_after_audio: bool = False
) -> tuple[web.Application, list[dict[str, object]]]:
    """A Rime v1 socket at ``/coda/ws``: ready, then started / audio / done for each context.

    Requests are decoded and recorded in their JSON form; responses go out in whichever
    encoding (``rime.v1.binary`` or ``rime.v1.json``) the client negotiated.
    """
    received: list[dict[str, object]] = []

    async def handle(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(protocols=("rime.v1.binary", "rime.v1.json"))
        await ws.prepare(request)

        async def send(payload: dict[str, object]) -> None:
            if ws.ws_protocol == "rime.v1.json":
                await ws.send_json(payload)
            else:
                response = json_format.ParseDict(payload, proto.WebSocketResponse())
                await ws.send_bytes(response.SerializeToString())

        await send({"ready": {"protocol": 1, "languages": ["eng"]}})
        async for message in ws:
            envelope = proto.WebSocketRequest()
            if message.type == aiohttp.WSMsgType.BINARY:
                envelope.ParseFromString(message.data)
            else:
                json_format.Parse(message.data, envelope)
            event = json_format.MessageToDict(envelope)
            received.append(event)
            context = event.get("contextId", "")
            if "start" in event:
                await send({"contextId": context, "started": {"requestId": "req-1"}})
            elif "text" in event and event["text"]:
                await send({"contextId": context, "audio": base64.b64encode(PCM).decode()})
                if fail_after_audio:
                    await send(
                        {"contextId": context, "error": {"kind": "unavailable", "message": "late"}}
                    )
            elif "end" in event:
                await send({"contextId": context, "done": {}})
            elif "cancel" in event:
                await send({"contextId": context, "cancelled": {}})
        return ws

    app = web.Application()
    app.router.add_get("/coda/ws", handle)
    return app, received


@pytest.mark.parametrize("encoding", ["binary", "json"])
async def test_v1_turn_streams_audio_in_either_encoding(
    encoding: Literal["binary", "json"],
) -> None:
    """A streamed turn sends start, text and end, and plays the audio Rime returns."""
    app, received = rime_v1_peer()
    async with TestServer(app) as server, http.open():
        tts = TTS(
            api_key="key",
            websocket_url=str(server.make_url("/coda/ws").with_scheme("ws")),
            websocket_protocol=encoding,
        )
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=0))
        stream.push_text("Hello.")
        stream.end_input()

        audio = b"".join([bytes(ev.frame.data) async for ev in stream])
        await stream.aclose()
        await tts.aclose()

    assert audio[: len(PCM)] == PCM
    kinds = [next(k for k in event if k != "contextId") for event in received]
    assert kinds[0] == "start" and "text" in kinds and kinds[-1] == "end"
    assert "".join(str(e["text"]) for e in received if "text" in e).strip() == "Hello."


async def test_v1_does_not_retry_after_partial_audio() -> None:
    """A failure after audio was played is final, never replayed on a new socket.

    https://github.com/livekit/agents/pull/6978
    """
    app, received = rime_v1_peer(fail_after_audio=True)
    async with TestServer(app) as server, http.open():
        tts = TTS(api_key="key", websocket_url=str(server.make_url("/coda/ws").with_scheme("ws")))
        stream = tts.stream(conn_options=APIConnectOptions(max_retry=1, retry_interval=0))
        stream.push_text("hello")
        stream.end_input()

        with pytest.raises(APIStatusError):
            _ = [ev async for ev in stream]
        await stream.aclose()
        await tts.aclose()

    assert sum("start" in event for event in received) == 1


@pytest.mark.parametrize("model", ["mistv2", "coda"])
async def test_http_returns_raw_pcm_with_the_updated_speed(model: str) -> None:
    """HTTP synthesis plays raw PCM unchanged and sends the speed set by ``update_options``.

    https://github.com/livekit/agents/issues/2461 https://github.com/livekit/agents/pull/2202
    https://github.com/livekit/agents/pull/6287
    """
    pcm = b"\x17\x00" * 4800
    requests: list[tuple[str, dict[str, object]]] = []

    async def speak(request: web.Request) -> web.Response:
        requests.append((request.headers["Accept"], await request.json()))
        return web.Response(body=pcm, content_type="audio/pcm")

    app = web.Application()
    app.router.add_post("/custom/tts", speak)
    async with TestServer(app) as server, http.open():
        tts = TTS(
            api_key="key",
            model=model,
            speaker="speaker",
            sample_rate=24000,
            base_url=str(server.make_url("/custom/tts")),
            allow_custom_endpoint=True,
        )
        tts.update_options(speed_alpha=1.2)

        async with tts.synthesize("Hello.", conn_options=APIConnectOptions(max_retry=0)) as stream:
            audio = b"".join([bytes(ev.frame.data) async for ev in stream])
        await tts.aclose()

    assert audio[: len(pcm)] == pcm
    assert not any(audio[len(pcm) :])
    [(accept, body)] = requests
    assert accept == "audio/pcm"
    assert body["speedAlpha"] == 1.2
    assert body["samplingRate"] == 24000


@pytest.mark.parametrize(
    ("mode", "status"),
    [("http", 400), ("http", 500), ("handshake", 400), ("handshake", 500), ("redirect", 302)],
)
async def test_errors_do_not_expose_provider_or_transport_details(
    mode: str, status: int, caplog: pytest.LogCaptureFixture
) -> None:
    """A rejected request raises a status error with no provider text, URL token or body.

    https://github.com/livekit/agents/pull/6978 https://github.com/livekit/agents/pull/7964
    """
    requests = 0

    async def reject(request: web.Request) -> web.Response:
        nonlocal requests
        requests += 1
        if mode == "redirect":  # never followed; the Location (a token) stays out of the error
            return web.Response(status=302, headers={"Location": f"ftp://localhost/f?t={SECRET}"})
        return web.Response(status=status, reason=SECRET, text=SECRET)

    caplog.set_level(logging.WARNING, logger="livekit.agents")
    app = web.Application()
    app.router.add_route("*", "/coda/ws", reject)
    options = APIConnectOptions(max_retry=1, retry_interval=0, connect_timeout=1, read_timeout=1)
    async with TestServer(app) as server, http.open():
        url = server.make_url("/coda/ws")
        if mode == "handshake":
            tts = TTS(api_key="key", websocket_url=str(url.with_scheme("ws")))
            streaming = tts.stream(conn_options=options)
            streaming.push_text("hello")
            streaming.end_input()
            stream: SynthesizeStream | ChunkedStream = streaming
        else:
            tts = TTS(api_key="key", base_url=str(url))
            stream = tts.synthesize("hello", conn_options=options)

        with pytest.raises(APIError) as raised:
            _ = [ev async for ev in stream]
        await stream.aclose()
        await tts.aclose()

    error = raised.value
    retryable = mode == "redirect" or status == 500
    assert isinstance(error, APIStatusError)
    assert (error.status_code, error.request_id, error.body) == (status, None, None)
    assert error.retryable is retryable
    assert requests == (2 if retryable else 1)
    rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert SECRET not in rendered and SECRET not in repr(error)
    assert error.__cause__ is not None, "the cause is kept, redacted"
    assert SECRET not in caplog.text
    assert sum("retrying in" in r.getMessage() for r in caplog.records) == (1 if retryable else 0)
