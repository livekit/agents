from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livekit.agents import APIConnectOptions, APIError, APIStatusError
from livekit.agents.inference import STT
from livekit.agents.stt import SpeechEventType

pytestmark = pytest.mark.unit


def _make_stt(base_url: str, session: aiohttp.ClientSession, **kwargs: Any) -> STT:
    return STT(
        model="deepgram/nova-3",
        api_key="test-key",
        api_secret="test-secret",
        base_url=base_url,
        http_session=session,
        **kwargs,
    )


@contextlib.asynccontextmanager
async def _gateway(
    handler: Callable[[web.Request], Any],
) -> AsyncIterator[tuple[str, aiohttp.ClientSession]]:
    app = web.Application()
    app.router.add_get("/stt", handler)
    server = TestServer(app)
    await server.start_server()
    session = aiohttp.ClientSession()
    try:
        yield str(server.make_url("")).rstrip("/"), session
    finally:
        await session.close()
        await server.close()


async def test_dial_includes_model_and_input_end_closes_session() -> None:
    message_types: list[str] = []
    transcripts: list[str] = []
    request_model = ""
    socket_closed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal request_model
        request_model = request.query.get("model", "")
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for msg in ws:
                event = json.loads(msg.data)
                message_types.append(event["type"])
                if event["type"] == "session.close":
                    await ws.send_json(
                        {"type": "final_transcript", "transcript": "final words", "language": "en"}
                    )
                    await ws.send_json({"type": "session.closed"})
        finally:
            socket_closed.set()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        try:
            stream.end_input()
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    transcripts.append(event.alternatives[0].text)
            await asyncio.wait_for(socket_closed.wait(), timeout=1.0)
        finally:
            await stream.aclose()

    assert request_model == "deepgram/nova-3"
    assert message_types == ["session.create", "session.finalize", "session.close"]
    assert transcripts == ["final words"]


async def test_disconnect_before_session_closed_after_input_end_is_not_retried() -> None:
    connection_count = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if json.loads(msg.data)["type"] == "session.close":
                await ws.close(code=1011, message=b"missing session.closed")
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001, timeout=1.0)
        )
        try:
            stream.end_input()
            with pytest.raises(APIStatusError) as exc_info:
                async for _ in stream:
                    pass
        finally:
            await stream.aclose()

    assert connection_count == 1
    assert errors == [exc_info.value]
    assert exc_info.value.retryable is False
    assert exc_info.value.status_code == 1011


async def test_stream_close_sends_session_close_and_closes_socket() -> None:
    message_types: list[str] = []
    session_created = asyncio.Event()
    socket_closed = asyncio.Event()

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for msg in ws:
                event = json.loads(msg.data)
                message_types.append(event["type"])
                if event["type"] == "session.create":
                    session_created.set()
        finally:
            socket_closed.set()
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0, timeout=1.0))
        await asyncio.wait_for(session_created.wait(), timeout=1.0)
        await stream.aclose()
        await asyncio.wait_for(socket_closed.wait(), timeout=1.0)

    assert message_types == ["session.create", "session.close"]


async def test_inactivity_timeout_is_not_retried() -> None:
    connection_count = 0

    async def handler(request: web.Request) -> web.WebSocketResponse:
        nonlocal connection_count
        connection_count += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if json.loads(msg.data)["type"] == "session.finalize":
                await ws.send_json(
                    {
                        "type": "error",
                        "code": 2007,
                        "message": "customer content must not reach the API error",
                    }
                )
        return ws

    async with _gateway(handler) as (base_url, session):
        stt = _make_stt(base_url, session)
        errors: list[Exception] = []
        stt.on("error", lambda event: errors.append(event.error))
        stream = stt.stream(
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0.001, timeout=1.0)
        )
        try:
            stream.end_input()
            with pytest.raises(APIError) as exc_info:
                async for _ in stream:
                    pass
        finally:
            await stream.aclose()

    assert connection_count == 1
    assert errors == [exc_info.value]
    assert exc_info.value.message == "LiveKit Inference STT returned an error"
    assert exc_info.value.body == {"code": 2007}
    assert exc_info.value.retryable is False
