"""Test that a rejected HTTP request or WebSocket handshake keeps the provider's status,
request id, and body, and never the request headers that carry the API key."""

from __future__ import annotations

import asyncio
import contextlib

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from multidict import CIMultiDict

from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    create_api_error_from_response,
)
from livekit.agents.llm import RealtimeModelError
from livekit.plugins.openai.realtime.realtime_model import RealtimeModel

pytestmark = pytest.mark.unit

API_KEY = "sk-test-secret-key"


def test_reads_the_request_id_header_case_insensitively() -> None:
    err = create_api_error_from_response(
        "Unauthorized",
        status=401,
        headers=CIMultiDict({"DG-Request-Id": "req_1"}),
        request_id_header="dg-request-id",
    )

    assert err.message == "Unauthorized"  # used as is, like the APIStatusError it replaces
    assert err.status_code == 401
    assert err.request_id == "req_1"
    assert err.body is None
    assert "request_id=req_1" in str(err)


@pytest.mark.parametrize(
    ("status", "retryable"), [(401, False), (403, False), (429, True), (503, True)]
)
def test_retryable_follows_the_status(status: int, retryable: bool) -> None:
    err = create_api_error_from_response("error", status=status, headers=None)

    assert err.retryable is retryable
    assert err.request_id is None


def test_body_is_masked_then_truncated() -> None:
    body = f"invalid API key {API_KEY}" + "x" * 1000
    err = create_api_error_from_response(
        "Unauthorized", status=401, headers=None, body=body, secret=API_KEY
    )

    assert isinstance(err.body, str)
    assert err.body.startswith("invalid API key ***")
    assert len(err.body) == 500
    assert API_KEY not in str(err)


async def _start_rejecting_server(status: int) -> tuple[TestServer, list[int]]:
    attempts: list[int] = []

    async def reject(request: web.Request) -> web.Response:
        attempts.append(status)
        return web.Response(
            status=status, text='{"error": "invalid API key"}', headers={"X-Request-Id": "req_gw"}
        )

    app = web.Application()
    app.router.add_get("/realtime", reject)
    server = TestServer(app)
    await server.start_server()
    return server, attempts


async def test_handshake_error_keeps_status_and_request_id_without_headers() -> None:
    server, _ = await _start_rejecting_server(401)
    try:
        async with aiohttp.ClientSession() as session:
            with pytest.raises(aiohttp.WSServerHandshakeError) as exc_info:
                await session.ws_connect(
                    server.make_url("/realtime"), headers={"Authorization": f"Bearer {API_KEY}"}
                )
        e = exc_info.value
        assert API_KEY in repr(e)  # the reason the helper takes fields, not the error

        err = create_api_error_from_response(e.message, status=e.status, headers=e.headers)
    finally:
        await server.close()

    assert err.status_code == 401
    assert err.request_id == "req_gw"
    assert err.retryable is False
    assert API_KEY not in str(err)
    assert API_KEY not in repr(err)


async def _first_error(model: RealtimeModel) -> RealtimeModelError:
    session = model.session()
    errors: asyncio.Queue[RealtimeModelError] = asyncio.Queue()
    session.on("error", errors.put_nowait)
    try:
        return await asyncio.wait_for(errors.get(), 10)
    finally:
        # aclose re-raises the error that ended the session
        with contextlib.suppress(APIError):
            await session.aclose()


async def test_openai_realtime_rejected_handshake_retries_with_status() -> None:
    server, attempts = await _start_rejecting_server(401)
    model = RealtimeModel(api_key=API_KEY, base_url=str(server.make_url("")))
    session = model.session()
    errors: asyncio.Queue[RealtimeModelError] = asyncio.Queue()
    session.on("error", errors.put_nowait)
    try:
        first = await asyncio.wait_for(errors.get(), 10)
        second = await asyncio.wait_for(errors.get(), 10)
    finally:
        with contextlib.suppress(APIError):
            await session.aclose()
        await model.aclose()
        await server.close()

    # a rejected handshake stays a retryable connection error, as before
    assert len(attempts) >= 2
    assert first.recoverable is True
    assert second.recoverable is True
    assert isinstance(first.error, APIConnectionError)
    assert first.error.retryable is True
    assert "status_code=401" in str(first.error)
    assert "request_id=req_gw" in str(first.error)
    assert API_KEY not in str(first.error)
    assert API_KEY not in repr(first.error)


async def test_openai_realtime_unexpected_close_names_the_close_code() -> None:
    async def close_at_once(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close(code=4001, message=b"session expired")
        return ws

    app = web.Application()
    app.router.add_get("/realtime", close_at_once)
    server = TestServer(app)
    await server.start_server()
    model = RealtimeModel(api_key=API_KEY, base_url=str(server.make_url("")))
    try:
        error = await _first_error(model)
    finally:
        await model.aclose()
        await server.close()

    assert isinstance(error.error, APIConnectionError)
    assert "close_code=4001" in str(error.error)
    assert "reason='session expired'" in str(error.error)


async def test_deepgram_chunked_tts_keeps_the_response_body() -> None:
    from livekit.plugins.deepgram import TTS

    async def reject(request: web.Request) -> web.Response:
        # some providers echo the credential they rejected
        body = f'{{"err_msg": "Invalid credentials: {API_KEY}"}}'
        return web.Response(status=401, text=body, headers={"dg-request-id": "req_dg"})

    app = web.Application()
    app.router.add_post("/v1/speak", reject)
    server = TestServer(app)
    await server.start_server()
    async with aiohttp.ClientSession() as http_session:
        tts = TTS(
            api_key=API_KEY,
            base_url=str(server.make_url("/v1/speak")),
            http_session=http_session,
        )
        try:
            with pytest.raises(APIStatusError) as exc_info:
                await tts.synthesize("hello").collect()
        finally:
            await tts.aclose()
            await server.close()

    err = exc_info.value
    # same message and retryable as the APIStatusError raised before
    assert err.message == "Unauthorized"
    assert err.retryable is False
    assert err.status_code == 401
    assert err.request_id == "req_dg"
    assert err.body == '{"err_msg": "Invalid credentials: ***"}'
    assert API_KEY not in str(err)
