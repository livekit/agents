"""The framework serves conversation routes; application HTTP routes stay application owned."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope

from livekit.agents import AgentServer
from livekit.agents.a2a import A2ASessionContext, Credentials
from livekit.agents.http import serve_endpoint
from livekit.agents.storage import AgentDBStorage

from .test_http import FakeStream, parse_response, preamble, request_bytes

pytestmark = pytest.mark.unit


async def test_endpoint_route_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    key, secret = "APIroute", "route-inventory-secret-secret-secret"
    monkeypatch.setenv("LIVEKIT_URL", "wss://agents.invalid")
    storage = AgentDBStorage(api_key=key, api_secret=secret)
    server = AgentServer(storage=storage, agent_name="routes", api_key=key, api_secret=secret)

    @server.a2a_session(endpoint="desk", description="Answers routing test messages.")
    async def desk(ctx: A2ASessionContext) -> str:
        return "received: " + ctx.message.text_content

    @server._http.get("/application-content")
    async def application_content(_: Request) -> PlainTextResponse:
        return PlainTextResponse("application owned")

    token = Credentials(key, secret).project_token(identity="route-test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server._http.app),
        base_url="http://agents.test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        card = await client.get("/desk/.well-known/agent-card.json")
        assert card.status_code == 200
        assert card.json()["name"] == "desk"
        for method in ["message/send", "message/stream"]:
            response = await client.post(
                "/desk/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": method,
                    "method": method,
                    "params": "invalid",
                },
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/json")
            assert response.json()["error"]["code"] == -32602
        for path in [
            "/content",
            "/history",
            "/tasks",
            "/threads/AT_example/history",
            "/desk/content",
            "/desk/history",
            "/desk/tasks",
        ]:
            assert (await client.get(path)).status_code == 404, path
        response = await client.get("/application-content")
        assert response.status_code == 200
        assert response.text == "application owned"


async def test_asgi_send_accepts_a_message_keyword_on_the_native_endpoint_bridge() -> None:
    async def app(scope: Scope, receive: Receive, send: Callable[..., Awaitable[None]]) -> None:
        await send(message={"type": "http.response.start", "status": 200})
        await send(message={"type": "http.response.body", "body": b"keyword response"})

    stream = await _native_asgi_exchange(app)
    head, body, _ = parse_response(bytes(stream.out))
    assert head.status_code == 200
    assert body == b"keyword response"
    assert stream.finished and stream.reset_code is None


@pytest.mark.parametrize("arguments", ["missing", "duplicate", "unknown", "extra-positional"])
async def test_asgi_send_refuses_invalid_arguments_before_sending_a_response(
    arguments: str,
) -> None:
    failures: list[Exception] = []

    async def app(scope: Scope, receive: Receive, send: Callable[..., Awaitable[None]]) -> None:
        message = {"type": "http.response.start", "status": 200}
        try:
            if arguments == "missing":
                await send()
            elif arguments == "duplicate":
                await send(message, message=message)
            elif arguments == "unknown":
                await send(message, unexpected=True)
            else:
                await send(message, message)
        except Exception as error:
            failures.append(error)
            await send(message)
        await send({"type": "http.response.body", "body": b"recovered"})

    stream = await _native_asgi_exchange(app)
    head, body, _ = parse_response(bytes(stream.out))
    assert len(failures) == 1, "invalid arguments reached the ASGI connection"
    assert isinstance(failures[0], TypeError), failures[0]
    assert head.status_code == 200 and body == b"recovered"
    assert stream.finished and stream.reset_code is None


async def _native_asgi_exchange(app: ASGIApp) -> FakeStream:
    # A raw app receives the native callables; Starlette middleware wraps send in a Python
    # function that normalizes keyword arguments before they reach the native bridge.
    stream = FakeStream()
    stream.feed(request_bytes("GET", "/"))
    stream.feed_eof()
    await asyncio.wait_for(serve_endpoint(stream, preamble(), app), 5)
    await stream.wait_done()
    return stream
