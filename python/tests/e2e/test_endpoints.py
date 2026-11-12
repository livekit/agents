"""HTTP and WebSocket routes through the LiveKit node: a registered worker serves them over the
endpoint data plane and clients call the node's /agents/{agent}/{deployment}/ front. Against a
local development node, set LIVEKIT_WORKER_INSECURE=1 for its self-signed WebTransport identity."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from .resources import REQUEST_SECONDS, STARTUP_SECONDS

pytestmark = pytest.mark.e2e


@dataclass
class Endpoint:
    base: str
    server: object
    # a project token: the agentEndpoint call grant the node admits non-public routes with
    token: str
    # signed by the project, without the call grant
    room_token: str
    closed: list[int] = field(default_factory=list)


def seen_by_the_app(scope: Request | WebSocket) -> dict[str, object]:
    """What reached the application: the node's verdict and the caller's token as sent."""
    return {
        "authorized": scope.state.lk_request.authorized,
        "authorization": scope.headers.get("authorization"),
        "query": scope.url.query,
    }


def add_routes(endpoint: Endpoint) -> None:
    http = endpoint.server.http  # type: ignore[attr-defined]

    @http.get("/hello", public=True)
    async def hello(request: Request) -> Response:
        return JSONResponse({"route": request.state.lk_request.route, "query": request.url.query})

    @http.post("/echo-size", public=True)
    async def echo_size(request: Request) -> Response:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        return PlainTextResponse(str(size))

    @http.get("/stream", public=True)
    async def stream(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            for i in range(5):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0.02)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/late-boom", public=True)
    async def late_boom(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"first chunk\n"
            await asyncio.sleep(0.05)
            raise RuntimeError("the source died")

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/ws", public=True)
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="chat")
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    endpoint.closed.append(message["code"])
                    return
                if message.get("text") is not None:
                    if message["text"] == "close-me":
                        await websocket.close(code=4000, reason="bye")
                        return
                    await websocket.send_text("echo:" + message["text"])
                else:
                    await websocket.send_bytes(bytes(reversed(message["bytes"])))
        except WebSocketDisconnect as error:
            endpoint.closed.append(error.code)

    @http.get("/slow-stream", public=True)
    async def slow_stream(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            for i in range(10):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0.1)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/endless", public=True)
    async def endless(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            while True:
                yield b"tick\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/push", public=True)
    async def push(websocket: WebSocket) -> None:
        # Sends only: the worker still answers pings and notices the close.
        await websocket.accept()
        for i in range(1000):
            await websocket.send_text(f"tick {i}")
            await asyncio.sleep(0.01)

    @http.websocket("/refused", public=True)
    async def refused(websocket: WebSocket) -> None:
        await websocket.close()

    # non-public: the node admits only a caller holding the endpoint call grant
    @http.get("/private")
    async def private(request: Request) -> Response:
        return JSONResponse(seen_by_the_app(request))

    @http.websocket("/private-ws")
    async def private_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(seen_by_the_app(websocket))
        await websocket.close()


async def _entrypoint(ctx: object) -> None:
    """The voice door's entrypoint; these journeys dispatch no job."""


@pytest.fixture
async def endpoint(
    live_credentials: tuple[str, str, str], tmp_path: Path
) -> AsyncIterator[Endpoint]:
    from livekit import api
    from livekit.agents import AgentServer
    from livekit.agents.a2a import Credentials
    from livekit.agents.storage import SQLiteStorage

    url, key, secret = live_credentials
    name = "e2e-endpoints-" + uuid.uuid4().hex[:8]
    parts = urlsplit(url)
    scheme = "https" if parts.scheme in ("https", "wss") else "http"
    server = AgentServer(
        agent_name=name,
        storage=SQLiteStorage(directory=tmp_path),
        listeners={"health": {"port": 0}},
        num_idle_processes=0,
    )
    endpoint = Endpoint(
        f"{scheme}://{parts.netloc}/agents/{name}/_",
        server,
        token=Credentials(key, secret).project_token(),
        room_token=api.AccessToken(key, secret).with_identity("caller").to_jwt(),
    )
    add_routes(endpoint)

    # A server serves at least one door; a voice door adds no endpoint route.
    server.rtc_session()(_entrypoint)
    run = asyncio.create_task(server._run(devmode=False))
    try:
        async with aiohttp.ClientSession() as http:
            async with asyncio.timeout(STARTUP_SECONDS):
                while True:
                    if run.done():
                        run.result()
                    try:
                        async with http.get(f"{endpoint.base}/hello") as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.1)
        yield endpoint
    finally:
        await server.aclose()
        await asyncio.wait_for(run, REQUEST_SECONDS)


async def test_http_requests_and_streams_through_the_node(endpoint: Endpoint) -> None:
    base = endpoint.base
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as http:
        async with http.get(f"{base}/hello?x=1") as response:
            assert await response.json() == {"route": "/hello", "query": "x=1"}

        body = os.urandom(3 << 20)
        async with http.post(f"{base}/echo-size", data=body) as response:
            assert await response.text() == str(len(body))

        async def upload() -> AsyncIterator[bytes]:
            for _ in range(10):
                yield b"x" * 1000

        async with http.post(f"{base}/echo-size", data=upload()) as response:
            assert await response.text() == "10000"

        async with http.get(f"{base}/stream") as response:
            assert await response.text() == "".join(f"chunk {i}\n" for i in range(5))

        # A failure after the response head must not reach the client as a clean body.
        with pytest.raises(aiohttp.ClientPayloadError):
            async with http.get(f"{base}/late-boom") as response:
                await response.read()

        async def get(i: int) -> str:
            async with http.get(f"{base}/hello?x={i}") as response:
                return (await response.json())["query"]

        queries = await asyncio.gather(*(get(i) for i in range(100)))
        assert queries == [f"x={i}" for i in range(100)]


async def test_websockets_through_the_node(endpoint: Endpoint) -> None:
    base = endpoint.base
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as http:
        async with http.ws_connect(f"{base}/ws", protocols=["chat"], heartbeat=0.2) as ws:
            assert ws.protocol == "chat"
            await ws.send_str("hi")
            assert await ws.receive_str(timeout=REQUEST_SECONDS) == "echo:hi"
            await ws.send_bytes(b"abc")
            assert await ws.receive_bytes(timeout=REQUEST_SECONDS) == b"cba"
            large = os.urandom(1 << 20)
            await ws.send_bytes(large)
            assert await ws.receive_bytes(timeout=REQUEST_SECONDS) == large[::-1]
            # Unanswered heartbeat pings would make aiohttp close the connection.
            await asyncio.sleep(1)
            await ws.send_str("still-there")
            assert await ws.receive_str(timeout=REQUEST_SECONDS) == "echo:still-there"
        await wait_closed(endpoint, [1000])

        async with http.ws_connect(f"{base}/ws", protocols=["chat"]) as ws:
            await ws.send_str("close-me")
            message = await ws.receive(timeout=REQUEST_SECONDS)
            assert message.type == aiohttp.WSMsgType.CLOSE and ws.close_code == 4000

        async with http.ws_connect(f"{base}/push", heartbeat=0.2) as ws:
            ticks = [await ws.receive_str(timeout=REQUEST_SECONDS) for _ in range(50)]
            assert ticks[-1] == "tick 49"

        with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
            await http.ws_connect(f"{base}/refused")
        assert refused.value.status == 403

        async def session(i: int) -> list[str]:
            async with http.ws_connect(f"{base}/ws", protocols=["chat"]) as ws:
                replies = []
                for j in range(5):
                    await ws.send_str(f"{i}-{j}")
                    replies.append(await ws.receive_str(timeout=REQUEST_SECONDS))
                return replies

        sessions = await asyncio.gather(*(session(i) for i in range(50)))
        assert sessions == [[f"echo:{i}-{j}" for j in range(5)] for i in range(50)]


async def test_a_vanished_websocket_client_is_an_abnormal_disconnect(endpoint: Endpoint) -> None:
    parts = urlsplit(endpoint.base)
    if parts.scheme == "https":
        pytest.skip("the raw-socket client speaks plain HTTP only")
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port or 80)
    writer.write(
        f"GET {parts.path}/ws HTTP/1.1\r\nHost: {parts.netloc}\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: chat\r\n\r\n".encode()
    )
    await writer.drain()
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), REQUEST_SECONDS)
    assert head.startswith(b"HTTP/1.1 101")
    writer.transport.abort()
    await wait_closed(endpoint, [1006])


async def test_drain_closes_open_websockets_with_going_away(endpoint: Endpoint) -> None:
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"]) as ws:
            await ws.send_str("before-drain")
            await ws.receive_str(timeout=REQUEST_SECONDS)
            runtime = endpoint.server._runtime  # type: ignore[attr-defined]
            draining = asyncio.ensure_future(runtime.drain(timeout=REQUEST_SECONDS))
            message = await ws.receive(timeout=REQUEST_SECONDS)
            assert message.type == aiohttp.WSMsgType.CLOSE and ws.close_code == 1001
            await asyncio.wait_for(draining, REQUEST_SECONDS)


async def wait_closed(endpoint: Endpoint, codes: list[int]) -> None:
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            while endpoint.closed[-len(codes) :] != codes:
                await asyncio.sleep(0.05)
    except TimeoutError:
        raise AssertionError(f"the app saw closes {endpoint.closed}, not {codes}") from None


async def test_drain_lets_a_streaming_response_finish(endpoint: Endpoint) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as http:
        async with http.get(f"{endpoint.base}/slow-stream") as response:
            first = await response.content.readline()
            runtime = endpoint.server._runtime  # type: ignore[attr-defined]
            draining = asyncio.ensure_future(runtime.drain(timeout=REQUEST_SECONDS))
            rest = await response.read()
        assert first + rest == "".join(f"chunk {i}\n" for i in range(10)).encode()
        await asyncio.wait_for(draining, REQUEST_SECONDS)


async def test_stopping_mid_stream_is_never_a_clean_body(endpoint: Endpoint) -> None:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as http:
        async with http.get(f"{endpoint.base}/endless") as response:
            assert await response.content.readline() == b"tick\n"
            closing = asyncio.ensure_future(endpoint.server.aclose())  # type: ignore[attr-defined]
            with pytest.raises(aiohttp.ClientPayloadError):
                await response.read()
        await asyncio.wait_for(closing, REQUEST_SECONDS)


async def test_the_node_authenticates_and_forwards_the_callers_token(endpoint: Endpoint) -> None:
    base, token = endpoint.base, endpoint.token
    bearer = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as http:
        async with http.get(f"{base}/private") as response:
            assert response.status == 401, "a non-public route needs a token"
        room = {"Authorization": f"Bearer {endpoint.room_token}"}
        async with http.get(f"{base}/private", headers=room) as response:
            assert response.status == 403, "a token without the call grant is refused"

        async with http.get(f"{base}/private", headers=bearer) as response:
            assert response.status == 200
            assert await response.json() == {
                "authorized": True,
                "authorization": f"Bearer {token}",
                "query": "",
            }
        async with http.get(f"{base}/private?access_token={token}&x=1") as response:
            assert response.status == 200
            assert await response.json() == {
                "authorized": True,
                "authorization": None,
                "query": f"access_token={token}&x=1",
            }

        with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
            await http.ws_connect(f"{base}/private-ws")
        assert refused.value.status == 401
        async with http.ws_connect(f"{base}/private-ws", headers=bearer) as ws:
            assert await ws.receive_json(timeout=REQUEST_SECONDS) == {
                "authorized": True,
                "authorization": f"Bearer {token}",
                "query": "",
            }
        # a browser cannot set headers on an upgrade: its token rides in the query
        async with http.ws_connect(f"{base}/private-ws?access_token={token}") as ws:
            assert await ws.receive_json(timeout=REQUEST_SECONDS) == {
                "authorized": True,
                "authorization": None,
                "query": f"access_token={token}",
            }
