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
    closed: list[int] = field(default_factory=list)


def add_routes(endpoint: Endpoint) -> None:
    http = endpoint.server.http  # type: ignore[attr-defined]

    @http.get("/hello")
    async def hello(request: Request) -> Response:
        return JSONResponse({"route": request.state.lk_request.route, "query": request.url.query})

    @http.post("/echo-size")
    async def echo_size(request: Request) -> Response:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        return PlainTextResponse(str(size))

    @http.get("/stream")
    async def stream(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            for i in range(5):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0.02)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/late-boom")
    async def late_boom(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"first chunk\n"
            await asyncio.sleep(0.05)
            raise RuntimeError("the source died")

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/ws")
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

    @http.get("/slow-stream")
    async def slow_stream(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            for i in range(10):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0.1)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/endless")
    async def endless(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            while True:
                yield b"tick\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/push")
    async def push(websocket: WebSocket) -> None:
        # Sends only: the worker still answers pings and notices the close.
        await websocket.accept()
        for i in range(1000):
            await websocket.send_text(f"tick {i}")
            await asyncio.sleep(0.01)

    @http.websocket("/refused")
    async def refused(websocket: WebSocket) -> None:
        await websocket.close()

    paths = ("/hello", "/echo-size", "/stream", "/slow-stream", "/endless", "/late-boom")
    for path in (*paths, "/ws", "/push", "/refused"):
        http.configure_endpoint(path, public=True)


@pytest.fixture
async def endpoint(
    live_credentials: tuple[str, str, str], tmp_path: Path
) -> AsyncIterator[Endpoint]:
    from livekit.agents import AgentHandler, AgentServer
    from livekit.agents.a2a import A2ASessionContext
    from livekit.agents.storage import SQLiteStorage

    url, _, _ = live_credentials
    name = "e2e-endpoints-" + uuid.uuid4().hex[:8]
    parts = urlsplit(url)
    scheme = "https" if parts.scheme in ("https", "wss") else "http"
    server = AgentServer(
        agent_name=name,
        storage=SQLiteStorage(directory=tmp_path),
        listeners={"health": {"port": 0}},
    )
    endpoint = Endpoint(f"{scheme}://{parts.netloc}/agents/{name}/_", server)
    add_routes(endpoint)
    backend = AgentHandler(lane="backend")

    @backend.a2a_session(endpoint="noop", description="Unused.")
    async def noop(ctx: A2ASessionContext) -> str:  # noqa: ARG001
        return ""

    server.add(backend)
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
    async with asyncio.timeout(REQUEST_SECONDS):
        while endpoint.closed[-len(codes) :] != codes:
            await asyncio.sleep(0.05)


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
