"""The ASGI adapter (`livekit.agents.http`) behind a real worker: an unregistered `AgentServer`
serves its routes on the loopback public listener, the Rust worker parses each request and hands
the exchange to Starlette. Protocol behavior (framing, limits, deadlines, WebSocket closing) is
covered by the worker's own tests; these pin the mapping between exchanges and ASGI."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.websockets import WebSocket

from livekit.agents.http import AgentRoutes

pytestmark = pytest.mark.unit


def add_routes(http: AgentRoutes) -> None:
    @http.get("/hello/{name}")
    async def hello(request: Request) -> Response:
        info = request.state.lk_request
        return JSONResponse(
            {
                "name": request.path_params["name"],
                "query": request.url.query,
                "route": info.route,
                "authorized": info.authorized,
                "scheme": request.url.scheme,
            }
        )

    @http.post("/echo-size")
    async def echo_size(request: Request) -> Response:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        return PlainTextResponse(str(size))

    @http.get("/stream")
    async def stream(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            for i in range(3):
                yield f"chunk {i}\n".encode()
                await asyncio.sleep(0)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/boom")
    async def boom(request: Request) -> Response:
        raise RuntimeError("before any head")

    @http.get("/late-boom")
    async def late_boom(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"first chunk\n"
            raise RuntimeError("the source died")

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="chat")
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("text") is not None:
                await websocket.send_text("echo:" + message["text"])
            else:
                await websocket.send_bytes(bytes(reversed(message["bytes"])))

    @http.websocket("/refused")
    async def refused(websocket: WebSocket) -> None:
        await websocket.close()


@pytest.fixture
async def base_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    from livekit.agents import AgentHandler, AgentServer
    from livekit.agents.a2a import A2ASessionContext
    from livekit.agents.storage import SQLiteStorage

    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_WORKER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    server = AgentServer(
        agent_name="clinic",
        storage=SQLiteStorage(directory=tmp_path),
        listeners={"health": {"port": 0}},
    )
    add_routes(server.http)
    # A server serves at least one door.
    backend = AgentHandler(lane="backend")

    @backend.a2a_session(endpoint="noop", description="Unused.")
    async def noop(ctx: A2ASessionContext) -> str:  # noqa: ARG001
        return ""

    server.add(backend)
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        url = None
        for _ in range(200):
            with contextlib.suppress(RuntimeError):
                url = server._config.delegation_url("").rstrip("/")
                break
            if run.done():
                run.result()
            await asyncio.sleep(0.05)
        if url is None:
            raise TimeoutError("the public listener did not start")
        yield url
    finally:
        await server.aclose()
        await run


async def raw_get(base_url: str, path: str) -> bytes:
    """The response bytes as sent, trailers included."""
    host, port = base_url.removeprefix("http://").split("/", 1)[0].split(":")
    prefix = "/" + base_url.removeprefix("http://").split("/", 1)[1]
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(f"GET {prefix}{path} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
    await writer.drain()
    try:
        return await asyncio.wait_for(reader.read(-1), 5)
    finally:
        writer.close()


async def test_a_request_reaches_the_route_with_its_preamble(base_url: str) -> None:
    async with aiohttp.ClientSession() as http:
        async with http.get(f"{base_url}/hello/a%20b?x=1") as response:
            assert response.status == 200
            assert await response.json() == {
                "name": "a b",
                "query": "x=1",
                "route": "/hello/{name}",
                "authorized": True,
                "scheme": "http",
            }


async def test_bodies_stream_both_ways(base_url: str) -> None:
    async with aiohttp.ClientSession() as http:
        body = bytes(range(256)) * 4096
        async with http.post(f"{base_url}/echo-size", data=body) as response:
            assert await response.text() == str(len(body))
        async with http.get(f"{base_url}/stream") as response:
            assert response.headers["transfer-encoding"] == "chunked"
            assert await response.text() == "chunk 0\nchunk 1\nchunk 2\n"


async def test_application_failures_become_a_500_or_completion_trailers(base_url: str) -> None:
    raw = await raw_get(base_url, "/boom")
    assert raw.startswith(b"HTTP/1.1 500")

    raw = await raw_get(base_url, "/late-boom")
    assert raw.startswith(b"HTTP/1.1 200")
    assert b"first chunk\n" in raw
    trailers = raw.rsplit(b"\r\n0\r\n", 1)[1]
    assert b"x-lk-completion: internal\r\n" in trailers
    assert b"the source died" in trailers


async def test_a_websocket_echoes_and_a_refused_upgrade_is_a_403(base_url: str) -> None:
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"{base_url}/ws", protocols=["chat"]) as ws:
            assert ws.protocol == "chat"
            await ws.send_str("hi")
            assert await ws.receive_str() == "echo:hi"
            await ws.send_bytes(b"abc")
            assert await ws.receive_bytes() == b"cba"
        with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
            await http.ws_connect(f"{base_url}/refused")
        assert refused.value.status == 403


def test_the_manifest_carries_methods_kinds_and_public() -> None:
    from livekit.agents._generated.worker_server import EndpointRoute

    http = AgentRoutes()
    add_routes(http)

    async def raw(scope, receive, send) -> None:  # type: ignore[no-untyped-def]
        return

    http.mount("/raw", raw)
    http.configure_endpoint("/echo-size", public=True)
    by_path = {route.path: route for route in http.manifest()}
    assert by_path["/hello/{name}"] == EndpointRoute(
        path="/hello/{name}", methods=["GET", "HEAD"], kind="http", public=False
    )
    assert by_path["/echo-size"].methods == ["POST"] and by_path["/echo-size"].public
    assert by_path["/ws"] == EndpointRoute(path="/ws", methods=["GET"], kind="http", public=False)
    assert by_path["/raw/{path:path}"].methods == [
        "GET",
        "HEAD",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "OPTIONS",
        "TRACE",
    ]
    assert [route.path for route in http.manifest()][0] == "/hello/{name}", "order is preserved"
