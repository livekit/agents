"""HTTP and WebSocket routes through the LiveKit node: a registered worker serves them over the
endpoint data plane and clients call the node's /agents/{agent}/{deployment}/ front. Against a
local development node, set LIVEKIT_WORKER_INSECURE=1 for its self-signed WebTransport identity.

The console journey needs no node: a worker delegates to its own endpoint over loopback."""

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

from livekit import api
from livekit.agents import Agent, AgentHandler, AgentServer, APIConnectOptions, JobContext
from livekit.agents.a2a import A2ASessionContext, Credentials, RemoteAgent
from livekit.agents.storage import SQLiteStorage
from tests.fakes import FakeLLM, tool_call

from .resources import REQUEST_SECONDS, STARTUP_SECONDS, serving


@dataclass
class Endpoint:
    base: str
    server: AgentServer
    # a project token: the agentEndpoint call grant the node admits non-public routes with
    token: str
    # signed by the project, without the call grant
    room_token: str
    # the close code of each WebSocket the application saw end
    closes: asyncio.Queue[int] = field(default_factory=asyncio.Queue)


def seen_by_the_app(scope: Request | WebSocket) -> dict[str, object]:
    """What reached the application: the node's verdict and the caller's token as sent."""
    return {
        "authorized": scope.state.lk_request.authorized,
        "authorization": scope.headers.get("authorization"),
        "query": scope.url.query,
    }


def add_routes(endpoint: Endpoint) -> None:
    http = endpoint.server.http

    @http.get("/hello", public=True)
    async def hello(request: Request) -> Response:
        return JSONResponse({"route": request.state.lk_request.route, "query": request.url.query})

    @http.post("/echo-size", public=True)
    async def echo_size(request: Request) -> Response:
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        return PlainTextResponse(str(size))

    def paced(lines: list[bytes], pause: float) -> StreamingResponse:
        async def chunks() -> AsyncIterator[bytes]:
            for line in lines:
                yield line
                await asyncio.sleep(pause)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/stream", public=True)
    async def stream(request: Request) -> Response:
        return paced([f"chunk {i}\n".encode() for i in range(5)], 0.02)

    @http.get("/slow-stream", public=True)
    async def slow_stream(request: Request) -> Response:
        return paced([f"chunk {i}\n".encode() for i in range(10)], 0.1)

    @http.get("/late-boom", public=True)
    async def late_boom(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"first chunk\n"
            await asyncio.sleep(0.05)
            raise RuntimeError("the source died")

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.get("/endless", public=True)
    async def endless(request: Request) -> Response:
        async def chunks() -> AsyncIterator[bytes]:
            while True:
                yield b"tick\n"
                await asyncio.sleep(0.05)

        return StreamingResponse(chunks(), media_type="text/plain")

    @http.websocket("/ws", public=True)
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="chat")
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    endpoint.closes.put_nowait(message["code"])
                    return
                if message.get("text") is not None:
                    if message["text"] == "close-me":
                        await websocket.close(code=4000, reason="bye")
                        return
                    await websocket.send_text("echo:" + message["text"])
                else:
                    await websocket.send_bytes(bytes(reversed(message["bytes"])))
        except WebSocketDisconnect as error:
            endpoint.closes.put_nowait(error.code)

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


async def _entrypoint(ctx: JobContext) -> None:
    """The voice door's entrypoint; these journeys dispatch no job."""


@pytest.fixture
async def endpoint(
    live_credentials: tuple[str, str, str], tmp_path: Path
) -> AsyncIterator[Endpoint]:
    """A worker registered with the live node, serving ``add_routes`` once the node routes."""
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
        async with aiohttp.ClientSession() as http, asyncio.timeout(STARTUP_SECONDS):
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
        await run


@pytest.fixture
async def http() -> AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(REQUEST_SECONDS)) as session:
        yield session


# -- HTTP


async def test_a_request_reaches_its_route_with_its_query(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A GET through the node reaches the handler with its route and query string."""
    async with http.get(f"{endpoint.base}/hello?x=1") as response:
        body = await response.json()

    assert body == {"route": "/hello", "query": "x=1"}


async def upload() -> AsyncIterator[bytes]:
    for _ in range(10):
        yield b"x" * 1000


@pytest.mark.parametrize(
    ("body", "size"),
    [(os.urandom(3 << 20), 3 << 20), (upload, 10_000)],
    ids=["3MiB", "chunked"],
)
async def test_a_request_body_arrives_whole(
    endpoint: Endpoint, http: aiohttp.ClientSession, body: object, size: int
) -> None:
    """A large body and a chunked upload both reach the handler byte for byte."""
    data = body() if callable(body) else body

    async with http.post(f"{endpoint.base}/echo-size", data=data) as response:
        received = await response.text()

    assert received == str(size)


async def test_a_streamed_response_arrives_whole(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A streaming response's chunks all reach the caller, in order."""
    async with http.get(f"{endpoint.base}/stream") as response:
        body = await response.text()

    assert body == "".join(f"chunk {i}\n" for i in range(5))


async def test_a_failure_after_the_response_head_is_never_a_clean_body(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A stream that raises after its first chunk reaches the caller as a broken body."""
    with pytest.raises(aiohttp.ClientPayloadError):
        async with http.get(f"{endpoint.base}/late-boom") as response:
            await response.read()


async def test_concurrent_requests_each_get_their_own_answer(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A hundred requests in flight at once are answered without crossing."""

    async def query(i: int) -> str:
        async with http.get(f"{endpoint.base}/hello?x={i}") as response:
            body: dict[str, str] = await response.json()
            return body["query"]

    queries = await asyncio.gather(*(query(i) for i in range(100)))

    assert queries == [f"x={i}" for i in range(100)]


# -- WebSocket


async def test_a_websocket_echoes_text_and_binary_with_its_subprotocol(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """Text, bytes and a 1 MiB frame cross both ways on the negotiated subprotocol."""
    large = os.urandom(1 << 20)

    async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"]) as ws:
        await ws.send_str("hi")
        text = await ws.receive_str()
        await ws.send_bytes(b"abc")
        small = await ws.receive_bytes()
        await ws.send_bytes(large)
        echoed = await ws.receive_bytes()

    assert ws.protocol == "chat"
    assert (text, small, echoed) == ("echo:hi", b"cba", large[::-1])


async def test_a_websocket_outlives_heartbeats_and_closes_cleanly(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """The worker answers the client's pings across an idle second, and the app sees 1000."""
    async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"], heartbeat=0.2) as ws:
        await asyncio.sleep(1)  # idle: unanswered pings would close the connection
        await ws.send_str("still-there")
        reply = await ws.receive_str()

    assert reply == "echo:still-there"
    assert await endpoint.closes.get() == 1000


async def test_the_apps_close_code_reaches_the_client(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A WebSocket the application closes with 4000 closes the client with 4000."""
    async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"]) as ws:
        await ws.send_str("close-me")
        message = await ws.receive()

    assert message.type == aiohttp.WSMsgType.CLOSE
    assert ws.close_code == 4000


async def test_a_pushing_websocket_keeps_streaming_through_heartbeats(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A route that only sends keeps delivering while the client pings it."""
    async with http.ws_connect(f"{endpoint.base}/push", heartbeat=0.2) as ws:
        ticks = [await ws.receive_str() for _ in range(50)]

    assert ticks == [f"tick {i}" for i in range(50)]


async def test_a_websocket_the_app_refuses_is_a_403(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """Closing before accept refuses the upgrade with 403."""
    with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
        await http.ws_connect(f"{endpoint.base}/refused")

    assert refused.value.status == 403


async def test_concurrent_websockets_each_get_their_own_echoes(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """Fifty WebSockets open at once each receive only their own replies."""

    async def session(i: int) -> list[str]:
        async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"]) as ws:
            replies = []
            for j in range(5):
                await ws.send_str(f"{i}-{j}")
                replies.append(await ws.receive_str())
            return replies

    sessions = await asyncio.gather(*(session(i) for i in range(50)))

    assert sessions == [[f"echo:{i}-{j}" for j in range(5)] for i in range(50)]


async def test_a_vanished_websocket_client_is_an_abnormal_disconnect(endpoint: Endpoint) -> None:
    """A client whose socket is aborted after the upgrade reaches the app as close code 1006."""
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
    assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 101")

    writer.transport.abort()

    assert await endpoint.closes.get() == 1006


# -- drain and shutdown


async def test_drain_closes_open_websockets_with_going_away(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """Draining the worker closes an open WebSocket with 1001."""
    async with http.ws_connect(f"{endpoint.base}/ws", protocols=["chat"]) as ws:
        draining = asyncio.ensure_future(endpoint.server._drain(REQUEST_SECONDS))
        message = await ws.receive()
    await draining

    assert message.type == aiohttp.WSMsgType.CLOSE
    assert ws.close_code == 1001


async def test_drain_lets_a_streaming_response_finish(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A response streaming when the drain starts still ends whole."""
    async with http.get(f"{endpoint.base}/slow-stream") as response:
        first = await response.content.readline()
        draining = asyncio.ensure_future(endpoint.server._drain(REQUEST_SECONDS))
        rest = await response.read()
    await draining

    assert first + rest == "".join(f"chunk {i}\n" for i in range(10)).encode()


async def test_stopping_mid_stream_is_never_a_clean_body(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """Closing the worker under an endless response breaks the caller's body."""
    async with http.get(f"{endpoint.base}/endless") as response:
        await response.content.readline()
        closing = asyncio.ensure_future(endpoint.server.aclose())
        with pytest.raises(aiohttp.ClientPayloadError):
            await response.read()
    await closing


# -- authentication


async def test_a_private_route_refuses_a_caller_without_the_call_grant(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """No token is 401 and a token without the endpoint call grant is 403, on HTTP and WS."""
    room = {"Authorization": f"Bearer {endpoint.room_token}"}

    async with http.get(f"{endpoint.base}/private") as anonymous:
        pass
    async with http.get(f"{endpoint.base}/private", headers=room) as ungranted:
        pass
    with pytest.raises(aiohttp.WSServerHandshakeError) as refused:
        await http.ws_connect(f"{endpoint.base}/private-ws")

    assert (anonymous.status, ungranted.status, refused.value.status) == (401, 403, 401)


async def test_a_private_route_receives_the_callers_bearer_token(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """The node admits a granted bearer token and forwards it to the app unchanged."""
    bearer = {"Authorization": f"Bearer {endpoint.token}"}
    forwarded = {"authorized": True, "authorization": f"Bearer {endpoint.token}", "query": ""}

    async with http.get(f"{endpoint.base}/private", headers=bearer) as response:
        over_http = await response.json()
    async with http.ws_connect(f"{endpoint.base}/private-ws", headers=bearer) as ws:
        over_ws = await ws.receive_json()

    assert over_http == over_ws == forwarded


async def test_a_private_route_receives_a_token_in_the_query(
    endpoint: Endpoint, http: aiohttp.ClientSession
) -> None:
    """A browser's token rides in the query (it cannot set upgrade headers) and is admitted."""
    query = f"access_token={endpoint.token}"

    async with http.get(f"{endpoint.base}/private?{query}&x=1") as response:
        over_http = await response.json()
    async with http.ws_connect(f"{endpoint.base}/private-ws?{query}") as ws:
        over_ws = await ws.receive_json()

    assert over_http == {"authorized": True, "authorization": None, "query": f"{query}&x=1"}
    assert over_ws == {"authorized": True, "authorization": None, "query": query}


# -- console


async def test_a_console_worker_delegates_to_its_own_endpoint_over_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no LiveKit credentials, a delegation reaches the worker's own text door on loopback
    (not Cloud, where nothing serves it)."""
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_WORKER_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    received: list[str] = []
    back_office = AgentHandler()

    @back_office.a2a_session(
        partition_key="backend", endpoint="coverage_expert", description="Checks account coverage."
    )
    async def text(ctx: A2ASessionContext) -> str:
        received.append(ctx.message.text_content or "")
        return "covered"

    server = AgentServer(agent_name="clinic", storage=SQLiteStorage(directory=tmp_path))
    server.add(back_office)
    remote = RemoteAgent(text, conn_options=APIConnectOptions(max_retry=0, read_timeout=2.0))
    model = FakeLLM({"Check": tool_call("coverage_expert", request="Check"), "covered": ""})
    agent = Agent(instructions="Call coverage_expert.", tools=[remote.as_tool()], llm=model)

    async with serving(server):
        await agent.run("Check")

    assert received == ["Check"]
