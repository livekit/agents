from __future__ import annotations

import asyncio
import contextlib
import statistics
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import urlparse

import aiohttp
import pytest
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import StreamingResponse

from livekit.agents import AgentServer, JobContext
from livekit.agents.http import _HttpRunner, _proxied_endpoints
from livekit.agents.tunnel import WebSocketTunnel
from livekit.agents.tunnel._base import CHUNK
from livekit.agents.tunnel._websocket import _CONN_WINDOW, _STREAM_WINDOW, Mux, Stream
from livekit.protocol.agent_proxy import (
    AgentHttpEof,
    AgentHttpFrame,
    AgentHttpOpen,
    AgentHttpRegistration,
    AgentHttpReset,
)

from .tunnel_controller import LocalController

pytestmark = pytest.mark.unit

API_KEY = "devkey"
API_SECRET = "devsecretdevsecretdevsecretdevsecret"  # noqa: S105
WIRES = 2


def _make_app() -> AgentServer:
    server = AgentServer(ws_url="ws://localhost:7880", api_key="key", api_secret="secret")

    @server.rtc_session()
    async def _entrypoint(ctx: JobContext) -> None:
        pass

    server._proc_pool = SimpleNamespace(processes=[])  # type: ignore[assignment]

    @server.http.get("/ping")
    async def ping() -> dict:
        return {"ok": True}

    @server.http.post("/upload")
    async def upload(request: Request) -> dict:
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
        return {"bytes": total}

    @server.http.get("/bytes/{n}")
    async def bytes_n(n: int) -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            sent = 0
            while sent < n:
                size = min(65536, n - sent)
                yield b"x" * size
                sent += size

        return StreamingResponse(gen(), media_type="application/octet-stream")

    @server.http.get("/sse/{n}")
    async def sse(n: int) -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            for i in range(n):
                yield f"data: {i}\n\n".encode()
                await asyncio.sleep(0.1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @server.http.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        with contextlib.suppress(Exception):
            while True:
                await sock.send_text(f"echo:{await sock.receive_text()}")

    return server


@asynccontextmanager
async def _proxied(
    wires: int = WIRES,
) -> AsyncIterator[tuple[str, WebSocketTunnel, LocalController]]:
    """Bring up app, controller and tunnel; yield the public url and both halves."""
    server = _make_app()
    runner = _HttpRunner(server.http, host="127.0.0.1", port=0)
    await runner.start()

    controller = LocalController(api_key=API_KEY, api_secret=API_SECRET)
    await controller.start()
    tunnel = WebSocketTunnel(
        ws_url=controller.ws_url, api_key=API_KEY, api_secret=API_SECRET, wires=wires
    )
    await tunnel.start(target_port=runner.port, endpoints=_proxied_endpoints(server.http))
    await controller.wait_ready()

    try:
        yield controller.base_url, tunnel, controller
    finally:
        await tunnel.aclose()
        await controller.aclose()
        await runner.aclose()


async def _settles(predicate: Callable[[], bool], timeout: float = 8.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return False


async def _drain_sse(base: str) -> int:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as sess:
        async with sess.get(f"{base}/sse/5") as resp:
            return len([ln async for ln in resp.content if ln.strip()])


def _wire_streams(controller: LocalController) -> list[int]:
    return [mux.open_streams for muxes in controller._workers.values() for mux in muxes]


# --------------------------------------------------------------------- through the tunnel


async def test_json_round_trip() -> None:
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/ping") as resp:
            assert resp.status == 200
            assert await resp.json() == {"ok": True}


async def test_large_upload_is_byte_exact() -> None:
    """12 MiB crosses the 8 MiB wire window, so it finishes only if credit comes back."""
    payload = b"z" * (12 * 1024 * 1024)
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.post(f"{base}/upload", data=payload) as resp:
            assert resp.status == 200
            assert (await resp.json())["bytes"] == len(payload)


async def test_large_download_is_byte_exact() -> None:
    """The same crossing in the other direction."""
    size = 12 * 1024 * 1024
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/bytes/{size}") as resp:
            assert resp.status == 200
            assert len(await resp.read()) == size


async def test_sse_is_not_buffered() -> None:
    loop = asyncio.get_running_loop()
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.get(f"{base}/sse/5") as resp:
            assert resp.status == 200
            arrivals = []
            async for line in resp.content:
                if line.strip():
                    arrivals.append(loop.time())
    assert len(arrivals) == 5
    # a buffered response would deliver every event at once
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:], strict=False)]
    assert statistics.median(gaps) > 0.03, f"events arrived together: {gaps}"


async def test_websocket_round_trips() -> None:
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"{base}/ws") as sock:
            for token in ("a", "b", "c"):
                await sock.send_str(token)
                assert (await sock.receive()).data == f"echo:{token}"


async def test_concurrent_requests() -> None:
    async with _proxied() as (base, _, controller):

        async def one() -> int:
            # own connection per request, so each becomes its own proxied stream
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            ) as sess:
                async with sess.get(f"{base}/ping") as resp:
                    return resp.status

        statuses = await asyncio.gather(*(one() for _ in range(32)))
        assert statuses == [200] * 32
        # its own connection each, so each had to become its own stream
        assert controller.streams_opened >= 32


async def test_websocket_survives_other_traffic() -> None:
    """A long-lived stream must not block the others, nor be blocked by them."""
    async with _proxied() as (base, _, _e), aiohttp.ClientSession() as sess:
        async with sess.ws_connect(f"{base}/ws") as sock:
            await sock.send_str("before")
            assert (await sock.receive()).data == "echo:before"

            slow = asyncio.create_task(_drain_sse(base))
            await asyncio.sleep(0.15)
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            ) as small:
                async with small.get(f"{base}/ping") as resp:
                    assert resp.status == 200
            assert not slow.done(), "the small request waited for the slow stream"

            assert await slow == 5
            await sock.send_str("after")
            assert (await sock.receive()).data == "echo:after"


async def test_streams_spread_over_the_wires() -> None:
    """Holding several wires only pays off if streams land on different ones."""
    async with _proxied() as (base, tunnel, controller):
        assert tunnel.wire_count == WIRES
        assert await _settles(lambda: controller.wire_count == WIRES)

        streams = [asyncio.create_task(_drain_sse(base)) for _ in range(4)]
        spread = await _settles(lambda: sorted(_wire_streams(controller)) == [2, 2])
        assert await asyncio.gather(*streams) == [5] * 4
        assert spread, f"four equal streams landed as {_wire_streams(controller)}"


async def test_teardown_in_any_order_finishes() -> None:
    """A byte carrier has no half-close, so closing the controller first must not park forever."""
    server = _make_app()
    runner = _HttpRunner(server.http, host="127.0.0.1", port=0)
    await runner.start()
    controller = LocalController(api_key=API_KEY, api_secret=API_SECRET)
    await controller.start()
    tunnel = WebSocketTunnel(
        ws_url=controller.ws_url, api_key=API_KEY, api_secret=API_SECRET, wires=WIRES
    )
    await tunnel.start(target_port=runner.port, endpoints=_proxied_endpoints(server.http))
    await controller.wait_ready()

    async with aiohttp.ClientSession() as sess, sess.get(f"{controller.base_url}/ping") as resp:
        assert resp.status == 200

    # deliberately the reverse of the order the fixture uses
    await asyncio.wait_for(controller.aclose(), timeout=5)
    await asyncio.wait_for(tunnel.aclose(), timeout=5)
    await runner.aclose()


async def test_client_is_released_when_the_wire_dies() -> None:
    """A cut wire must disconnect the client, not leave it waiting for a reply."""
    server = _make_app()
    runner = _HttpRunner(server.http, host="127.0.0.1", port=0)
    await runner.start()
    controller = LocalController(api_key=API_KEY, api_secret=API_SECRET)
    await controller.start()
    tunnel = WebSocketTunnel(
        ws_url=controller.ws_url, api_key=API_KEY, api_secret=API_SECRET, wires=WIRES
    )
    await tunnel.start(target_port=runner.port, endpoints=_proxied_endpoints(server.http))
    await controller.wait_ready()

    try:
        slow = asyncio.create_task(_drain_sse(controller.base_url))
        await asyncio.sleep(0.2)
        await tunnel.aclose()  # the worker vanishes mid-response

        # asyncio.wait does not cancel, so a still-pending task means a real hang
        done, _ = await asyncio.wait({slow}, timeout=3)
        assert done, "client was left hanging after the wire died"
        slow.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await slow
    finally:
        await controller.aclose()
        await runner.aclose()


async def test_a_reader_that_stops_does_not_stop_the_wire() -> None:
    """The whole point of crediting on drain: one stalled client, one stalled stream."""
    async with _proxied(wires=1) as (base, _, _e):
        url = urlparse(base)
        # a client that asks for a large body and then reads none of it
        _reader, writer = await asyncio.open_connection(url.hostname, url.port)
        writer.write(b"GET /bytes/8388608 HTTP/1.1\r\nHost: controller\r\n\r\n")
        await writer.drain()
        await asyncio.sleep(0.5)  # long enough to fill its window

        try:
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            ) as sess:
                async with sess.get(f"{base}/ping", timeout=aiohttp.ClientTimeout(5)) as resp:
                    assert resp.status == 200, "the wire was blocked by the stalled reader"
        finally:
            writer.close()


# ------------------------------------------------------------------------- advertisement


def test_endpoints_are_the_first_segment_of_each_route() -> None:
    """What the controller matches on, so a route that cannot name one must not be advertised."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def health() -> dict:  # the built-in health route, which stays local
        return {}

    @app.get("/orders/{order_id}")
    async def order(order_id: str) -> dict:
        return {}

    @app.post("/orders")
    async def create() -> dict:  # the same endpoint as the route above
        return {}

    @app.get("/{anything}/tail")
    async def wildcard(anything: str) -> dict:  # a parameter names nothing
        return {}

    assert _proxied_endpoints(app) == ["orders"]


async def test_the_wire_announces_its_endpoints_before_any_request() -> None:
    """Whatever the router declares, FastAPI's own documentation routes included."""
    async with _proxied() as (_base, tunnel, controller):
        assert controller.endpoints_of(tunnel.worker_id) == {
            "ping",
            "upload",
            "bytes",
            "sse",
            "ws",
            "docs",
            "redoc",
            "openapi.json",
        }


async def test_an_unannounced_endpoint_is_refused() -> None:
    """The controller answers for an unclaimed endpoint rather than leasing a wire."""
    async with _proxied() as (base, _, controller), aiohttp.ClientSession() as sess:
        opened = controller.streams_opened
        async with sess.get(f"{base}/nope/at/all") as resp:
            assert resp.status == 503
        assert controller.streams_opened == opened, "a wire was leased for an unknown endpoint"


# ------------------------------------------------------------------------------- the mux


class _FakeWs:
    """A websocket that records what a mux sends and replays what a test feeds it."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        self.closed = False

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def receive(self) -> object:
        return await self._inbox.get()

    async def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    def feed(self, frame: AgentHttpFrame) -> None:
        msg = SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=frame.SerializeToString())
        self._inbox.put_nowait(msg)

    def frames(self) -> list[AgentHttpFrame]:
        out = []
        for buf in self.sent:
            frame = AgentHttpFrame()
            frame.ParseFromString(buf)
            out.append(frame)
        return out

    def kinds(self) -> list[str | None]:
        return [frame.WhichOneof("message") for frame in self.frames()]

    def data_bytes(self, sid: int) -> int:
        return sum(
            len(f.data)
            for f in self.frames()
            if f.WhichOneof("message") == "data" and f.stream_id == sid
        )


@asynccontextmanager
async def _fake_mux() -> AsyncIterator[tuple[Mux, _FakeWs]]:
    ws = _FakeWs()
    mux = Mux(ws, opener=True)  # type: ignore[arg-type]
    await mux.start()
    try:
        yield mux, ws
    finally:
        await mux.aclose()


def test_every_frame_kind_survives_the_wire() -> None:
    """One message carries all six, so an empty data frame is not an end of stream."""
    cases = [
        AgentHttpFrame(registration=AgentHttpRegistration(worker_id="AHW_1", endpoints=["a"])),
        AgentHttpFrame(stream_id=7, open=AgentHttpOpen(request_id="req_42")),
        AgentHttpFrame(stream_id=9, data=bytes(range(256)) * 4),
        AgentHttpFrame(stream_id=9, data=b""),  # the empty write a half-closing app makes
        AgentHttpFrame(stream_id=11, eof=AgentHttpEof()),
        AgentHttpFrame(stream_id=13, reset=AgentHttpReset()),
        AgentHttpFrame(credit=8 << 20),  # stream 0 addresses the wire itself
        AgentHttpFrame(stream_id=15, credit=1 << 20),
    ]
    for sent in cases:
        got = AgentHttpFrame()
        got.ParseFromString(sent.SerializeToString())
        assert got == sent
        assert got.WhichOneof("message") == sent.WhichOneof("message")


async def test_a_sender_is_held_by_the_window_it_was_given() -> None:
    """Without this a large upload is buffered by the receiver without bound."""
    async with _fake_mux() as (mux, ws):
        sid = await mux.open_stream("req_1")
        body = b"x" * (2 * _STREAM_WINDOW)
        sender = asyncio.create_task(mux.send(sid, body))

        assert await _settles(lambda: ws.data_bytes(sid) == _STREAM_WINDOW)
        await asyncio.sleep(0.05)
        assert not sender.done(), "the sender ran past the window it was given"
        assert ws.data_bytes(sid) == _STREAM_WINDOW

        ws.feed(AgentHttpFrame(stream_id=sid, credit=_STREAM_WINDOW))
        await asyncio.wait_for(sender, 5)
        assert ws.data_bytes(sid) == len(body)


async def test_the_wire_window_holds_a_sender_the_stream_window_would_not() -> None:
    """Per-stream credit alone lets K streams overrun the wire, so the wire has its own."""
    async with _fake_mux() as (mux, ws):
        streams = [await mux.open_stream(f"req_{i}") for i in range(12)]
        body = b"x" * _STREAM_WINDOW
        senders = [asyncio.create_task(mux.send(sid, body)) for sid in streams]

        # every stream is inside its own window, so only the wire's can stop them
        assert await _settles(lambda: sum(ws.data_bytes(sid) for sid in streams) == _CONN_WINDOW)
        await asyncio.sleep(0.05)
        assert sum(ws.data_bytes(sid) for sid in streams) == _CONN_WINDOW
        assert any(not sender.done() for sender in senders)

        for sender in senders:
            sender.cancel()
        await asyncio.gather(*senders, return_exceptions=True)


async def test_wire_credit_releases_the_senders_it_parked() -> None:
    """What a transfer larger than the wire window rides on: parked senders resume."""
    async with _fake_mux() as (mux, ws):
        sid = await mux.open_stream("req_1")
        body = b"x" * (_CONN_WINDOW + _STREAM_WINDOW)
        # the stream window would park this first, so it is credited out of the way
        streaming = asyncio.create_task(_credit_stream(mux, ws, sid, len(body)))
        sender = asyncio.create_task(mux.send(sid, body))

        assert await _settles(lambda: ws.data_bytes(sid) == _CONN_WINDOW)
        await asyncio.sleep(0.05)
        assert not sender.done(), "the sender ran past the wire window"

        ws.feed(AgentHttpFrame(credit=_STREAM_WINDOW))  # stream 0 is the wire itself
        await asyncio.wait_for(sender, 5)
        assert ws.data_bytes(sid) == len(body)
        streaming.cancel()
        await asyncio.gather(streaming, return_exceptions=True)


async def _credit_stream(mux: Mux, ws: _FakeWs, sid: int, total: int) -> None:
    """Keep one stream's window open, so only the wire's window can park its sender."""
    while True:
        if ws.data_bytes(sid) >= total:
            return
        ws.feed(AgentHttpFrame(stream_id=sid, credit=_STREAM_WINDOW))
        await asyncio.sleep(0.01)


async def test_eof_is_a_half_close_not_a_teardown() -> None:
    """Done writing, still reading: what a request body ending mid-response needs."""
    async with _fake_mux() as (mux, ws):
        sid = await mux.open_stream("req_1")
        stream = Stream(mux, sid)

        await stream.write_eof()
        assert ws.kinds()[-1] == "eof"

        # the reply direction is untouched by this side's EOF
        ws.feed(AgentHttpFrame(stream_id=sid, data=b"reply"))
        assert await asyncio.wait_for(stream.read(), 5) == b"reply"

        ws.feed(AgentHttpFrame(stream_id=sid, eof=AgentHttpEof()))
        assert await asyncio.wait_for(stream.read(), 5) == b""


async def test_credit_waits_for_the_reader() -> None:
    """Credit on arrival would hand the window back before anything had consumed it."""
    async with _fake_mux() as (mux, ws):
        sid = await mux.open_stream("req_1")
        chunk = b"x" * CHUNK
        for _ in range(_STREAM_WINDOW // CHUNK):
            ws.feed(AgentHttpFrame(stream_id=sid, data=chunk))

        assert await _settles(lambda: mux._undrained[sid] == _STREAM_WINDOW)
        assert "credit" not in ws.kinds(), "the window came back before anything read it"

        stream = Stream(mux, sid)
        drained = 0
        while drained < _STREAM_WINDOW:
            drained += len(await asyncio.wait_for(stream.read(), 5))
        assert ws.kinds().count("credit") >= 1, "reading gave no window back"


async def test_a_dropped_stream_gives_its_window_back() -> None:
    """Bytes nobody will read still cost the peer, so the wire has to be repaid."""
    async with _fake_mux() as (mux, ws):
        sid = await mux.open_stream("req_1")
        ws.feed(AgentHttpFrame(stream_id=sid, data=b"x" * CHUNK))
        assert await _settles(lambda: mux._undrained[sid] == CHUNK)

        mux.forget(sid)
        assert mux._conn_owed == CHUNK, "the wire lost the window that stream was holding"


async def test_data_for_a_forgotten_stream_is_reset_once() -> None:
    """Silence would leave the peer spending its window on a stream that never credits back."""
    async with _fake_mux() as (mux, ws):
        for _ in range(3):
            ws.feed(AgentHttpFrame(stream_id=99, data=b"orphan"))
        assert await _settles(lambda: ws.kinds().count("reset") == 1)
        await asyncio.sleep(0.05)
        assert ws.kinds().count("reset") == 1
