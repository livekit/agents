"""A local stand-in for the cloud controller, so a test can drive a tunnel without one.

It mirrors ``cloud/pkg/agent/httpproxy``, but parses no HTTP: it reads the request line
far enough to find the endpoint and hands the bytes over unread, so a test exercises the
tunnel and not a second HTTP implementation.
"""

from __future__ import annotations

import asyncio
import contextlib

from aiohttp import web

from livekit import api
from livekit.agents.log import logger
from livekit.agents.tunnel._base import CHUNK
from livekit.agents.tunnel._websocket import _MAX_MESSAGE, TUNNEL_PATH, Mux, Stream
from livekit.protocol.agent_proxy import AgentHttpFrame


class LocalController:
    """Accepts a worker's wires on one port, and clients on another."""

    def __init__(self, *, api_key: str, api_secret: str) -> None:
        self._verifier = api.TokenVerifier(api_key, api_secret)
        self._workers: dict[str, list[Mux]] = {}
        self._endpoints: dict[str, set[str]] = {}
        self._ready = asyncio.Event()
        self._runner: web.AppRunner | None = None
        self._clients: asyncio.Server | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()
        self._ws_url = ""
        self._base_url = ""
        self._opened = 0
        self._closing = False

    @property
    def ws_url(self) -> str:
        """What a worker is given as its LiveKit url."""
        return self._ws_url

    @property
    def base_url(self) -> str:
        """Where a client sends requests."""
        return self._base_url

    @property
    def wire_count(self) -> int:
        return sum(len([m for m in muxes if not m.closed]) for muxes in self._workers.values())

    @property
    def streams_opened(self) -> int:
        """Streams leased to workers, which is what a request costs when it is routed."""
        return self._opened

    def endpoints_of(self, worker_id: str) -> set[str]:
        return self._endpoints.get(worker_id, set())

    async def start(self, host: str = "127.0.0.1") -> None:
        app = web.Application()
        app.router.add_get(TUNNEL_PATH, self._on_wire)
        self._runner = web.AppRunner(app, shutdown_timeout=1.0)
        await self._runner.setup()
        site = web.TCPSite(self._runner, host, 0)
        await site.start()
        self._ws_url = f"ws://{host}:{self._runner.addresses[0][1]}"

        self._clients = await asyncio.start_server(self._on_client, host, 0)
        self._base_url = f"http://{host}:{self._clients.sockets[0].getsockname()[1]}"

    async def wait_ready(self, timeout: float = 10.0) -> None:
        """Returns once one wire is up; the rest land behind it."""
        await asyncio.wait_for(self._ready.wait(), timeout)

    async def aclose(self) -> None:
        self._closing = True
        self._ready.clear()

        # cancelled first, or wait_closed() waits on clients parked in stream.read()
        for task in list(self._client_tasks):
            task.cancel()
        if self._clients is not None:
            self._clients.close()
            with contextlib.suppress(Exception):
                await self._clients.wait_closed()
            self._clients = None

        muxes = [mux for muxes in self._workers.values() for mux in muxes]
        self._workers.clear()
        if muxes:
            await asyncio.gather(*(mux.aclose() for mux in muxes), return_exceptions=True)

        if self._runner is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._runner.cleanup(), 2.0)
            self._runner = None

    async def _on_wire(self, request: web.Request) -> web.StreamResponse:
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        try:
            self._verifier.verify(bearer)
        except Exception:  # noqa: BLE001
            return web.Response(status=401)
        if self._closing:
            return web.Response(status=503)

        ws = web.WebSocketResponse(max_msg_size=_MAX_MESSAGE)
        await ws.prepare(request)

        # read before the mux starts, since it is the frame that names the pool
        msg = await ws.receive()
        frame = AgentHttpFrame()
        frame.ParseFromString(msg.data)
        if frame.WhichOneof("message") != "registration":
            await ws.close()
            return ws
        worker_id = frame.registration.worker_id
        self._endpoints[worker_id] = set(frame.registration.endpoints)

        mux = Mux(ws, opener=True)
        await mux.start()
        self._workers.setdefault(worker_id, []).append(mux)
        self._ready.set()
        try:
            await mux.wait_closed()  # the handler outlives every stream on this wire
        finally:
            live = [m for m in self._workers.get(worker_id, []) if m is not mux]
            if live:
                self._workers[worker_id] = live
            else:
                self._workers.pop(worker_id, None)
                self._endpoints.pop(worker_id, None)
            if not self._workers:
                self._ready.clear()
        return ws

    async def open_stream(self, endpoint: str) -> Stream:
        """A byte pipe to a worker that announced this endpoint."""
        for worker_id, muxes in self._workers.items():
            if endpoint not in self._endpoints.get(worker_id, set()):
                continue
            live = [mux for mux in muxes if not mux.closed]
            if not live:
                continue
            # a wire stalled behind a lost packet holds its streams, so it stops
            # attracting new ones until it drains
            mux = min(live, key=lambda candidate: candidate.open_streams)
            self._opened += 1
            sid = await mux.open_stream(f"req_{self._opened}")
            return Stream(mux, sid)
        raise LookupError(f"no worker serves {endpoint!r}")

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # tracked so aclose() can cancel a handler parked in stream.read()
        task = asyncio.current_task()
        if task is not None:
            self._client_tasks.add(task)
            task.add_done_callback(self._client_tasks.discard)

        try:
            # the reader's own limit bounds a peer that never sends a newline
            head = await reader.readuntil(b"\r\n")
        except (asyncio.LimitOverrunError, asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return

        try:
            stream = await self.open_stream(_endpoint_of(head))
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"local controller refused a request: {exc}")
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\n\r\n")
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()
            return

        async def to_worker() -> None:
            await stream.write(head)  # read above to find the endpoint
            while chunk := await reader.read(CHUNK):
                await stream.write(chunk)
            # the client is gone, and without this the worker's reader parks on it
            await stream.write_eof()

        async def to_client() -> None:
            while chunk := await stream.read():
                writer.write(chunk)
                await writer.drain()
            # the worker ended, maybe by losing its wire; let the client see it go
            with contextlib.suppress(Exception):
                writer.close()

        try:
            await asyncio.gather(to_worker(), to_client())
        except Exception:  # noqa: BLE001
            pass
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()
            with contextlib.suppress(Exception):
                writer.close()


def _endpoint_of(request_line: bytes) -> str:
    """The endpoint in ``GET /get_order/42 HTTP/1.1``, which is the first segment."""
    parts = request_line.split(b" ")
    if len(parts) < 2:
        return ""
    path = parts[1].split(b"?", 1)[0].decode("latin-1")
    return path.lstrip("/").split("/", 1)[0]
