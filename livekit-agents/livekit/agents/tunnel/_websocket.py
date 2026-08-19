"""Serving ``AgentServer.http`` from LiveKit Cloud, over a pool of websockets."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aiohttp

from livekit import api
from livekit.protocol.agent_proxy import (
    AgentHttpEof,
    AgentHttpFrame,
    AgentHttpOpen,
    AgentHttpRegistration,
    AgentHttpReset,
)

from .. import utils
from ..log import logger
from ..utils import aio
from ._base import CHUNK, ByteStream, Tunnel

if TYPE_CHECKING:
    from aiohttp import web

    # the edge half runs on a server websocket, the worker half on a client one
    _WebSocket = aiohttp.ClientWebSocketResponse | web.WebSocketResponse

TUNNEL_PATH = "/agent-http-tunnel"
"""Where the edge accepts a wire."""

_TOKEN_TTL = datetime.timedelta(hours=6)
"""Life of a wire's token. Minted per dial, so a redial always carries a fresh one."""

_STREAM_WINDOW = 1024 * 1024
"""Bytes in flight on one stream before its sender waits for credit."""

_CONN_WINDOW = 8 * 1024 * 1024
"""The same for a whole wire, sized for several concurrent bulk streams."""

_CREDIT_FRACTION = 2
"""Credit is returned at this fraction of a window, not per frame."""

_MAX_FRAME = CHUNK
"""Largest data payload, so one frame carries one pipe read."""

_MAX_MESSAGE = _MAX_FRAME + 4096

_MAX_STREAMS = 4096
"""Streams one wire carries. The app's own limits bind first; this caps a runaway."""

_SLOT_TIMEOUT = 10.0
"""How long a new stream waits for a slot on a full wire."""

_TEARDOWN_TIMEOUT = 2.0


class Mux:
    """Drives one websocket carrying many streams, and owns their flow control.

    Credit is returned as a stream is drained, not as bytes arrive, so a reader that
    stops stops the peer sending to it instead of filling a buffer here.
    """

    def __init__(self, ws: _WebSocket, *, opener: bool) -> None:
        self._ws = ws
        self._inbox: dict[int, asyncio.Queue[bytes | None]] = {}
        self._send_window: dict[int, int] = {}
        self._owed: dict[int, int] = {}  # drained on a stream and not yet credited back
        self._undrained: dict[int, int] = {}  # delivered to a stream and not yet read out
        self._conn_send = _CONN_WINDOW
        self._conn_owed = 0
        self._dead: set[int] = set()
        self._next_sid = 1 if opener else 2  # split, so both sides could open one day
        self._write_lock = asyncio.Lock()
        # one future per sender: a shared event loses a wakeup between a check and a wait
        self._window_waiters: list[asyncio.Future[None]] = []
        self._slots = asyncio.Event()  # a stream ended, so the wire has room for another
        self._slots.set()
        self._done = asyncio.Event()
        self._read_task: asyncio.Task[None] | None = None
        self.accepted: asyncio.Queue[int | None] = asyncio.Queue()
        """Stream ids the peer opened, then ``None`` once the wire is gone."""

    @property
    def closed(self) -> bool:
        return self._done.is_set()

    @property
    def open_streams(self) -> int:
        """Streams alive on this wire, which is what the edge balances on."""
        return len(self._inbox)

    async def start(self) -> None:
        self._read_task = asyncio.create_task(self._read_loop())

    async def wait_closed(self) -> None:
        await self._done.wait()

    async def register(self, registration: AgentHttpRegistration) -> None:
        """The first frame of every wire: who this worker is and what it serves."""
        await self._write(AgentHttpFrame(registration=registration))

    async def open_stream(self, request_id: str) -> int:
        deadline = time.monotonic() + _SLOT_TIMEOUT
        while len(self._inbox) >= _MAX_STREAMS:
            # wait for a stream to end rather than fail the new one
            self._slots.clear()
            left = deadline - time.monotonic()
            if left <= 0 or self._done.is_set():
                raise RuntimeError(f"tunnel wire stuck at {_MAX_STREAMS} open streams")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._slots.wait(), left)

        sid = self._next_sid
        self._next_sid += 2
        self._track(sid)
        await self._write(AgentHttpFrame(stream_id=sid, open=AgentHttpOpen(request_id=request_id)))
        return sid

    async def send(self, sid: int, data: bytes) -> None:
        view = memoryview(data)
        while view:
            if self._done.is_set() or sid in self._dead or sid not in self._send_window:
                return  # a torn-down wire drops writes rather than failing the local app
            room = min(self._send_window[sid], self._conn_send, _MAX_FRAME, len(view))
            if room <= 0:
                fut = asyncio.get_running_loop().create_future()
                self._window_waiters.append(fut)
                # re-check after registering, so credit landing in between is not lost
                if self._done.is_set() or min(self._send_window[sid], self._conn_send) > 0:
                    self._wake_senders()
                await fut
                continue

            self._send_window[sid] -= room
            self._conn_send -= room
            await self._write(AgentHttpFrame(stream_id=sid, data=bytes(view[:room])))
            view = view[room:]

    async def recv(self, sid: int) -> bytes | None:
        inbox = self._inbox.get(sid)
        if inbox is None:
            return None
        return await inbox.get()

    async def end(self, sid: int) -> None:
        """Half-close: nothing more this way, the other direction keeps going."""
        if not self._done.is_set() and sid not in self._dead:
            await self._write(AgentHttpFrame(stream_id=sid, eof=AgentHttpEof()))

    def forget(self, sid: int) -> None:
        if (inbox := self._inbox.pop(sid, None)) is not None:
            inbox.put_nowait(None)  # wake a reader parked on a stream that is going away
        self._send_window.pop(sid, None)
        self._owed.pop(sid, None)
        # whatever it buffered and nobody will read still cost the peer connection
        # window, so it goes back rather than being lost to the wire for good
        self._conn_owed += self._undrained.pop(sid, 0)
        self._dead.discard(sid)
        self._slots.set()

    async def drained(self, sid: int, size: int) -> None:
        """Give back the window a reader just freed."""
        if size <= 0 or sid not in self._inbox:
            return  # forget() has already given this stream's window back
        self._undrained[sid] -= size
        self._conn_owed += size
        if self._conn_owed >= _CONN_WINDOW // _CREDIT_FRACTION:
            owed, self._conn_owed = self._conn_owed, 0
            await self._write(AgentHttpFrame(stream_id=0, credit=owed))  # the wire's own

        owed = self._owed.get(sid, 0) + size
        if owed >= _STREAM_WINDOW // _CREDIT_FRACTION:
            self._owed[sid] = 0
            await self._write(AgentHttpFrame(stream_id=sid, credit=owed))
        else:
            self._owed[sid] = owed

    async def aclose(self) -> None:
        if self._read_task is not None:
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task
            self._read_task = None
        self._shutdown()
        # bounded: the close handshake would otherwise wait on a peer that is gone
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._ws.close(), _TEARDOWN_TIMEOUT)

    def _track(self, sid: int) -> None:
        self._inbox[sid] = asyncio.Queue()
        self._send_window[sid] = _STREAM_WINDOW
        self._owed[sid] = 0
        self._undrained[sid] = 0

    async def _read_loop(self) -> None:
        try:
            while True:
                msg = await self._ws.receive()
                if msg.type is not aiohttp.WSMsgType.BINARY:
                    return  # a close, an error, or a text frame nobody here sends

                frame = AgentHttpFrame()
                frame.ParseFromString(msg.data)
                kind = frame.WhichOneof("message")
                if kind == "data":
                    await self._on_data(frame.stream_id, frame.data)
                elif kind == "credit":
                    if frame.stream_id == 0:
                        self._conn_send += frame.credit  # the window the wire shares
                    elif frame.stream_id in self._send_window:
                        self._send_window[frame.stream_id] += frame.credit
                    self._wake_senders()
                elif kind == "open":
                    self._track(frame.stream_id)
                    self.accepted.put_nowait(frame.stream_id)
                elif kind == "eof":
                    self._deliver(frame.stream_id, None)
                elif kind == "reset":
                    self._dead.add(frame.stream_id)
                    self._deliver(frame.stream_id, None)
                    self._slots.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"tunnel read loop ended: {exc}")
        finally:
            self._shutdown()

    async def _on_data(self, sid: int, payload: bytes) -> None:
        if sid not in self._inbox:
            # a forgotten stream, told once: silence spends the peer's whole window
            if sid not in self._dead:
                self._dead.add(sid)
                await self._write(AgentHttpFrame(stream_id=sid, reset=AgentHttpReset()))
            return

        # booked, not credited: the peer gets its window back when the reader takes it
        self._undrained[sid] += len(payload)
        self._deliver(sid, payload)

    async def _write(self, frame: AgentHttpFrame) -> None:
        if self._done.is_set():
            return
        buf = frame.SerializeToString()
        # aiohttp may split one frame across writes, so senders must not interleave
        async with self._write_lock:
            try:
                await self._ws.send_bytes(buf)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"tunnel write failed: {exc}")
                self._shutdown()

    def _deliver(self, sid: int, chunk: bytes | None) -> None:
        if (inbox := self._inbox.get(sid)) is not None:
            inbox.put_nowait(chunk)

    def _wake_senders(self) -> None:
        """Release every sender parked on a window."""
        for fut in self._window_waiters:
            if not fut.done():
                fut.set_result(None)
        self._window_waiters.clear()

    def _shutdown(self) -> None:
        if self._done.is_set():
            return
        self._done.set()
        self._slots.set()
        self._wake_senders()
        self.accepted.put_nowait(None)
        for inbox in self._inbox.values():
            inbox.put_nowait(None)


class Stream(ByteStream):
    """One multiplexed stream as a byte pipe, carrying one HTTP connection."""

    def __init__(self, mux: Mux, sid: int, *, on_close: Callable[[], None] | None = None) -> None:
        self._mux = mux
        self._sid = sid
        self._on_close = on_close
        self._eof = False
        self._sent_eof = False
        self._closed = False

    async def read(self) -> bytes:
        """Next chunk, or empty bytes at end of stream."""
        if self._eof:
            return b""
        chunk = await self._mux.recv(self._sid)
        if chunk is None:
            self._eof = True
            return b""
        await self._mux.drained(self._sid, len(chunk))
        return chunk

    async def write(self, data: bytes) -> None:
        if not self._closed:
            await self._mux.send(self._sid, data)

    async def write_eof(self) -> None:
        """Half-close: the peer reads EOF while this side keeps reading its reply."""
        if not self._sent_eof and not self._closed:
            self._sent_eof = True
            await self._mux.end(self._sid)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._sent_eof:
            self._sent_eof = True
            await self._mux.end(self._sid)  # an end, so the peer does not read a failure
        self._mux.forget(self._sid)
        if self._on_close is not None:
            self._on_close()


@dataclass(eq=False)  # identity, so a wire is still itself after a redial
class _Wire:
    """One dialled websocket and the task serving the streams that arrive on it."""

    mux: Mux
    task: asyncio.Task[None] = field(init=False)


class WebSocketTunnel(Tunnel):
    """Websockets to LiveKit, carrying the requests it sends this worker.

    Every wire opens by announcing the worker and its endpoints, so the node that took it
    can route there without a lookup. Credentials are read from the environment when they
    are not given, the same way ``AgentServer`` reads its own.

    ``wires`` is how many are held. More of them survive packet loss better, since a
    stream only waits behind the others on its own wire; fewer cost less to keep open.
    """

    def __init__(
        self,
        *,
        ws_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        wires: int = 4,
        max_concurrent_requests: int = 0,
    ) -> None:
        super().__init__()
        self._ws_url = ws_url or os.environ.get("LIVEKIT_URL") or ""
        self._api_key = api_key or os.environ.get("LIVEKIT_API_KEY") or ""
        self._api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET") or ""
        self._wire_target = max(1, wires)
        self._max_concurrent = max_concurrent_requests
        # the pool's own identity: registration has not happened when a wire is dialled,
        # and two workers sharing an id would merge into one pool
        self._worker_id = utils.shortuuid("AHW_")
        self._registration = AgentHttpRegistration(worker_id=self._worker_id)
        self._wires: list[_Wire] = []
        self._live: set[Mux] = set()  # so a close is accounted exactly once
        # every wire's streams merge here. Built in _connect, since a Chan binds to the
        # loop it is made on and a tunnel is usually built at import time.
        self._incoming: aio.Chan[ByteStream] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._closing = False

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def wire_count(self) -> int:
        return len(self._wires)

    async def _connect(self) -> None:
        if not self._ws_url:
            raise ValueError("ws_url is required, or set the LIVEKIT_URL environment variable")
        if not self._api_key:
            raise ValueError("api_key is required, or set the LIVEKIT_API_KEY environment variable")
        if not self._api_secret:
            raise ValueError(
                "api_secret is required, or set the LIVEKIT_API_SECRET environment variable"
            )
        self._registration = AgentHttpRegistration(
            worker_id=self._worker_id,
            endpoints=self.endpoints,
            max_concurrent_requests=self._max_concurrent,
        )
        incoming = self._incoming = aio.Chan[ByteStream]()
        # a pooled websocket never returns to the connector, so any limit caps the pool
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0, limit_per_host=0)
        )
        # an unreachable edge is the caller's problem, not a retry
        muxes = await asyncio.gather(*(self._dial() for _ in range(self._wire_target)))
        for mux in muxes:
            wire = _Wire(mux=mux)
            wire.task = asyncio.create_task(self._read_wire(wire, incoming))
            self._wires.append(wire)

    async def _accept(self) -> AsyncIterator[ByteStream]:
        if self._incoming is None:
            return
        async for stream in self._incoming:
            yield stream

    async def _disconnect(self) -> None:
        self._closing = True

        wires, self._wires = self._wires, []
        for wire in wires:
            wire.task.cancel()
        if wires:
            await asyncio.gather(*(wire.task for wire in wires), return_exceptions=True)

        muxes = list(self._live)
        for mux in muxes:
            self._drop(mux)
        # together, so one wire whose peer vanished cannot hold up the rest
        if muxes:
            await asyncio.gather(*(mux.aclose() for mux in muxes), return_exceptions=True)

        if self._session is not None:
            await self._session.close()
            self._session = None

        if self._incoming is not None:
            self._incoming.close()  # no wire is left to feed it, so _accept can end

    async def _dial(self) -> Mux:
        if self._session is None:
            raise RuntimeError("tunnel is not started")
        # minted per dial, so a wire that redials is never refused for an expired token
        token = (
            api.AccessToken(self._api_key, self._api_secret)
            .with_identity(self._worker_id)
            .with_grants(api.VideoGrants(agent=True))
            .with_ttl(_TOKEN_TTL)
            .to_jwt()
        )
        ws = await self._session.ws_connect(
            _tunnel_url(self._ws_url),
            headers={"Authorization": f"Bearer {token}"},
            max_msg_size=_MAX_MESSAGE,
        )
        mux = Mux(ws, opener=False)
        await mux.register(self._registration)
        await mux.start()
        self._live.add(mux)
        return mux

    def _drop(self, mux: Mux) -> None:
        """Forget a wire once, whichever path closed it."""
        self._live.discard(mux)

    async def _read_wire(self, wire: _Wire, incoming: aio.Chan[ByteStream]) -> None:
        """Feed one wire's streams out, redialling it alone so the others keep working."""
        backoff = 0.1
        while not self._closing:
            while (sid := await wire.mux.accepted.get()) is not None:
                incoming.send_nowait(Stream(wire.mux, sid))

            # accounted before the await, so a cancel here cannot double-count the wire
            self._drop(wire.mux)
            await wire.mux.aclose()
            if self._closing:
                return

            # its streams are lost either way, so redial without waiting for them
            while not self._closing:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                try:
                    wire.mux = await self._dial()
                except (OSError, aiohttp.ClientError) as exc:
                    logger.debug(f"tunnel redial failed: {exc}")
                    continue
                backoff = 0.1
                break


def _tunnel_url(ws_url: str) -> str:
    """Where the edge accepts a wire, from the url the worker was given."""
    if ws_url.startswith(("http://", "https://")):
        ws_url = ws_url.replace("http", "ws", 1)
    return ws_url.rstrip("/") + TUNNEL_PATH
