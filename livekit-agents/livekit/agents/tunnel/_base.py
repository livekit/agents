from __future__ import annotations

import abc
import asyncio
import contextlib
from collections.abc import AsyncIterator

from ..log import logger

CHUNK = 64 * 1024


class ByteStream(abc.ABC):
    """One HTTP connection as a byte pipe. The edge opens it, the worker answers on it."""

    @abc.abstractmethod
    async def read(self) -> bytes:
        """The next bytes from the edge, or empty once it has sent everything.

        Empty ends the reading half only: the worker's reply may still be going out.
        """

    @abc.abstractmethod
    async def write(self, data: bytes) -> None:
        """Send bytes to the edge, waiting while the carrier has no room for them."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """End the stream. Anything unread or unsent is lost."""

    async def write_eof(self) -> None:
        """Finish writing while still reading.

        A carrier that can half-close should override this. The default cannot, so it
        ends the stream, and the reply is whatever arrived before that.
        """
        await self.aclose()


class Tunnel(abc.ABC):
    """Worker side: the connection to the edge, and the streams the edge opens on it.

    A subclass writes three methods and nothing else: dial, hand over the streams the
    edge opens, hang up. Serving one is the same whatever carries it, so this class does
    that — a task per stream, piping it to the local HTTP server::

        class MyTunnel(Tunnel):
            async def _connect(self) -> None:
                self._conn = await dial(self.endpoints)  # raises if the edge refuses

            async def _accept(self) -> AsyncIterator[ByteStream]:
                while stream := await self._conn.next_stream():
                    yield stream

            async def _disconnect(self) -> None:
                await self._conn.aclose()  # ends _accept, which ends the serving
    """

    def __init__(self) -> None:
        self._target_port = 0
        self._endpoints: list[str] = []
        self._serving: set[asyncio.Task[None]] = set()
        self._serve_atask: asyncio.Task[None] | None = None

    @property
    def target_port(self) -> int:
        """Local HTTP server every stream is piped to."""
        return self._target_port

    @property
    def endpoints(self) -> list[str]:
        """First path segments the local server answers under, as announced to the edge.

        The edge routes on one segment and sends nothing outside them.
        """
        return self._endpoints

    async def start(self, *, target_port: int, endpoints: list[str]) -> None:
        """Connect and begin serving. Returns once requests can arrive."""
        self._target_port = target_port
        self._endpoints = list(endpoints)
        await self._connect()
        self._serve_atask = asyncio.create_task(self._serve_task())

    async def aclose(self) -> None:
        """Hang up, then let go of the requests that were in flight."""
        await self._disconnect()
        if self._serve_atask is not None:
            await self._serve_atask  # ends once _accept does, dropping the streams with it
            self._serve_atask = None

    @abc.abstractmethod
    async def _connect(self) -> None:
        """Dial the edge and announce ``endpoints``.

        Returns once traffic can arrive, and raises if it cannot: an edge that will not
        take this worker is the caller's problem, not something to retry behind its back.

        This is the first thing to run on the loop that serves, so anything bound to one
        belongs here rather than in ``__init__``, which a caller may run at import time.
        """

    @abc.abstractmethod
    def _accept(self) -> AsyncIterator[ByteStream]:
        """Yield every stream the edge opens, until ``_disconnect`` ends the connection.

        Each one is served as its own task, so this may yield again while earlier streams
        are still running.
        """

    @abc.abstractmethod
    async def _disconnect(self) -> None:
        """Close the connection, which ends ``_accept``. Called once, by ``aclose``."""

    async def _serve_task(self) -> None:
        """Pipe each stream to the local server, concurrently, until the tunnel ends."""
        async for stream in self._accept():
            task = asyncio.create_task(self._pipe(stream))
            self._serving.add(task)
            task.add_done_callback(self._serving.discard)

        serving, self._serving = self._serving, set()
        for task in serving:
            task.cancel()
        await asyncio.gather(*serving, return_exceptions=True)

    async def _pipe(self, stream: ByteStream) -> None:
        """One stream to the local HTTP server, until both directions finish."""
        reader = writer = None
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", self._target_port)

            async def to_app() -> None:
                while chunk := await stream.read():
                    writer.write(chunk)
                    await writer.drain()

            async def from_app() -> None:
                while chunk := await reader.read(CHUNK):
                    await stream.write(chunk)
                # the app closed its socket, so the exchange is over either way
                await stream.write_eof()

            await asyncio.gather(to_app(), from_app())
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"tunnel stream failed: {exc}")
        finally:
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
            with contextlib.suppress(Exception):
                await stream.aclose()
