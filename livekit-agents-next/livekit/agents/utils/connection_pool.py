import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

from .. import _core
from ..log import logger
from . import aio

T = TypeVar("T")


class ConnectionPool(Generic[T]):
    """Helper class to manage persistent connections like websockets.

    Handles connection pooling and reconnection after max duration.
    Can be used as an async context manager to automatically return connections to the pool.
    Which connection is reused, retired or closed is the core's (``_core.ConnectionPoolState``);
    this holds the connections and runs the callbacks.
    """

    def __init__(
        self,
        *,
        max_session_duration: float | None = None,
        mark_refreshed_on_get: bool = False,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
        close_cb: Callable[[T], Awaitable[None]] | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        """Initialize the connection wrapper.

        Args:
            max_session_duration: Maximum duration in seconds before forcing reconnection
            mark_refreshed_on_get: If True, the session will be marked as fresh when get() is called. only used when max_session_duration is set.
            connect_cb: Optional async callback to create new connections
            close_cb: Optional async callback to close connections
        """  # noqa: E501
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._connect_timeout = connect_timeout
        self._state = _core.ConnectionPoolState(max_session_duration, mark_refreshed_on_get)
        self._conns: dict[int, T] = {}  # every connection the state knows, by id()
        self._connect_lock = asyncio.Lock()
        self._prewarm_task: weakref.ref[asyncio.Task[None]] | None = None

    @property
    def last_acquire_time(self) -> float:
        """Seconds the last ``get()`` spent connecting; 0 when it reused a connection."""
        return self._state.last_acquire_time

    @property
    def last_connection_reused(self) -> bool:
        return self._state.last_connection_reused

    async def _connect(self, timeout: float, *, started_at: float | None = None) -> T:
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        while True:
            generation = self._state.generation
            conn = await self._connect_cb(timeout)
            self._conns[id(conn)] = conn
            elapsed = None if started_at is None else time.perf_counter() - started_at
            if self._state.connected(id(conn), time.time(), generation, elapsed):
                return conn
            # options changed during the handshake: the state queued this socket for closing
            await self._drain_to_close()

    async def _drain_to_close(self) -> None:
        """Close the connections queued for closing."""
        while (cid := self._state.next_closing()) is not None:
            conn = self._conns[cid]
            try:
                if self._close_cb is not None:
                    await self._close_cb(conn)
            except Exception as e:
                logger.warning("error closing connection %s: %s", conn, e)
            self._state.closed(cid)
            del self._conns[cid]

    @asynccontextmanager
    async def connection(self, *, timeout: float) -> AsyncGenerator[T, None]:
        """Get a connection from the pool and automatically return it when done.

        Yields:
            An active connection object
        """
        conn = await self.get(timeout=timeout)
        try:
            yield conn
        except BaseException:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(self, *, timeout: float) -> T:
        """Get an available connection or create a new one if needed.

        Returns:
            An active connection object
        """
        async with self._connect_lock:
            await self._drain_to_close()
            if (cid := self._state.get(time.time())) is not None:
                return self._conns[cid]
            return await self._connect(timeout, started_at=time.perf_counter())

    def put(self, conn: T) -> None:
        """Mark a connection as available for reuse.

        If connection has been reset, it will not be added to the pool.
        A connection retired by :meth:`invalidate` while it was checked out is queued
        for closing instead of being made available again.

        Args:
            conn: The connection to make available
        """
        self._state.put(id(conn))

    def remove(self, conn: T) -> None:
        """Remove a specific connection from the pool.

        Marks the connection to be closed during the next drain cycle.

        Args:
            conn: The connection to reset
        """
        self._state.remove(id(conn))

    def invalidate(self) -> None:
        """Stop reusing every existing connection.

        Idle connections are marked to be closed during the next drain cycle.
        Connections that are currently checked out are *retired* instead: they keep
        working for whoever holds them and are queued for closing when returned via
        :meth:`put` or :meth:`remove`. Closing them here would sever a connection that
        is still streaming, so a caller changing options mid-session would interrupt
        the request in flight. A handshake that was already in flight is discarded
        and retried, since the socket it produced carries the old settings and no
        caller has been handed it yet.
        """
        self._state.invalidate()

    def prewarm(self) -> None:
        """Initiate prewarming of the connection pool without blocking.

        This method starts a background task that creates a new connection if none exist.
        The task automatically cleans itself up when the connection pool is closed.
        """
        task = self._prewarm_task() if self._prewarm_task is not None else None
        if (task is not None and not task.done()) or not self._state.empty():
            return

        async def _prewarm_impl() -> None:
            try:
                async with self._connect_lock:
                    if self._state.empty():
                        self.put(await self._connect(self._connect_timeout))
            except Exception as e:
                # exception details can contain request headers or URL credentials.
                logger.warning(
                    "failed to prewarm connection pool",
                    extra={"exception_type": type(e).__name__},
                )

        self._prewarm_task = weakref.ref(asyncio.create_task(_prewarm_impl()))

    async def aclose(self) -> None:
        """Close all connections, draining any pending connection closures."""
        if self._prewarm_task is not None and (task := self._prewarm_task()) is not None:
            await aio.gracefully_cancel(task)
        # the retired connections too, rather than waiting for holders that may never return them
        self._state.close_all()
        await self._drain_to_close()
