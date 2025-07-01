import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Awaitable
from contextlib import asynccontextmanager
from typing import Callable, Generic, Optional, TypeVar

from . import aio

T = TypeVar("T")


class ConnectionPool(Generic[T]):
    """Helper class to manage persistent connections like websockets.

    Handles connection pooling and reconnection after max duration.
    Can be used as an async context manager to automatically return connections to the pool.
    """

    def __init__(
        self,
        *,
        max_session_duration: Optional[float] = None,
        mark_refreshed_on_get: bool = False,
        connect_cb: Optional[Callable[[float], Awaitable[T]]] = None,
        close_cb: Optional[Callable[[T], Awaitable[None]]] = None,
        connect_timeout: float = 10.0,
    ) -> None:
        """Initialize the connection wrapper.

        Args:
            max_session_duration: Maximum duration in seconds before forcing reconnection
            mark_refreshed_on_get: If True, the session will be marked as fresh when get() is called. only used when max_session_duration is set.
            connect_cb: Optional async callback to create new connections
            close_cb: Optional async callback to close connections
        """  # noqa: E501
        self._max_session_duration = max_session_duration
        self._mark_refreshed_on_get = mark_refreshed_on_get
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._connections: dict[T, float] = {}  # conn -> connected_at timestamp
        self._available: set[T] = set()
        self._connect_timeout = connect_timeout

        # store connections to be reaped (closed) later.
        self._to_close: set[T] = set()

        self._prewarm_task: Optional[weakref.ref[asyncio.Task[None]]] = None

    async def _connect(self, timeout: float) -> T:
        """Create a new connection.

        Returns:
            The new connection object

        Raises:
            NotImplementedError: If no connect callback was provided
        """
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        connection = await self._connect_cb(timeout)
        self._connections[connection] = time.time()
        return connection

    async def _drain_to_close(self) -> None:
        """Drain and close all the connections queued for closing."""
        for conn in list(self._to_close):
            await self._maybe_close_connection(conn)
        self._to_close.clear()

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
        await self._drain_to_close()
        now = time.time()

        # try to reuse an available connection that hasn't expired
        while self._available:
            conn = self._available.pop()
            if (
                self._max_session_duration is None
                or now - self._connections[conn] <= self._max_session_duration
            ):
                if self._mark_refreshed_on_get:
                    self._connections[conn] = now
                return conn
            # connection expired; mark it for resetting.
            self.remove(conn)

        return await self._connect(timeout)

    def put(self, conn: T) -> None:
        """Mark a connection as available for reuse.

        If connection has been reset, it will not be added to the pool.

        Args:
            conn: The connection to make available
        """
        if conn in self._connections:
            self._available.add(conn)

    async def _maybe_close_connection(self, conn: T) -> None:
        """Close a connection if close_cb is provided.

        Args:
            conn: The connection to close
        """
        if self._close_cb is not None:
            await self._close_cb(conn)

    def remove(self, conn: T) -> None:
        """Remove a specific connection from the pool.

        Marks the connection to be closed during the next drain cycle.

        Args:
            conn: The connection to reset
        """
        self._available.discard(conn)
        if conn in self._connections:
            self._to_close.add(conn)
            self._connections.pop(conn, None)

    def invalidate(self) -> None:
        """Clear all existing connections.

        Marks all current connections to be closed during the next drain cycle.
        """
        for conn in list(self._connections.keys()):
            self._to_close.add(conn)
        self._connections.clear()
        self._available.clear()

    def prewarm(self) -> None:
        """Initiate prewarming of the connection pool without blocking.

        This method starts a background task that creates a new connection if none exist.
        The task automatically cleans itself up when the connection pool is closed.
        """
        if self._prewarm_task is not None or self._connections:
            return

        async def _prewarm_impl() -> None:
            if not self._connections:
                conn = await self._connect(timeout=self._connect_timeout)
                self._available.add(conn)

        task = asyncio.create_task(_prewarm_impl())
        self._prewarm_task = weakref.ref(task)

    async def aclose(self) -> None:
        """Close all connections, draining any pending connection closures."""
        if self._prewarm_task is not None:
            task = self._prewarm_task()
            if task:
                await aio.gracefully_cancel(task)

        self.invalidate()
        await self._drain_to_close()
