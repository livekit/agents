import asyncio
import time
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Generic, TypeVar

from ..log import logger
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
        self._max_session_duration = max_session_duration
        self._mark_refreshed_on_get = mark_refreshed_on_get
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._connections: dict[T, float] = {}  # conn -> connected_at timestamp
        self._connection_keys: dict[T, object | None] = {}
        self._available: set[T] = set()
        self._connect_timeout = connect_timeout
        self._connect_lock = asyncio.Lock()
        self._closed = False

        # store connections to be reaped (closed) later.
        self._to_close: set[T] = set()

        # connections that were invalidated while checked out. they stay usable for
        # their current holder and are queued for closing once returned.
        self._retired: set[T] = set()

        # bumped by invalidate() so a connection whose handshake was already in
        # flight can be recognised as stale once it completes.
        self._invalidations: int = 0

        self._prewarm_task: weakref.ref[asyncio.Task[None]] | None = None

        # Timing info from the last get() call
        self.last_acquire_time: float = 0.0
        self.last_connection_reused: bool = False

    @staticmethod
    def _resolve_connection_key(
        key: object | Callable[[], object] | None,
    ) -> object | None:
        return key() if callable(key) else key

    async def _connect(
        self,
        timeout: float,
        *,
        key: object | Callable[[], object] | None = None,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
    ) -> T:
        """Create a new connection.

        Returns:
            The new connection object

        Raises:
            NotImplementedError: If no connect callback was provided
        """
        if self._closed:
            raise RuntimeError("ConnectionPool is closed")
        callback = connect_cb or self._connect_cb
        if callback is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        while True:
            invalidations = self._invalidations
            connection = await callback(timeout)
            if self._closed:
                self._to_close.add(connection)
                await self._drain_to_close()
                raise RuntimeError("ConnectionPool is closed")
            if invalidations == self._invalidations:
                break
            # options changed during the handshake, so this socket carries the old ones.
            # close it here rather than leaving it queued: the drain at the top of get()
            # has already run, so nothing else would reach it until the next acquisition.
            self._to_close.add(connection)
            await self._drain_to_close()
        self._connections[connection] = time.time()
        self._connection_keys[connection] = self._resolve_connection_key(key)
        return connection

    async def _drain_to_close(self) -> None:
        """Drain and close all the connections queued for closing."""
        while self._to_close:
            conn = self._to_close.pop()
            try:
                await self._maybe_close_connection(conn)
            except Exception as e:
                logger.warning("error closing connection %s: %s", conn, e)
            except BaseException:
                # the connection has already left _to_close and is not in _connections,
                # so nothing else owns it. put it back before unwinding, otherwise a
                # cancelled drain strands an open connection for good.
                self._to_close.add(conn)
                raise

    @asynccontextmanager
    async def connection(
        self,
        *,
        timeout: float,
        key: object | Callable[[], object] | None = None,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
    ) -> AsyncGenerator[T, None]:
        """Get a connection from the pool and automatically return it when done.

        Yields:
            An active connection object
        """
        conn = await self.get(timeout=timeout, key=key, connect_cb=connect_cb)
        try:
            yield conn
        except BaseException:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(
        self,
        *,
        timeout: float,
        key: object | Callable[[], object] | None = None,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
    ) -> T:
        """Get an available connection or create a new one if needed.

        Returns:
            An active connection object
        """
        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("ConnectionPool is closed")
            await self._drain_to_close()
            now = time.time()
            requested_key = self._resolve_connection_key(key)
            mismatched: list[T] = []

            # try to reuse an available connection that hasn't expired
            while self._available:
                conn = self._available.pop()
                if self._connection_keys.get(conn) == requested_key and (
                    self._max_session_duration is None
                    or now - self._connections[conn] <= self._max_session_duration
                ):
                    self._available.update(mismatched)
                    if self._mark_refreshed_on_get:
                        self._connections[conn] = now
                    self.last_acquire_time = 0.0
                    self.last_connection_reused = True
                    return conn
                if self._connection_keys.get(conn) != requested_key:
                    mismatched.append(conn)
                    continue
                # connection expired; mark it for resetting.
                self.remove(conn)

            self._available.update(mismatched)
            await self._drain_to_close()
            t0 = time.perf_counter()
            conn = await self._connect(timeout, key=key, connect_cb=connect_cb)
            self.last_acquire_time = time.perf_counter() - t0
            self.last_connection_reused = False
            return conn

    def put(self, conn: T) -> None:
        """Mark a connection as available for reuse.

        If connection has been reset, it will not be added to the pool.
        A connection retired by :meth:`invalidate` while it was checked out is queued
        for closing instead of being made available again.

        Args:
            conn: The connection to make available
        """
        if conn in self._retired:
            self.remove(conn)
            return

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
        if conn in self._retired:
            self._retired.discard(conn)
            self._to_close.add(conn)
            return

        if conn in self._connections:
            self._to_close.add(conn)
            self._connections.pop(conn, None)
            self._connection_keys.pop(conn, None)

    def invalidate(self) -> None:
        """Stop reusing every existing connection.

        Idle connections are marked to be closed during the next drain cycle.
        Connections that are currently checked out are *retired* instead: they keep
        working for whoever holds them and are queued for closing when returned via
        :meth:`put` or :meth:`remove`. Closing them here would sever a connection that
        is still streaming, so a caller changing options mid-session would interrupt
        the request in flight. A handshake that was already in flight is discarded
        and retried by :meth:`_connect`, since the socket it produced carries the old
        settings and no caller has been handed it yet.
        """
        self._invalidations += 1
        for conn in list(self._connections.keys()):
            if conn in self._available:
                self._to_close.add(conn)
            else:
                self._retired.add(conn)
        self._connections.clear()
        self._connection_keys.clear()
        self._available.clear()

    def prewarm(
        self,
        *,
        key: object | Callable[[], object] | None = None,
        connect_cb: Callable[[float], Awaitable[T]] | None = None,
    ) -> None:
        """Initiate prewarming of the connection pool without blocking.

        This method starts a background task that creates a new connection if none exist.
        The task automatically cleans itself up when the connection pool is closed.
        """
        if self._closed:
            return

        if self._prewarm_task is not None:
            task = self._prewarm_task()
            if task is not None and not task.done():
                return
            self._prewarm_task = None

        requested_key = self._resolve_connection_key(key)
        if any(self._connection_keys.get(conn) == requested_key for conn in self._connections):
            return

        async def _prewarm_impl() -> None:
            try:
                async with self._connect_lock:
                    if self._closed:
                        return
                    if not any(
                        self._connection_keys.get(conn) == self._resolve_connection_key(key)
                        for conn in self._connections
                    ):
                        conn = await self._connect(
                            timeout=self._connect_timeout,
                            key=key,
                            connect_cb=connect_cb,
                        )
                        self._available.add(conn)
            except Exception as e:
                # exception details can contain request headers or URL credentials.
                logger.warning(
                    "failed to prewarm connection pool",
                    extra={"exception_type": type(e).__name__},
                )

        task = asyncio.create_task(_prewarm_impl())
        self._prewarm_task = weakref.ref(task)

    async def aclose(self) -> None:
        """Close all connections, draining any pending connection closures."""
        if self._prewarm_task is not None:
            task = self._prewarm_task()
            if task:
                await aio.gracefully_cancel(task)

        # Take the same lock used by get() and prewarm(). This keeps shutdown from
        # returning while a handshake can still register a replacement connection.
        async with self._connect_lock:
            if self._closed:
                return
            self._closed = True
            self.invalidate()
            # the pool is going away, so retired connections are closed too rather than
            # waiting for holders that may never return them.
            self._to_close.update(self._retired)
            self._retired.clear()
            await self._drain_to_close()
