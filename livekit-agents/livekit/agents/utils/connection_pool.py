import time
from contextlib import asynccontextmanager
from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Set,
    TypeVar,
)

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
        inactivity_timeout: Optional[float] = None,
        connect_cb: Optional[Callable[[], Awaitable[T]]] = None,
        close_cb: Optional[Callable[[T], Awaitable[None]]] = None,
    ) -> None:
        """Initialize the connection wrapper.

        Args:
            max_session_duration: Maximum duration in seconds before forcing reconnection
            inactivity_timeout: Inactivity timeout in seconds.
            connect_cb: Optional async callback to create new connections
            close_cb: Optional async callback to close connections
        """
        self._max_session_duration = max_session_duration
        self._inactivity_timeout = inactivity_timeout
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._connections: dict[T, float] = {}  # conn -> connected_at timestamp
        self._last_activity: dict[T, float] = {}  # con -> last activity timestamp

        self._available: Set[T] = set()

        # store connections to be reaped (closed) later.
        self._to_close: Set[T] = set()

    async def _connect(self) -> T:
        """Create a new connection.

        Returns:
            The new connection object

        Raises:
            NotImplementedError: If no connect callback was provided
        """
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        conn = await self._connect_cb()
        now = time.monotonic()
        self._connections[conn] = now
        self._last_activity[conn] = now
        return conn

    async def _drain_to_close(self) -> None:
        """Drain and close all the connections queued for closing."""
        for conn in list(self._to_close):
            await self._maybe_close_connection(conn)
        self._to_close.clear()

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator[T, None]:
        """Get a connection from the pool and automatically return it when done.

        Yields:
            An active connection object
        """
        conn = await self.get()
        try:
            yield conn
        except Exception:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(self) -> T:
        """Get an available connection or create a new one if needed.

        Returns:
            An active connection object
        """
        await self._drain_to_close()

        # try to reuse an available connection that hasn't expired
        while self._available:
            conn = self._available.pop()
            if self._is_connection_valid(conn):
                return conn
            # connection expired; mark it for resetting.
            self.remove(conn)

        return await self._connect()

    def _is_connection_valid(self, conn: T, buffer: float = 10.0) -> bool:
        last_activity_time = self._last_activity[conn]
        connected_at_time = self._connections[conn]
        now = time.monotonic()

        # check max session duration
        session_valid = (
            self._max_session_duration is None
            or now - connected_at_time <= self._max_session_duration
        )

        # check last activity timeout
        activity_valid = (
            self._inactivity_timeout is None
            or now - last_activity_time <= self._inactivity_timeout - buffer
        )

        return session_valid and activity_valid

    def put(self, conn: T) -> None:
        """Mark a connection as available for reuse.

        If connection has been reset, it will not be added to the pool.

        Args:
            conn: The connection to make available
        """
        if conn in self._connections:
            self._available.add(conn)
            self.touch(conn)

    def touch(self, conn: T) -> None:
        """Update the last activity timestamp for a connection.

        Args:
            conn: The connection to touch
        """
        if conn in self._connections:
            self._last_activity[conn] = time.monotonic()

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
            self._last_activity.pop(conn, None)

    def invalidate(self) -> None:
        """Clear all existing connections.

        Marks all current connections to be closed during the next drain cycle.
        """
        for conn in list(self._connections.keys()):
            self._to_close.add(conn)
        self._connections.clear()
        self._last_activity.clear()
        self._available.clear()

    async def aclose(self):
        """Close all connections, draining any pending connection closures."""
        self.invalidate()
        await self._drain_to_close()
