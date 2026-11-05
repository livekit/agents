"""asyncio helpers: channels, intervals, task sets, cancellation and ``ConnectionPool``."""

from __future__ import annotations

import asyncio
import contextlib
import time
import weakref
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Generic, Protocol, TypeVar, overload

from .. import _core
from .._bindings import APITimeoutError
from ..log import logger
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class ChanClosed(Exception):
    pass


class ChanFull(Exception):
    pass


class ChanEmpty(Exception):
    pass


class ChanReceiver(Protocol[T_co]):
    async def recv(self) -> T_co: ...

    def recv_nowait(self) -> T_co: ...

    def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[T_co]: ...

    async def __anext__(self) -> T_co: ...


class Chan(Generic[T]):
    def __init__(
        self,
        maxsize: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        _on_receive: Callable[[T], None] | None = None,
        _on_eof: Callable[[], None] | None = None,
    ) -> None:
        self._on_receive = _on_receive
        self._on_eof = _on_eof
        self._eof_seen = False
        self._loop = loop or asyncio.get_event_loop()
        self._maxsize = max(maxsize, 0)
        self._close_ev = asyncio.Event()
        self._closed = False
        self._gets: deque[asyncio.Future[T | None]] = deque()
        self._puts: deque[asyncio.Future[T | None]] = deque()
        self._queue: deque[T] = deque()

    def _wakeup_next(self, waiters: deque[asyncio.Future[T | None]]) -> None:
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break

    async def send(self, value: T) -> None:
        while self.full() and not self._close_ev.is_set():
            p = self._loop.create_future()
            self._puts.append(p)
            try:
                await p
            except ChanClosed:
                raise
            except:
                p.cancel()
                with contextlib.suppress(ValueError):
                    self._puts.remove(p)

                if not self.full() and not p.cancelled():
                    self._wakeup_next(self._puts)
                raise

        self.send_nowait(value)

    def send_nowait(self, value: T) -> None:
        if self._close_ev.is_set():
            raise ChanClosed

        if self.full():
            raise ChanFull

        self._queue.append(value)
        self._wakeup_next(self._gets)

    async def recv(self) -> T:
        while self.empty() and not self._close_ev.is_set():
            g = self._loop.create_future()
            self._gets.append(g)

            try:
                await g
            except ChanClosed:
                self._eof()
                raise
            except BaseException:
                g.cancel()
                with contextlib.suppress(ValueError):
                    self._gets.remove(g)

                if not self.empty() and not g.cancelled():
                    self._wakeup_next(self._gets)

                raise

        return self.recv_nowait()

    def recv_nowait(self) -> T:
        if self.empty():
            if self._close_ev.is_set():
                self._eof()
                raise ChanClosed
            else:
                raise ChanEmpty
        item = self._queue.popleft()
        if self._on_receive is not None:
            self._on_receive(item)
        # the last item of a closed channel is its end, whether or not the reader reads past it
        if self._close_ev.is_set() and self.empty():
            self._eof()
        self._wakeup_next(self._puts)
        return item

    def _eof(self) -> None:
        if self._on_eof is not None and not self._eof_seen:
            self._eof_seen = True
            self._on_eof()

    def close(self) -> None:
        self._closed = True
        self._close_ev.set()
        while self._puts:
            putter = self._puts.popleft()
            if not putter.done():
                putter.set_exception(ChanClosed())

        while len(self._gets) > self.qsize():
            getter = self._gets.pop()
            if not getter.cancelled():
                getter.set_exception(ChanClosed())

        while self._gets:
            self._wakeup_next(self._gets)

    #        if self.empty():
    #            self._finished_ev.set()

    @property
    def closed(self) -> bool:
        return self._closed

    #    async def join(self) -> None:
    #        await self._finished_ev.wait()

    def qsize(self) -> int:
        """the number of elements queued (unread) in the channel buffer"""
        return len(self._queue)

    def full(self) -> bool:
        if self._maxsize <= 0:
            return False
        else:
            return self.qsize() >= self._maxsize

    def empty(self) -> bool:
        return not self._queue

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.recv()
        except ChanClosed:
            raise StopAsyncIteration from None


def _finish_fut(fut: asyncio.Future[None]) -> None:
    if fut.cancelled():
        return
    fut.set_result(None)


# MissedBehaviour is "Delay"
class Interval:
    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._last_sleep = 0.0
        self._i = 0
        self._handler: asyncio.TimerHandle | None = None
        self._fut: asyncio.Future[None] | None = None

    def reset(self) -> None:
        if self._fut and self._handler and not self._handler.cancelled():
            self._handler.cancel()
            loop = asyncio.get_event_loop()
            self._handler = loop.call_later(self._interval, _finish_fut, self._fut)
        else:
            self._last_sleep = 0

    async def tick(self) -> int:
        loop = asyncio.get_event_loop()

        if self._last_sleep:
            self._fut = loop.create_future()
            delay = self._last_sleep - loop.time() + self._interval
            self._handler = loop.call_later(delay, _finish_fut, self._fut)
            try:
                await self._fut
            finally:
                self._handler.cancel()
            self._i += 1

        self._last_sleep = loop.time()
        return self._i

    def __aiter__(self) -> Interval:
        return self

    async def __anext__(self) -> int:
        return await self.tick()


def interval(interval: float) -> Interval:
    return Interval(interval)


_T = TypeVar("_T")


class TaskSet:
    """Small utility to create tasks in a fire-and-forget fashion."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._set = set[asyncio.Task[object]]()
        self._closed = False

    def create_task(
        self, coro: Coroutine[object, object, _T], name: str | None = None
    ) -> asyncio.Task[_T]:
        if self._closed:
            raise RuntimeError("TaskSet is closed")

        task = self._loop.create_task(coro, name=name)
        self._set.add(task)
        task.add_done_callback(self._set.remove)
        return task

    @property
    def tasks(self) -> set[asyncio.Task[object]]:
        return self._set.copy()


class _CancellableFuture(Protocol):
    def cancel(self, msg: object = None) -> bool: ...
    def done(self) -> bool: ...
    def cancelled(self) -> bool: ...
    def exception(self) -> BaseException | None: ...
    def add_done_callback(self, callback: Callable[[_CancellableFuture], object], /) -> None: ...
    def remove_done_callback(self, callback: Callable[[_CancellableFuture], object], /) -> int: ...


# Process-scoped references, not a second cancellation/timeout policy. A cancelled waiter or
# expired native budget cannot release an authored finally block that still runs on its loop.
# Only the future's completion releases this reference; job/process shutdown owns the loops.
_pending_cleanup: set[_CancellableFuture] = set()


def _cleanup_done(future: _CancellableFuture) -> None:
    _pending_cleanup.discard(future)
    if not future.cancelled():
        future.exception()


def _retain_cleanup(future: _CancellableFuture) -> None:
    if future not in _pending_cleanup:
        _pending_cleanup.add(future)
        future.add_done_callback(_cleanup_done)


def pending_cleanup_count() -> int:
    """Process-wide host cleanup references awaiting completion callbacks, across job loops."""
    return len(_pending_cleanup)


async def await_cleanup(cleanup: Awaitable[_T]) -> _T:
    """Start cleanup once and retain it to settlement if its waiter is cancelled.

    The operation supplying ``cleanup`` owns its deadline. Cancelling this wait does not cancel
    cleanup a second time or abandon native stream closure behind an authored finally block.
    """
    future = asyncio.ensure_future(cleanup)
    _retain_cleanup(future)
    return await asyncio.shield(future)


@overload
async def cancel_and_wait(*futures: _CancellableFuture) -> None: ...


@overload
async def cancel_and_wait(*futures: _CancellableFuture, timeout: float) -> bool: ...


async def cancel_and_wait(
    *futures: _CancellableFuture, timeout: float | None = None
) -> bool | None:
    """Cancel once and wait for settlement, retaining unfinished tasks after an interrupted wait.

    With an operation-supplied ``timeout``, return whether every future settled within it.
    Rejoining cleanup never injects another cancellation into a task's authored finally block.
    """
    pending = {future for future in futures if not future.done()}
    for future in futures:
        if future.done() and not future.cancelled():
            future.exception()
    if not pending:
        return None if timeout is None else True
    observed = tuple(pending)
    waiter = asyncio.get_running_loop().create_future()

    def settled(future: _CancellableFuture) -> None:
        pending.discard(future)
        if not pending and not waiter.done():
            waiter.set_result(None)

    for future in observed:
        _retain_cleanup(future)
        future.add_done_callback(settled)
        if not isinstance(future, asyncio.Task) or not future.cancelling():
            future.cancel()

    try:
        if timeout is None:
            await waiter
            return None
        _, waiting = await asyncio.wait((waiter,), timeout=timeout)
        return not waiting
    finally:
        for future in observed:
            future.remove_done_callback(settled)
        waiter.cancel()


gracefully_cancel = cancel_and_wait


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
        connect_cb: Callable[[APIConnectOptions], Awaitable[T]] | None = None,
        close_cb: Callable[[T], Awaitable[None]] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Initialize the connection wrapper.

        Args:
            max_session_duration: Maximum duration in seconds before forcing reconnection
            mark_refreshed_on_get: If True, the session will be marked as fresh when get() is called. only used when max_session_duration is set.
            connect_cb: Optional async callback to create new connections, within the connect
                (and, over HTTP, pool and read) limits of the ``conn_options`` it is given
            close_cb: Optional async callback to close connections
            conn_options: The limits of a connection :meth:`prewarm` opens
        """  # noqa: E501
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._conn_options = conn_options
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

    async def _connect(
        self, conn_options: APIConnectOptions, *, started_at: float | None = None
    ) -> T:
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        while True:
            generation = self._state.generation
            conn = await self._connect_cb(conn_options)
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
    async def connection(self, *, conn_options: APIConnectOptions) -> AsyncGenerator[T, None]:
        """Get a connection from the pool and automatically return it when done.

        Yields:
            An active connection object
        """
        conn = await self.get(conn_options=conn_options)
        try:
            yield conn
        except BaseException:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(self, *, conn_options: APIConnectOptions) -> T:
        """Get an available connection or create a new one if needed.

        Waiting for the pool runs under ``conn_options.pool_timeout``; a new connection is opened
        by ``connect_cb`` under the limits of ``conn_options``.

        Returns:
            An active connection object
        """
        try:
            async with asyncio.timeout(conn_options.pool_timeout):
                await self._connect_lock.acquire()
        except TimeoutError as exc:
            raise APITimeoutError(
                "Timed out waiting for connection capacity", phase="pool"
            ) from exc
        try:
            await self._drain_to_close()
            if (cid := self._state.get(time.time())) is not None:
                return self._conns[cid]
            try:
                return await self._connect(conn_options, started_at=time.perf_counter())
            except TimeoutError as exc:
                # a connect_cb whose transport has no finer phases bounds it by connect_timeout
                raise APITimeoutError(
                    "Timed out establishing a connection", phase="connect"
                ) from exc
        finally:
            self._connect_lock.release()

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
                        self.put(await self._connect(self._conn_options))
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
            await gracefully_cancel(task)
        # the retired connections too, rather than waiting for holders that may never return them
        self._state.close_all()
        await self._drain_to_close()


__all__ = [
    "ChanClosed",
    "Chan",
    "Interval",
    "interval",
    "TaskSet",
    "cancel_and_wait",
    "await_cleanup",
    "pending_cleanup_count",
    "ChanEmpty",
    "gracefully_cancel",
]
