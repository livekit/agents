from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Generic, Protocol, TypeVar, overload

from .. import _core
from .._bindings import APITimeoutError
from .._generated import facades_utils_aio as _facades
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
        self._loop = loop
        self._maxsize = max(maxsize, 0)
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
        while self.full() and not self._closed:
            p = (self._loop or asyncio.get_running_loop()).create_future()
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
        if self._closed:
            raise ChanClosed

        if self.full():
            raise ChanFull

        self._queue.append(value)
        self._wakeup_next(self._gets)

    async def recv(self) -> T:
        while self.empty() and not self._closed:
            g = (self._loop or asyncio.get_running_loop()).create_future()
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
            if self._closed:
                self._eof()
                raise ChanClosed
            else:
                raise ChanEmpty
        item = self._queue.popleft()
        if self._on_receive is not None:
            self._on_receive(item)
        # the last item of a closed channel is its end, whether or not the reader reads past it
        if self._closed and self.empty():
            self._eof()
        self._wakeup_next(self._puts)
        return item

    def _eof(self) -> None:
        if self._on_eof is not None and not self._eof_seen:
            self._eof_seen = True
            self._on_eof()

    def close(self) -> None:
        self._closed = True
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

    @property
    def closed(self) -> bool:
        return self._closed

    def qsize(self) -> int:
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
            loop = asyncio.get_running_loop()
            self._handler = loop.call_later(self._interval, _finish_fut, self._fut)
        else:
            self._last_sleep = 0

    async def tick(self) -> int:
        loop = asyncio.get_running_loop()

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


class _CancellableFuture(Protocol):
    def cancel(self, msg: object = None) -> bool: ...
    def done(self) -> bool: ...
    def cancelled(self) -> bool: ...
    def exception(self) -> BaseException | None: ...
    def add_done_callback(self, callback: Callable[[_CancellableFuture], object], /) -> None: ...
    def remove_done_callback(self, callback: Callable[[_CancellableFuture], object], /) -> int: ...


# Strong references so a cancelled waiter cannot drop a cleanup whose finally block still runs;
# only the future's completion releases one.
_pending_cleanup: set[_CancellableFuture] = set()


def _cleanup_done(future: _CancellableFuture) -> None:
    _pending_cleanup.discard(future)
    if not future.cancelled():
        future.exception()


def _retain_cleanup(future: _CancellableFuture) -> None:
    if future not in _pending_cleanup:
        _pending_cleanup.add(future)
        future.add_done_callback(_cleanup_done)


def _pending_cleanup_count() -> int:
    """The number of cleanups still running in this process, across job loops."""
    return len(_pending_cleanup)


async def _await_cleanup(cleanup: Awaitable[_T]) -> _T:
    """Start ``cleanup`` once and keep it running to completion if this wait is cancelled.

    ``cleanup`` owns its deadline; cancelling the wait never cancels it.
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
    """Cancel each future once and wait for them to settle.

    Futures still running when the wait is interrupted are kept alive until they finish; a task
    already cancelling is not cancelled again. With ``timeout``, returns whether every future
    settled within it.
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


class TaskSet:
    """The owner of a group of asyncio tasks.

    It holds each task strongly until it finishes, raises a task's failure to whoever awaits the
    task and logs one nobody awaits, and cancels and joins the tasks still running when it
    closes: on :meth:`aclose`, or on leaving its ``async with`` block.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def spawn(
        self,
        aw: Awaitable[T],
        *,
        name: str | None = None,
        context: contextvars.Context | None = None,
    ) -> asyncio.Task[T]:
        """Run ``aw`` as a task this set owns."""
        coro = aw if isinstance(aw, Coroutine) else _as_coroutine(aw)
        if self._closed:
            coro.close()
            raise RuntimeError("TaskSet is closed")
        task = asyncio.create_task(coro, name=name, context=context)
        self._tasks.add(task)
        task.add_done_callback(self._done)
        return task

    def _done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled():
            # an awaiter takes the exception when it wakes, in a callback queued before this one
            task.get_loop().call_soon(_report_unobserved, task)

    async def first_completed(self, *aws: Awaitable[Any]) -> asyncio.Future[Any]:
        """Wait for the first of ``aws`` to finish, then cancel and join the others.

        Returns the first to finish and raises its failure. Every awaitable is the call's to
        cancel: a coroutine runs as a task of this set.
        """
        futures = [aw if isinstance(aw, asyncio.Future) else self.spawn(aw) for aw in aws]
        first: asyncio.Future[asyncio.Future[Any]] = asyncio.get_running_loop().create_future()

        def finished(future: asyncio.Future[Any]) -> None:
            if not future.cancelled():
                future.exception()  # raised below for the winner, dropped for a later one
            if not first.done():
                first.set_result(future)

        for future in futures:
            future.add_done_callback(finished)
        try:
            winner = await first
        finally:
            for future in futures:
                future.remove_done_callback(finished)
            await cancel_and_wait(*(future for future in futures if not future.done()))
        winner.result()
        return winner

    async def aclose(self) -> None:
        """Cancel every task still running and wait for them to finish.

        A task of this set may close it: that task is neither cancelled nor waited for.
        """
        self._closed = True
        current = asyncio.current_task()
        await cancel_and_wait(*(task for task in self._tasks if task is not current))

    async def __aenter__(self) -> TaskSet:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


async def _as_coroutine(aw: Awaitable[T]) -> T:
    return await aw


def _report_unobserved(task: asyncio.Task[Any]) -> None:
    if task._log_traceback:
        logger.error("task %s failed", task.get_name(), exc_info=task.exception())


class ConnectionPool(_facades.ConnectionPool, Generic[T]):
    """A pool of persistent connections, such as websockets, reconnected after a max duration.

    ``connection()`` checks one out and returns it to the pool when the block exits.
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
        """
        Args:
            max_session_duration: Seconds before a connection is replaced.
            mark_refreshed_on_get: Restart a connection's duration on each ``get()``; only used
                with ``max_session_duration``.
            connect_cb: Opens a connection within the limits of the ``conn_options`` it is given.
            close_cb: Closes a connection.
            conn_options: The limits of a connection :meth:`prewarm` opens.
        """
        self._connect_cb = connect_cb
        self._close_cb = close_cb
        self._conn_options = conn_options
        self._native = _core.ConnectionPoolState(max_session_duration, mark_refreshed_on_get)
        self._conns: dict[int, T] = {}  # every connection the state knows, by id()
        self._connect_lock = asyncio.Lock()
        self._tasks = TaskSet()
        self._prewarm_task: asyncio.Task[None] | None = None

    async def _connect(
        self, conn_options: APIConnectOptions, *, started_at: float | None = None
    ) -> T:
        if self._connect_cb is None:
            raise NotImplementedError("Must provide connect_cb or implement connect()")
        while True:
            generation = self._native.generation
            conn = await self._connect_cb(conn_options)
            self._conns[id(conn)] = conn
            elapsed = None if started_at is None else time.perf_counter() - started_at
            if self._native.connected(id(conn), time.time(), generation, elapsed):
                return conn
            # options changed during the handshake: the state queued this socket for closing
            self._close_queued()

    def _close_queued(self) -> None:
        # each taken id gets its close task in this same step: a cancelled caller cannot drop one
        for cid in self._native.take_to_close():
            self._tasks.spawn(self._close(self._conns.pop(cid)), name="ConnectionPool.close")

    async def _close(self, conn: T) -> None:
        try:
            if self._close_cb is not None:
                await self._close_cb(conn)
        except Exception as e:
            logger.warning("error closing connection %s: %s", conn, e)

    @asynccontextmanager
    async def connection(self, *, conn_options: APIConnectOptions) -> AsyncGenerator[T, None]:
        """A connection from the pool, returned on exit or removed if the block raises."""
        conn = await self.get(conn_options=conn_options)
        try:
            yield conn
        except BaseException:
            self.remove(conn)
            raise
        else:
            self.put(conn)

    async def get(self, *, conn_options: APIConnectOptions) -> T:
        """An available connection, or a new one opened under ``conn_options``' limits.

        Waiting for the pool runs under ``conn_options.pool_timeout``.
        """
        self._close_queued()
        try:
            async with asyncio.timeout(conn_options.pool_timeout):
                await self._connect_lock.acquire()
        except TimeoutError as exc:
            raise APITimeoutError(
                "Timed out waiting for connection capacity", timeout="pool"
            ) from exc
        try:
            if (cid := self._native.get(time.time())) is not None:
                return self._conns[cid]
            try:
                return await self._connect(conn_options, started_at=time.perf_counter())
            except TimeoutError as exc:
                # a connect_cb whose transport has no finer phases bounds it by connect_timeout
                raise APITimeoutError(
                    "Timed out establishing a connection", timeout="connect"
                ) from exc
        finally:
            self._connect_lock.release()

    def put(self, conn: T) -> None:
        """Make a connection available for reuse; one removed or retired by :meth:`invalidate`
        is closed instead."""
        self._native.put(id(conn))
        self._close_queued()

    def remove(self, conn: T) -> None:
        """Remove a connection from the pool and close it."""
        self._native.remove(id(conn))
        self._close_queued()

    def invalidate(self) -> None:
        """Stop reusing every existing connection.

        Idle connections are closed. Checked-out ones keep working for their holder and are
        closed when returned, so a request in flight is not interrupted. A handshake in flight is
        discarded and retried, since its socket has the old settings.
        """
        self._native.invalidate()
        self._close_queued()

    def prewarm(self) -> None:
        """Open a connection in the background if the pool has none."""
        prewarming = self._prewarm_task is not None and not self._prewarm_task.done()
        if prewarming or self._tasks.closed or not self._native.empty():
            return

        async def _prewarm_impl() -> None:
            try:
                async with self._connect_lock:
                    if self._native.empty():
                        self.put(await self._connect(self._conn_options))
            except Exception as e:
                # exception details can contain request headers or URL credentials.
                logger.warning(
                    "failed to prewarm connection pool",
                    extra={"exception_type": type(e).__name__},
                )

        self._prewarm_task = self._tasks.spawn(_prewarm_impl(), name="ConnectionPool.prewarm")

    async def aclose(self) -> None:
        """Close every connection, including checked-out ones."""
        if self._prewarm_task is not None:
            await cancel_and_wait(self._prewarm_task)
        # the retired connections too, rather than waiting for holders that may never return them
        self._native.close_all()
        self._close_queued()
        # the closes run to completion, each bounded by close_cb; then the set closes
        if closing := [task for task in self._tasks._tasks if not task.done()]:
            await asyncio.wait(closing)
        await self._tasks.aclose()


# A close nobody awaits (a dropped stream's ``aclose()``), by loop: held until it finishes, since
# asyncio holds a task only weakly, and awaited by ``close_loop``.
_closing: dict[asyncio.AbstractEventLoop, set[asyncio.Task[object]]] = {}
# How long ``close_loop`` gives those closes, the core's teardown bound.
_CLOSING_BOUND = 5.0


def _keep_closing(task: asyncio.Task[object]) -> None:
    loop = task.get_loop()
    held = _closing.setdefault(loop, set())
    held.add(task)

    def done(task: asyncio.Task[object]) -> None:
        held.discard(task)
        if not held and _closing.get(loop) is held:
            del _closing[loop]

    task.add_done_callback(done)


def close_loop(loop: asyncio.AbstractEventLoop) -> list[Exception]:
    """``asyncio.run()``'s teardown of ``loop``, then close it, returning what its tasks raised.

    The closes still running for dropped streams finish first, bounded. Cancelling runs again
    after the generators and the default executor shut down: a task they started (a host call
    the framework is waiting on) is cancelled, never left pending in a closed loop with its
    waiter asleep.
    """
    failures: list[Exception] = []

    def cancel_all() -> None:
        while tasks := [task for task in asyncio.all_tasks(loop) if not task.done()]:
            for task in tasks:
                task.cancel()
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            failures.extend(result for result in results if isinstance(result, Exception))

    try:
        if closing := _closing.pop(loop, None):
            loop.run_until_complete(asyncio.wait(closing, timeout=_CLOSING_BOUND))
        cancel_all()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        cancel_all()
    finally:
        loop.close()
    return failures


__all__ = [
    "TaskSet",
    "ChanClosed",
    "Chan",
    "Interval",
    "interval",
    "cancel_and_wait",
    "ChanEmpty",
]
