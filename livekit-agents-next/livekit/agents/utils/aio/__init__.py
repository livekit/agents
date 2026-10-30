"""asyncio helpers: channels, intervals, task sets and cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections import deque
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Generic, Protocol, TypeVar

__all__ = [
    "ChanClosed",
    "Chan",
    "Interval",
    "interval",
    "TaskSet",
    "cancel_and_wait",
    "ChanEmpty",
    "gracefully_cancel",
]

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
        #        self._finished_ev = asyncio.Event()
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
        #        if self.empty() and self._close_ev.is_set():
        #            self._finished_ev.set()
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


async def cancel_and_wait(*futures: _CancellableFuture) -> None:
    loop = asyncio.get_running_loop()
    waiters: list[tuple[asyncio.Future[None], Callable[[_CancellableFuture], None]]] = []

    for fut in futures:
        waiter = loop.create_future()
        cb: Callable[[_CancellableFuture], None] = functools.partial(_release_waiter, waiter)
        waiters.append((waiter, cb))
        fut.add_done_callback(cb)
        fut.cancel()

    try:
        for waiter, _ in waiters:
            await waiter
    finally:
        for i, fut in enumerate(futures):
            _, cb = waiters[i]
            fut.remove_done_callback(cb)
            if fut.done() and not fut.cancelled():
                fut.exception()


def _release_waiter(waiter: asyncio.Future[None], *_: object) -> None:
    if not waiter.done():
        waiter.set_result(None)


gracefully_cancel = cancel_and_wait
