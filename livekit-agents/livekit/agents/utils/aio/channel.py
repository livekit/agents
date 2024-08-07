from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from typing import AsyncIterator, Deque, Generic, Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


# Based on asyncio.Queue, see https://github.com/python/cpython/blob/main/Lib/asyncio/queues.py


class ChanClosed(Exception):
    pass


class ChanFull(Exception):
    pass


class ChanEmpty(Exception):
    pass


class ChanSender(Protocol[T_contra]):
    async def send(self, value: T_contra) -> None: ...

    def send_nowait(self, value: T_contra) -> None: ...

    def close(self) -> None: ...


class ChanReceiver(Protocol[T_co]):
    async def recv(self) -> T_co: ...

    def recv_nowait(self) -> T_co: ...

    def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[T_co]: ...

    async def __anext__(self) -> T_co: ...


class Chan(Generic[T]):
    def __init__(
        self, maxsize: int = 0, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._maxsize = max(maxsize, 0)
        #        self._finished_ev = asyncio.Event()
        self._close_ev = asyncio.Event()
        self._closed = False
        self._gets: Deque[asyncio.Future[T | None]] = deque()
        self._puts: Deque[asyncio.Future[T | None]] = deque()
        self._queue: Deque[T] = deque()

    def _wakeup_next(self, waiters: deque[asyncio.Future[T | None]]):
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
        if self.full():
            raise ChanFull

        if self._close_ev.is_set():
            raise ChanClosed

        self._queue.append(value)
        self._wakeup_next(self._gets)

    async def recv(self) -> T:
        while self.empty() and not self._close_ev.is_set():
            g = self._loop.create_future()
            self._gets.append(g)

            try:
                await g
            except ChanClosed:
                raise
            except Exception:
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
                raise ChanClosed
            else:
                raise ChanEmpty
        item = self._queue.popleft()
        #        if self.empty() and self._close_ev.is_set():
        #            self._finished_ev.set()
        self._wakeup_next(self._puts)
        return item

    def close(self) -> None:
        self._closed = True
        self._close_ev.set()
        for putter in self._puts:
            if not putter.cancelled():
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
            raise StopAsyncIteration
