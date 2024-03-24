import asyncio
import contextlib
from collections import deque
from typing import Any, Generic, Tuple, TypeVar

T = TypeVar("T", bound=Any)

# Based on asyncio.Queue, see https://github.com/python/cpython/blob/main/Lib/asyncio/queues.py


class ChanClosed(Exception):
    pass


class ChanFull(Exception):
    pass


class ChanEmpty(Exception):
    pass


def channel(
    maxsize: int = 0, loop: asyncio.AbstractEventLoop | None = None
) -> Tuple["ChanSender[T]", "ChanReceiver[T]"]:
    chan = Chan(maxsize, loop)
    return ChanSender(chan), ChanReceiver(chan)


class ChanSender(Generic[T]):
    def __init__(self, chan: "Chan[T]") -> None:
        self._chan = chan

    async def send(self, value: T) -> None:
        await self._chan.send(value)

    def send_nowait(self, value: T) -> None:
        self._chan.send_nowait(value)

    def close(self) -> None:
        self._chan.close()


class ChanReceiver(Generic[T]):
    def __init__(self, chan: "Chan[T]") -> None:
        self._chan = chan

    async def recv(self) -> T:
        return await self._chan.recv()

    def recv_nowait(self) -> T:
        return self._chan.recv_nowait()

    def close(self) -> None:
        self._chan.close()

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            return await self.recv()
        except ChanClosed:
            raise StopAsyncIteration


class Chan(Generic[T]):
    def __init__(
        self, maxsize: int = 0, loop: asyncio.AbstractEventLoop | None = None
    ) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._maxsize = max(maxsize, 0)
        #        self._finished_ev = asyncio.Event()
        self._close_ev = asyncio.Event()
        self._closed = False
        self._gets = deque()
        self._puts = deque()
        self._queue = deque()

    def _wakeup_next(self, waiters):
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break

    async def send(self, item):
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

        return self.send_nowait(item)

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
            except:
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
        return len(self._queue)

    def full(self) -> bool:
        if self._maxsize <= 0:
            return False
        else:
            return self.qsize() >= self._maxsize

    def empty(self) -> bool:
        return not self._queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.recv()
        except ChanClosed:
            raise StopAsyncIteration
