"""
Asynchronous channel implementation inspired by Go's channels.

Provides thread-safe, async/await compatible message passing with:
- Buffered/unbuffered channels
- Blocking/non-blocking operations
- Close notifications
- Iterable interface
"""

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
    """Raised when operating on a closed channel."""


class ChanFull(Exception):
    """Raised when sending to a full channel using non-blocking methods."""


class ChanEmpty(Exception):
    """Raised when receiving from an empty channel using non-blocking methods."""


class ChanSender(Protocol[T_contra]):
    """Protocol defining the sending interface of a channel."""
    async def send(self, value: T_contra) -> None:
        """Async send that waits until space is available."""
        ...

    def send_nowait(self, value: T_contra) -> None:
        """Non-blocking send that raises ChanFull if unable to send immediately."""
        ...

    def close(self) -> None:
        """Close the channel and notify all receivers."""
        ...


class ChanReceiver(Protocol[T_co]):
    """Protocol defining the receiving interface of a channel."""
    async def recv(self) -> T_co:
        """Async receive that waits until data is available."""
        ...

    def recv_nowait(self) -> T_co:
        """Non-blocking receive that raises ChanEmpty if no data available."""
        ...

    def close(self) -> None:
        """Close the channel and notify all senders."""
        ...

    def __aiter__(self) -> AsyncIterator[T_co]:
        """Allow async iteration over received values."""
        ...

    async def __anext__(self) -> T_co:
        """Get next value through async iteration."""
        ...


class Chan(Generic[T]):
    """Asynchronous channel implementation for safe message passing between coroutines.
    
    Features:
    - Fixed-size buffering (when maxsize > 0)
    - Thread-safe operations
    - Async iterator support
    - Close propagation
    
    Usage:
        chan = Chan[int](maxsize=5)
        
        async def producer():
            await chan.send(42)
            chan.close()
            
        async def consumer():
            async for value in chan:
                print(value)
    """
    
    def __init__(
        self,
        maxsize: int = 0,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """
        Args:
            maxsize: Maximum channel capacity (0 = unbuffered)
            loop: Event loop to use (defaults to current)
        """
        self._loop = loop or asyncio.get_event_loop()
        self._maxsize = max(maxsize, 0)
        #        self._finished_ev = asyncio.Event()
        self._close_ev = asyncio.Event()
        self._closed = False
        self._gets: Deque[asyncio.Future[T | None]] = deque()
        self._puts: Deque[asyncio.Future[T | None]] = deque()
        self._queue: Deque[T] = deque()

    def _wakeup_next(self, waiters: deque[asyncio.Future[T | None]]):
        """Wake the next waiting coroutine in the queue."""
        while waiters:
            waiter = waiters.popleft()
            if not waiter.done():
                waiter.set_result(None)
                break

    async def send(self, value: T) -> None:
        """Send a value to the channel, waiting if necessary.
        
        Args:
            value: Data to send to the channel
            
        Raises:
            ChanClosed: If channel is closed during send
        """
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
        """Send without blocking, raising ChanFull if unable.
        
        Args:
            value: Data to send immediately
            
        Raises:
            ChanFull: If channel is at capacity
            ChanClosed: If channel is closed
        """
        if self.full():
            raise ChanFull

        if self._close_ev.is_set():
            raise ChanClosed

        self._queue.append(value)
        self._wakeup_next(self._gets)

    async def recv(self) -> T:
        """Receive a value, waiting until one is available.
        
        Returns:
            T: Received value
            
        Raises:
            ChanClosed: If channel is closed during receive
        """
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
        """Receive without blocking, raising ChanEmpty if no data.
        
        Returns:
            T: First available value
            
        Raises:
            ChanEmpty: If no data available
            ChanClosed: If channel is closed
        """
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
        """Close the channel and notify all waiters."""
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
        """Check if channel is closed."""
        return self._closed

    #    async def join(self) -> None:
    #        await self._finished_ev.wait()

    def qsize(self) -> int:
        """Current number of items in the buffer."""
        return len(self._queue)

    def full(self) -> bool:
        """Check if channel is at capacity."""
        if self._maxsize <= 0:
            return False
        else:
            return self.qsize() >= self._maxsize

    def empty(self) -> bool:
        """Check if channel buffer is empty."""
        return not self._queue

    def __aiter__(self) -> AsyncIterator[T]:
        """Support async iteration until channel closes."""
        return self

    async def __anext__(self) -> T:
        """Implement async iteration protocol."""
        try:
            return await self.recv()
        except ChanClosed:
            raise StopAsyncIteration
