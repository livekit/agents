"""asyncio helpers: channels, intervals and sleeps, task sets and wait groups, async tees and the
slow-callback hook."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from asyncio.base_events import _format_handle  # type: ignore
from collections import deque
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Generator,
    Iterator,
)
from types import TracebackType
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

from typing_extensions import AsyncContextManager

from ...log import logger

__all__ = [
    "AsyncAtomicCounter",
    "ChanClosed",
    "Chan",
    "ChanSender",
    "ChanReceiver",
    "Interval",
    "interval",
    "Sleep",
    "SleepFinished",
    "sleep",
    "TaskSet",
    "WaitGroup",
    "cancel_and_wait",
    "ChanEmpty",
    "Debounced",
    "debounced",
    "Tee",
    "tee",
    "hook_slow_callbacks",
    "gracefully_cancel",
]

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


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


class AsyncAtomicCounter:
    """Async atomic counter implementation."""

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = asyncio.Lock()

    async def increment(self, n: int = 1) -> int:
        async with self._lock:
            self._value += n
            return self._value

    async def decrement(self, n: int = 1) -> int:
        async with self._lock:
            self._value -= n
            return self._value

    async def get(self) -> int:
        async with self._lock:
            return self._value

    async def set(self, value: int) -> None:
        async with self._lock:
            self._value = value

    async def compare_and_swap(self, expected: int, new: int) -> bool:
        async with self._lock:
            if self._value == expected:
                self._value = new
                return True
            return False

    async def get_and_reset(self, reset_value: int = 0) -> int:
        """Atomically read the current value and reset it."""
        async with self._lock:
            prev = self._value
            self._value = reset_value
            return prev

    def get_nowait(self) -> int:
        """Best-effort non-async read — safe if no await between read and use."""
        return self._value


class Debounced(Generic[T]):
    def __init__(self, func: Callable[[], Coroutine[object, object, T]], delay: float) -> None:
        self._func = func
        self._delay = delay
        self._task: asyncio.Task[T] | None = None

    def schedule(self) -> asyncio.Task[T]:
        self.cancel()

        async def _func_with_timer() -> T:
            await asyncio.sleep(self._delay)
            return await self._func()

        self._task = asyncio.create_task(_func_with_timer())
        return self._task

    def run(self) -> asyncio.Task[T]:
        self.cancel()

        self._task = asyncio.create_task(self._func())
        return self._task

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._task.cancelled()

    def __call__(self) -> asyncio.Task[T]:
        return self.run()


def debounced(delay: float) -> Callable[[Callable[[], Coroutine[object, object, T]]], Debounced[T]]:
    def decorator(func: Callable[[], Coroutine[object, object, T]]) -> Debounced[T]:
        return Debounced(func, delay)

    return decorator


def hook_slow_callbacks(slow_duration: float) -> None:
    _run = asyncio.events.Handle._run

    def instrumented(self: asyncio.Handle) -> None:
        start = time.monotonic()
        val: None = _run(self)
        dt = time.monotonic() - start
        if dt >= slow_duration:
            logger.warning("Running %s took too long: %.2f seconds", str(_format_handle(self)), dt)
        return val

    asyncio.events.Handle._run = instrumented  # type: ignore


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


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self) -> None:
        """Asynchronously close this object"""


async def tee_peer(
    iterator: AsyncIterator[T],
    buffer: deque[T],
    peers: list[deque[T]],
    lock: AsyncContextManager[object],
    exception: list[BaseException | None],
) -> AsyncGenerator[T, None]:
    # exception is a shared mutable container across all peers. When the upstream
    # iterator raises, only the first peer to call __anext__() would normally see
    # the error — subsequent calls return StopAsyncIteration per Python async
    # generator semantics, silently swallowing the error for other peers.
    #
    # To fix this, the first peer to hit the exception stores it in exception[0].
    # Other peers check this before advancing the iterator and re-raise the same
    # exception, ensuring all peers observe the upstream failure.
    try:
        while True:
            if not buffer:
                async with lock:
                    if buffer:
                        continue
                    # a peer already hit an upstream error — re-raise for this peer
                    if exception[0] is not None:
                        raise exception[0]
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    except asyncio.CancelledError:
                        # CancelledError is task-specific — don't store it in the
                        # shared exception list to avoid cascading to other peers
                        raise
                    except BaseException as e:
                        exception[0] = e
                        raise
                    else:
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        for idx, peer_buffer in enumerate(peers):  # pragma: no branch
            if peer_buffer is buffer:
                peers.pop(idx)
                break

        if not peers and isinstance(iterator, _ACloseable):
            await iterator.aclose()


class Tee(Generic[T]):
    __slots__ = ("_iterator", "_buffers", "_children")

    def __init__(
        self,
        iterator: AsyncIterable[T],
        n: int = 2,
    ):
        self._iterator = iterator.__aiter__()
        self._buffers: list[deque[T]] = [deque() for _ in range(n)]

        lock = asyncio.Lock()
        exception: list[BaseException | None] = [None]
        self._children = tuple(
            tee_peer(
                iterator=self._iterator,
                buffer=buffer,
                peers=self._buffers,
                lock=lock,
                exception=exception,
            )
            for buffer in self._buffers
        )

    def __len__(self) -> int:
        return len(self._children)

    @overload
    def __getitem__(self, item: int) -> AsyncIterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> tuple[AsyncIterator[T], ...]: ...

    def __getitem__(self, item: int | slice) -> AsyncIterator[T] | tuple[AsyncIterator[T], ...]:
        return self._children[item]

    def __iter__(self) -> Iterator[AsyncIterator[T]]:
        yield from self._children

    async def __aenter__(self) -> Tee[T]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for child in self._children:
            try:
                await child.aclose()
            except Exception:
                pass

        if isinstance(self._iterator, _ACloseable):
            try:
                await self._iterator.aclose()
            except Exception:
                pass


tee = Tee


class SleepFinished(Exception):
    pass


class Sleep:
    """Same as asyncio.sleep except it is resettable"""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._handler: asyncio.TimerHandle | None = None

    def reset(self, new_delay: float | None = None) -> None:
        if new_delay is None:
            new_delay = self._delay

        self._delay = new_delay

        if self._handler is None:
            return

        if self._handler.cancelled() or self._fut.done():
            raise SleepFinished

        self._handler.cancel()
        loop = asyncio.get_event_loop()
        self._handler = loop.call_later(new_delay, _finish_fut, self._fut)

    def cancel(self) -> None:
        if self._handler is None:
            return

        self._handler.cancel()
        self._fut.cancel()

    async def _sleep(self) -> None:
        if self._delay <= 0:
            self._fut = asyncio.Future[None]()
            self._fut.set_result(None)
            return

        loop = asyncio.get_event_loop()
        self._fut = loop.create_future()
        self._handler = loop.call_later(self._delay, _finish_fut, self._fut)

        try:
            await asyncio.shield(self._fut)
        finally:
            self._handler.cancel()

    def __await__(self) -> Generator[object, None, None]:
        return self._sleep().__await__()


def sleep(delay: float) -> Sleep:
    return Sleep(delay)


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


class WaitGroup:
    """
    asyncio wait group implementation (similar to sync.WaitGroup in go)
    """

    def __init__(self) -> None:
        self._counter = 0
        self._zero_event = asyncio.Event()
        self._zero_event.set()

    def add(self, delta: int = 1) -> None:
        new_value = self._counter + delta
        if new_value < 0:
            raise ValueError("WaitGroup counter cannot go negative.")

        self._counter = new_value

        if self._counter == 0:
            self._zero_event.set()
        else:
            self._zero_event.clear()

    def done(self) -> None:
        self.add(-1)

    async def wait(self) -> None:
        await self._zero_event.wait()
