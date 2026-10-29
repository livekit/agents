import asyncio
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Iterator
from typing import Any, Generic, Protocol, TypeVar, overload, runtime_checkable

from typing_extensions import AsyncContextManager

# based on https://github.com/maxfischer2781/asyncstdlib/blob/master/asyncstdlib/itertools.py


@runtime_checkable
class _ACloseable(Protocol):
    async def aclose(self) -> None:
        """Asynchronously close this object"""


T = TypeVar("T")


class _TeeSource(Generic[T]):
    """Own upstream iteration and cleanup in one task, without prefetching."""

    def __init__(self, iterator: AsyncIterator[T]) -> None:
        self._iterator = iterator
        self._request = asyncio.Event()
        self._response: asyncio.Future[T] | None = None
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._closing = False
        self._cleaning_up = False

    async def next(self) -> T:
        """Read on demand, retaining the result if its waiting peer is cancelled."""
        if self._closing:
            raise StopAsyncIteration
        if self._response is None:
            self._response = asyncio.get_running_loop().create_future()
            self._request.set()
        if self._task is None:
            self._task = asyncio.create_task(self._run())
            # A source may fail cleanup while other peers still have buffered
            # items. Observe the failure now; aclose still re-raises it later.
            self._task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )

        # A cancelled peer relinquishes its wait, not the shared upstream read.
        # Keep the response for the next peer until its item has been delivered.
        item = await asyncio.shield(self._response)
        self._response = None
        return item

    async def _run(self) -> None:
        self._started = True
        try:
            while not self._closing:
                await self._request.wait()
                self._request.clear()
                response = self._response
                assert response is not None
                try:
                    item = await self._iterator.__anext__()
                except asyncio.CancelledError:
                    response.cancel()
                    raise
                except BaseException as error:
                    response.set_exception(error)
                    return
                else:
                    response.set_result(item)
        finally:
            self._cleaning_up = True
            if self._response is not None and not self._response.done():
                self._response.cancel()
            if isinstance(self._iterator, _ACloseable):
                await self._iterator.aclose()

    async def aclose(self) -> None:
        """Stop the producer and await upstream cleanup in its original context."""
        task = self._task
        if task is None:
            self._closing = True
            if isinstance(self._iterator, _ACloseable):
                await self._iterator.aclose()
            return

        if not self._closing:
            self._closing = True
            # A task cancelled before its first step never enters its finally
            # block. Let an unstarted producer run just its cleanup instead.
            if self._started and not self._cleaning_up:
                task.cancel()
        # Only cancel the producer once. A second close or a cancelled close
        # caller must not interrupt an upstream finally block already running.
        (result,) = await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if self._response is not None:
            if not self._response.done():
                self._response.cancel()
            elif not self._response.cancelled():
                self._response.exception()
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


async def tee_peer(
    source: _TeeSource[T],
    buffer: deque[T],
    peers: list[deque[T]],
    lock: AsyncContextManager[Any],
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
                        item = await source.next()
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

        if not peers:
            await source.aclose()


class Tee(Generic[T]):
    __slots__ = ("_source", "_buffers", "_children")

    def __init__(
        self,
        iterator: AsyncIterable[T],
        n: int = 2,
    ):
        self._source = _TeeSource(iterator.__aiter__())
        self._buffers: list[deque[T]] = [deque() for _ in range(n)]

        lock = asyncio.Lock()
        exception: list[BaseException | None] = [None]
        self._children = tuple(
            tee_peer(
                source=self._source,
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

    async def __aenter__(self) -> "Tee[T]":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        # Cancel/await an in-flight read before closing its consumer generators.
        try:
            await self._source.aclose()
        except Exception:
            pass

        for child in self._children:
            try:
                await child.aclose()
            except Exception:
                pass


tee = Tee
