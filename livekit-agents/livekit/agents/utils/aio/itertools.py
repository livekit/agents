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


async def _close_peer(iterator: AsyncIterator[T], buffer: deque[T], peers: list[deque[T]]) -> None:
    for idx, peer_buffer in enumerate(peers):
        if peer_buffer is buffer:
            peers.pop(idx)
            buffer.clear()
            if not peers and isinstance(iterator, _ACloseable):
                await iterator.aclose()
            return


async def tee_peer(
    iterator: AsyncIterator[T],
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
        await _close_peer(iterator, buffer, peers)


class _TeePeer(AsyncIterator[T]):
    def __init__(
        self,
        iterator: AsyncIterator[T],
        buffer: deque[T],
        peers: list[deque[T]],
        lock: AsyncContextManager[Any],
        exception: list[BaseException | None],
    ) -> None:
        self._iterator = iterator
        self._buffer = buffer
        self._peers = peers
        self._generator = tee_peer(iterator, buffer, peers, lock, exception)

    def __aiter__(self) -> "_TeePeer[T]":
        return self

    async def __anext__(self) -> T:
        return await self._generator.__anext__()

    async def aclose(self) -> None:
        await self._generator.aclose()
        # Closing an unstarted async generator does not execute its finally block.
        await _close_peer(self._iterator, self._buffer, self._peers)


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
            _TeePeer(
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

    async def __aenter__(self) -> "Tee[T]":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
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
