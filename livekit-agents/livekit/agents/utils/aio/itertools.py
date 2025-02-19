"""
Asynchronous iterator utilities for advanced iteration patterns.

Provides async-compatible versions of common itertools functionality.
"""

import asyncio
from collections import deque
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Deque,
    Generic,
    Iterator,
    List,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

from typing_extensions import AsyncContextManager

# based on https://github.com/maxfischer2781/asyncstdlib/blob/master/asyncstdlib/itertools.py


@runtime_checkable
class _ACloseable(Protocol):
    """Protocol for objects requiring async cleanup."""
    async def aclose(self) -> None:
        """Asynchronously close this object"""


T = TypeVar("T")


async def tee_peer(
    iterator: AsyncIterator[T],
    buffer: Deque[T],
    peers: List[Deque[T]],
    lock: AsyncContextManager[Any],
) -> AsyncGenerator[T, None]:
    """Internal helper for tee implementation.
    
    Manages buffered items and synchronization between tee'd iterators.
    """
    try:
        while True:
            if not buffer:
                async with lock:
                    # Re-check buffer after acquiring lock
                    if buffer:
                        continue
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
                    else:
                        # Distribute new item to all peer buffers
                        for peer_buffer in peers:
                            peer_buffer.append(item)
            yield buffer.popleft()
    finally:
        # Remove self from peers and cleanup if last iterator
        for idx, peer_buffer in enumerate(peers):  # pragma: no branch
            if peer_buffer is buffer:
                peers.pop(idx)
                break

        if not peers and isinstance(iterator, _ACloseable):
            await iterator.aclose()


class Tee(Generic[T]):
    """Asynchronous iterator tee implementation.
    
    Splits a single async iterator into multiple independent iterators.
    
    Usage:
        async def numbers():
            for i in range(5):
                yield i
                await asyncio.sleep(0.1)
                
        tee = Tee(numbers(), n=3)
        async with tee:
            iter1, iter2, iter3 = tee
            async for num in iter1:
                print(num)
    """
    __slots__ = ("_iterator", "_buffers", "_children")

    def __init__(
        self,
        iterator: AsyncIterable[T],
        n: int = 2,
    ):
        """
        Args:
            iterator: Source async iterable to split
            n: Number of independent iterators to create
        """
        self._iterator = iterator.__aiter__()
        self._buffers: List[Deque[T]] = [deque() for _ in range(n)]

        lock = asyncio.Lock()
        self._children = tuple(
            tee_peer(
                iterator=self._iterator,
                buffer=buffer,
                peers=self._buffers,
                lock=lock,
            )
            for buffer in self._buffers
        )

    def __len__(self) -> int:
        """Get number of tee'd iterators."""
        return len(self._children)

    @overload
    def __getitem__(self, item: int) -> AsyncIterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> Tuple[AsyncIterator[T], ...]: ...

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[AsyncIterator[T], Tuple[AsyncIterator[T], ...]]:
        """Access tee'd iterators via index or slice."""
        return self._children[item]

    def __iter__(self) -> Iterator[AsyncIterator[T]]:
        """Iterate over tee'd iterators."""
        yield from self._children

    async def __aenter__(self) -> "Tee[T]":
        """Context manager entry point."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - ensures proper cleanup."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close all tee'd iterators and cleanup resources."""
        for child in self._children:
            await child.aclose()


tee = Tee
