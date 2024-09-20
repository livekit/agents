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
    async def aclose(self) -> None:
        """Asynchronously close this object"""


T = TypeVar("T")


async def tee_peer(
    iterator: AsyncIterator[T],
    buffer: Deque[T],
    peers: List[Deque[T]],
    lock: AsyncContextManager[Any],
) -> AsyncGenerator[T, None]:
    try:
        while True:
            if not buffer:
                async with lock:
                    if buffer:
                        continue
                    try:
                        item = await iterator.__anext__()
                    except StopAsyncIteration:
                        break
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
        return len(self._children)

    @overload
    def __getitem__(self, item: int) -> AsyncIterator[T]: ...

    @overload
    def __getitem__(self, item: slice) -> Tuple[AsyncIterator[T], ...]: ...

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union[AsyncIterator[T], Tuple[AsyncIterator[T], ...]]:
        return self._children[item]

    def __iter__(self) -> Iterator[AsyncIterator[T]]:
        yield from self._children

    async def __aenter__(self) -> "Tee[T]":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for child in self._children:
            await child.aclose()


tee = Tee
