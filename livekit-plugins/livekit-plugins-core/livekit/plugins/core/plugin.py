# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from abc import abstractmethod, ABC
from typing import (Callable,
                    TypeVar,
                    Generic,
                    AsyncIterable,
                    Awaitable)


T = TypeVar('T')
U = TypeVar('U')


class Plugin(ABC, Generic[T, U]):

    @abstractmethod
    async def process(self, iterator: T) -> U:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

class PluginIterator(Generic[T]):
    class EOS:
        pass

    def __init__(self) -> None:
        self._queue = asyncio.Queue[T]()

    @classmethod
    def from_iterator(cls, iterator: AsyncIterable[T]) -> "PluginIterator[T]":
        res = PluginIterator()
        
        async def iterator_fn() -> AsyncIterable[T]:
            async for item in iterator:
                await res.put(item)
            await res.aclose()

        asyncio.create_task(iterator_fn())
        return res

    async def put(self, item: T) -> None:
        await self._queue.put(item)

    async def aclose(self):
        await self._queue.put(PluginIterator.EOS())

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        item = await self._queue.get()
        if isinstance(item, PluginIterator.EOS):
            raise StopAsyncIteration
        return item

    def filter(self, predicate: Callable[[T], bool]) -> "PluginIterator[T]":
        res = PluginIterator()

        async def iterator() -> AsyncIterable[T]:
            async for item in self:
                if predicate(item):
                    await res.put(item)
                await res.aclose()

        asyncio.create_task(iterator())
        return res

    def map(self, mapper: Callable[[T], U]) -> "PluginIterator[U]":
        res = PluginIterator()

        async def iterator() -> AsyncIterable[U]:
            async for item in self._iterator:
                await res.put(mapper(item))
            await res.aclose()

        asyncio.create_task(iterator())
        return res

    def map_async(self, mapper: Callable[[T], Awaitable[U]]) -> "PluginIterator[U]":
        res = PluginIterator()

        async def iterator() -> AsyncIterable[U]:
            async for item in self:
                mapped = await mapper(item)
                await res.put(mapped)
            await res.aclose()

        asyncio.create_task(iterator())
        return res
