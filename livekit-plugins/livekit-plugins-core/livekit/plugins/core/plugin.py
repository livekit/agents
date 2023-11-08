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
from enum import Enum
from dataclasses import dataclass
from typing import Callable, TypeVar, Generic, AsyncIterable, Optional
from abc import abstractmethod

T = TypeVar('T')
U = TypeVar('U')
E = TypeVar('E')

PluginEventType = Enum('PluginEventType', ['ERROR', 'SUCCESS'])


class Plugin(Generic[T, U]):

    @dataclass
    class Event(Generic[E]):
        type: PluginEventType
        data: Optional[E] = None
        error: Optional[Exception] = None

    def __init__(self, process: Callable[[AsyncIterable[T]], AsyncIterable[Event[U]]]) -> None:
        self._process = process

    def start(self, data: AsyncIterable[T]) -> "PluginResultIterator[U]":
        return PluginResultIterator(iterator=self._process(data))


class PluginResultIterator(Generic[T]):

    def __init__(self, iterator: AsyncIterable[Plugin.Event[T]]) -> None:
        self._iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self) -> Plugin.Event[T]:
        async for item in self._iterator:
            return item

    def filter(self, predicate: Callable[[T], bool]) -> "PluginResultIterator[T]":
        async def iteratator() -> AsyncIterable[Plugin.Event[T]]:
            async for item in self._iterator:
                if item.type == PluginEventType.ERROR:
                    yield item
                    continue

                if predicate(item.data):
                    yield item

        return PluginResultIterator(iterator=iteratator())

    def map(self, mapper: Callable[[T], U]) -> "PluginResultIterator[U]":
        async def iteratator() -> AsyncIterable[Plugin.Event[U]]:
            async for item in self._iterator:
                if item.type == PluginEventType.ERROR:
                    yield item
                    continue

                yield Plugin.Event(type=item.type, data=mapper(item.data))

        return PluginResultIterator(iterator=iteratator())

    def unwrap(self):
        async def iteratator() -> AsyncIterable[T]:
            async for item in self._iterator:
                if item.type == PluginEventType.ERROR:
                    continue
                yield item.data

        return iteratator()
