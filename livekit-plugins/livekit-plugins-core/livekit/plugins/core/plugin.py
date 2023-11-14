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
import threading
from enum import Enum
from dataclasses import dataclass
from typing import (Callable,
                    TypeVar,
                    Generic,
                    AsyncIterable,
                    Optional,
                    Dict,
                    Set,
                    Literal,
                    Awaitable,
                    Tuple)


T = TypeVar('T')
U = TypeVar('U')


EventTypes = Literal["error"]


class Plugin(Generic[T, U]):

    def __init__(self,
                 process: Callable[[AsyncIterable[T]],
                                   AsyncIterable[U]],
                 close: Callable) -> None:
        self.__process = process
        self.__close = close
        self.__events: Dict[T, Set[Callable]] = dict()
        self.__current_loop = asyncio.get_running_loop()
        self.__current_thread = threading.current_thread().ident

    def set_input(self, data: "PluginIterator[T]") -> "PluginIterator[U]":

        current_metadata: "[PluginIterator.ResultMetadata]" = [None]

        async def item_iterator():
            async for (item, metadata) in data:
                current_metadata[0] = metadata
                yield item

        async def iterator():
            async for item in self.__process(item_iterator()):
                yield item, current_metadata[0]

        return PluginIterator(iterator=iterator())

    def emit(self, event: EventTypes, *args, **kwargs) -> None:
        def emit_event():
            if event in self.__events:
                for callback in self.__events[event]:
                    callback(*args, **kwargs)

        if threading.current_thread().ident == self.__current_thread:
            emit_event()
        else:
            self.__current_loop.call_soon_threadsafe(emit_event)

    def on(
            self,
            event: EventTypes,
            callback: Optional[Callable] = None) -> Callable:
        if callback is not None:
            if event not in self.__events:
                self.__events[event] = set()
            self.__events[event].add(callback)
            return callback
        else:
            def decorator(callback: Callable) -> Callable:
                self.on(event, callback)
                return callback
            return decorator

    def off(self, event: T, callback: Callable) -> None:
        if event in self.__events:
            self.__events[event].remove(callback)

    def close(self) -> None:
        if threading.current_thread().ident != self.__current_thread:
            asyncio.run_coroutine_threadsafe(
                self.__close(), self.__current_loop)
        else:
            self.__current_loop.create_task(self.__close())


class PluginIterator(Generic[T]):

    @dataclass
    class ResultMetadata:
        sequence_number: int

    def __init__(
            self, iterator: AsyncIterable[Tuple[T, ResultMetadata]]) -> None:
        self._iterator = iterator

    def __aiter__(self):
        return self

    @classmethod
    def create(cls, iterator: AsyncIterable[T]) -> "PluginIterator[T]":
        async def _iterator() -> AsyncIterable[Tuple[T, PluginIterator.ResultMetadata]]:
            sn = -1
            async for item in iterator:
                sn += 1
                yield item, PluginIterator.ResultMetadata(sequence_number=sn)

        return _iterator()

    async def __anext__(self) -> Tuple[T, ResultMetadata]:
        async for item in self._iterator:
            return item

    def filter(self, predicate: Callable[[
               T, ResultMetadata], bool]) -> "PluginIterator[T]":
        async def iterator() -> AsyncIterable[T]:
            async for (item, metadata) in self._iterator:
                if predicate(item, metadata):
                    yield item, metadata

        return PluginIterator(iterator=iterator())

    def map(self, mapper: Callable[[
            T, ResultMetadata], U]) -> "PluginIterator[U]":
        async def iterator() -> AsyncIterable[U]:
            async for (item, metadata) in self._iterator:
                yield mapper(item, metadata), metadata

        return PluginIterator(iterator=iterator())

    def map_async(self,
                  mapper: Callable[[T,
                                    ResultMetadata],
                                   Awaitable[U]]) -> "PluginIterator[U]":
        async def iterator() -> AsyncIterable[U]:
            async for (item, metadata) in self._iterator:
                yield await mapper(item, metadata), metadata

        return PluginIterator(iterator=iterator())

    def do(self, callback: Callable[[
           T, ResultMetadata], None]) -> "PluginIterator[T]":
        async def iterator() -> AsyncIterable[T]:
            async for (item, metadata) in self._iterator:
                callback(item, metadata)
                yield item, metadata

        return PluginIterator(iterator=iterator())

    def do_async(self,
                 callback: Callable[[T,
                                     ResultMetadata],
                                    Awaitable[None]]) -> "PluginIterator[T]":
        async def iterator() -> AsyncIterable[T]:
            async for (item, metadata) in self._iterator:
                await callback(item, metadata)
                yield item, metadata

        return PluginIterator(iterator=iterator())

    def skip_while(self,
                   predicate: Callable[[T,
                                        ResultMetadata],
                                       bool]) -> "PluginIterator[T]":
        async def iterator() -> AsyncIterable[T]:
            async for (item, metadata) in self._iterator:
                if not predicate(item, metadata):
                    yield item, metadata

        return PluginIterator(iterator=iterator())

    def pipe(self, plugin: Plugin[T, U]) -> "PluginIterator[U]":
        return plugin.set_input(self)

    async def run(self):
        async for _ in self:
            pass
