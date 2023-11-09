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
from typing import Callable, TypeVar, Generic, AsyncIterable, Optional, Dict, Set, Literal
from abc import abstractmethod

T = TypeVar('T')
U = TypeVar('U')


EventTypes = Literal["error"]

class Plugin(Generic[T, U]):

    def __init__(self, process: Callable[[AsyncIterable[T]], AsyncIterable[U]]) -> None:
        self._process = process
        self._events: Dict[T, Set[Callable]] = dict()
        self._current_loop = asyncio.get_running_loop()
        self._current_thread = threading.current_thread().ident

    def start(self, data: AsyncIterable[T]) -> "PluginResultIterator[U]":
        return PluginResultIterator(iterator=self._process(data))

    def emit(self, event: EventTypes, *args, **kwargs) -> None:
        def emit_event():
            if event in self._events:
                for callback in self._events[event]:
                    callback(*args, **kwargs)

        if threading.current_thread().ident == self._current_thread:
            emit_event()
        else:
            self._current_loop.call_soon_threadsafe(emit_event)
            
    def on(self, event: EventTypes, callback: Optional[Callable] = None) -> Callable:
        if callback is not None:
            if event not in self._events:
                self._events[event] = set()
            self._events[event].add(callback)
            return callback
        else:
            def decorator(callback: Callable) -> Callable:
                self.on(event, callback)
                return callback
            return decorator

    def off(self, event: T, callback: Callable) -> None:
        if event in self._events:
            self._events[event].remove(callback)

    
class PluginResultIterator(Generic[T]):

    def __init__(self, iterator: AsyncIterable[T]) -> None:
        self._iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        async for item in self._iterator:
            return item

    def filter(self, predicate: Callable[[T], bool]) -> "PluginResultIterator[T]":
        async def iterator() -> AsyncIterable[T]:
            async for item in self._iterator:
                if predicate(item):
                    yield item

        return PluginResultIterator(iterator=iterator())

    def map(self, mapper: Callable[[T], U]) -> "PluginResultIterator[U]":
        async def iterator() -> AsyncIterable[U]:
            async for item in self._iterator:
                yield mapper(item)

        return PluginResultIterator(iterator=iterator())