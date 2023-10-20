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
from typing import AsyncIterator, Awaitable, Callable, TypeVar, Generic
from .utils.async_queue_iterator import AsyncQueueIterator

T = TypeVar('T')
U = TypeVar('U')

class Processor(Generic[T, U]):
    def __init__(self, process: Callable[[T], Awaitable[U]]) -> None:
        self._process = process
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        asyncio.create_task(self._process_loop())

    def stream(self) -> AsyncIterator[U]:
        return AsyncQueueIterator(self.output_queue)

    def push(self, data: T) -> None:
        self.input_queue.put_nowait(data)

    async def _process_loop(self):
        while True:
            data = await self.input_queue.get()
            result = await self._process(data)
            if result is not None:
                await self.output_queue.put(result)