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
from typing import AsyncIterable, TypeVar, Generic


T = TypeVar('T')


class AsyncQueueIterator(Generic[T]):

    class EOS:
        pass

    def __init__(self, queue: asyncio.Queue[T]):
        self.queue = queue

    def __aiter__(self):
        return self

    async def put(self, item):
        await self.queue.put(item)

    async def __anext__(self) -> T:
        item = await self.queue.get()
        if type(item) is AsyncQueueIterator.EOS:
            raise StopAsyncIteration
        return item

    async def aclose(self):
        await self.queue.put(AsyncQueueIterator.EOS())
