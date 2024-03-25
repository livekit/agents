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
import logging
from collections import deque
import ctypes
import random
from typing import Callable, Generic, List, TypeVar

logger = logging.getLogger("livekit")


class classproperty(object):
    def __init__(self, f):
        self.f = classmethod(f)

    def __get__(self, *a):
        return self.f.__get__(*a)()


def task_done_logger(task: asyncio.Task) -> None:
    if task.cancelled():
        logger.info("task cancelled: %s", task)
        return

    if task.exception():
        logger.error("task exception: %s", task, exc_info=task.exception())
        return


def get_address(data: memoryview) -> int:
    """Get the address of a buffer using ctypes"""
    nbytes = data.nbytes
    buffer = (ctypes.c_int8 * nbytes).from_buffer(data)
    return ctypes.addressof(buffer)


T = TypeVar("T")


class RingQueue(Generic[T]):
    def __init__(self, capacity: int = 0) -> None:
        self._capacity = capacity
        self._queue: deque[T] = deque()
        self._event = asyncio.Event()

    def put(self, item: T) -> None:
        if self._capacity > 0 and len(self._queue) == self._capacity:
            self._queue.pop()
        self._queue.append(item)
        self._event.set()

    async def get(self) -> T:
        while len(self._queue) == 0:
            await self._event.wait()
        self._event.clear()
        return self._queue.popleft()


class Queue(asyncio.Queue[T]):
    """asyncio.Queue with utility functions."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize)

    async def wait_for(self, fnc: Callable[[T], bool]) -> T:
        """Wait for an event that matches the given function.
        The previous events are discarded.
        """

        while True:
            event = await self.get()
            if fnc(event):
                # task_done must be manually called for the returned item
                return event

            self.task_done()


class BroadcastQueue(Generic[T]):
    """Queue with multiple subscribers."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subscribers: List[Queue[T]] = []

    def len_subscribers(self) -> int:
        return len(self._subscribers)

    def put_nowait(self, item: T) -> None:
        for queue in self._subscribers:
            queue.put_nowait(item)

    def subscribe(self) -> Queue[T]:
        queue = Queue[T]()
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: Queue[T]) -> None:
        self._subscribers.remove(queue)

    async def join(self) -> None:
        async with self._lock:
            subs = self._subscribers.copy()
            for queue in subs:
                await queue.join()


_base62_characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def generate_random_base62(length=12):
    """
    Generate a random base62 encoded string of a specified length.

    :param length: The desired length of the base62 encoded string.
    :return: A base62 encoded string.
    """
    global _base62_characters
    return "".join(random.choice(_base62_characters) for _ in range(length))
