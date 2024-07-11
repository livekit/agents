from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")


class TaskSet:
    """
    Small utility to create task in a fire-and-forget fashion.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_event_loop()
        self._set = set[asyncio.Task[Any]]()
        self._closed = False

    def create_task(self, coro: Coroutine[Any, Any, _T]) -> asyncio.Task[_T]:
        if self._closed:
            raise RuntimeError("TaskSet is closed")

        task = self._loop.create_task(coro)
        self._set.add(task)
        task.add_done_callback(self._set.remove)
        return task

    async def aclose(self) -> None:
        self._closed = True
        await asyncio.gather(*self._set, return_exceptions=True)
        self._set.clear()
