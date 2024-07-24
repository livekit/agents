from __future__ import annotations

import asyncio
from typing import Any


def _finish_fut(fut: asyncio.Future[Any]):
    if fut.cancelled():
        return
    fut.set_result(None)


class SleepFinished(Exception):
    pass


class Sleep:
    """Same as asyncio.sleep except it is resettable"""

    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._handler: asyncio.TimerHandle | None = None

    def reset(self, new_delay: float | None = None) -> None:
        if new_delay is None:
            new_delay = self._delay

        self._delay = new_delay

        if self._handler is None:
            return

        if self._handler.cancelled() or self._fut.done():
            raise SleepFinished

        self._handler.cancel()
        loop = asyncio.get_event_loop()
        self._handler = loop.call_later(new_delay, _finish_fut, self._fut)

    def cancel(self) -> None:
        if self._handler is None:
            return

        self._handler.cancel()
        self._fut.cancel()

    async def _sleep(self) -> None:
        if self._delay <= 0:
            self._fut = asyncio.Future[None]()
            self._fut.set_result(None)
            return

        loop = asyncio.get_event_loop()
        self._fut = loop.create_future()
        self._handler = loop.call_later(self._delay, _finish_fut, self._fut)

        try:
            await self._fut
        finally:
            self._handler.cancel()

    def __await__(self):
        return self._sleep().__await__()


def sleep(delay: float) -> Sleep:
    return Sleep(delay)
