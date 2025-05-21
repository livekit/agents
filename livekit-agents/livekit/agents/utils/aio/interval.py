from __future__ import annotations

import asyncio
from typing import Any


def _finish_fut(fut: asyncio.Future[Any]) -> None:
    if fut.cancelled():
        return
    fut.set_result(None)


# MissedBehaviour is "Delay"
class Interval:
    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._last_sleep = 0.0
        self._i = 0
        self._handler: asyncio.TimerHandle | None = None

    def reset(self) -> None:
        if self._fut and self._handler and not self._handler.cancelled():
            self._handler.cancel()
            loop = asyncio.get_event_loop()
            self._handler = loop.call_later(self._interval, _finish_fut, self._fut)
        else:
            self._last_sleep = 0

    async def tick(self) -> int:
        loop = asyncio.get_event_loop()

        if self._last_sleep:
            self._fut = loop.create_future()
            delay = self._last_sleep - loop.time() + self._interval
            self._handler = loop.call_later(delay, _finish_fut, self._fut)
            try:
                await self._fut
            finally:
                self._handler.cancel()
            self._i += 1

        self._last_sleep = loop.time()
        return self._i

    def __aiter__(self) -> Interval:
        return self

    async def __anext__(self) -> int:
        return await self.tick()


def interval(interval: float) -> Interval:
    return Interval(interval)
