import asyncio


class WaitGroup:
    def __init__(self) -> None:
        self._count = 0
        self._waiters = []

    def add(self, delta: int) -> None:
        self._count += delta

    def done(self) -> None:
        self._count -= 1
        if self._count == 0:
            for w in self._waiters:
                w.set_result(None)
            self._waiters.clear()

    async def wait(self) -> None:
        if self._count == 0:
            return

        f = asyncio.Future()
        self._waiters.append(f)
        await f
