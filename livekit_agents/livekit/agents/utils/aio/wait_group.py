import asyncio


class WaitGroup:
    """
    asyncio wait group implementation (similar to sync.WaitGroup in go)
    """

    def __init__(self) -> None:
        self._counter = 0
        self._zero_event = asyncio.Event()
        self._zero_event.set()

    def add(self, delta: int = 1) -> None:
        new_value = self._counter + delta
        if new_value < 0:
            raise ValueError("WaitGroup counter cannot go negative.")

        self._counter = new_value

        if self._counter == 0:
            self._zero_event.set()
        else:
            self._zero_event.clear()

    def done(self) -> None:
        self.add(-1)

    async def wait(self) -> None:
        await self._zero_event.wait()
