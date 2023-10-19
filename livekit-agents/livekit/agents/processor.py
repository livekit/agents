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