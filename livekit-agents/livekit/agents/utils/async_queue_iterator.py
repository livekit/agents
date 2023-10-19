import asyncio

class AsyncQueueIterator:
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return await self.queue.get()
        except asyncio.CancelledError:
            raise StopAsyncIteration