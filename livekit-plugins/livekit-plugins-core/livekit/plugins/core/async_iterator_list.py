import asyncio
from typing import TypeVar

T = TypeVar('T')


class AsyncIteratorList:
    def __init__(self, items: [T]):
        self._list = items

    def __aiter__(self):
        return self

    async def __anext__(self):
        if len(self._list) == 0:
            raise StopAsyncIteration

        return self._list.pop(0)
