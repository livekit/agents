import asyncio
from collections.abc import Awaitable
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Debounce(Generic[T]):
    def __init__(self, func: Callable[[], Awaitable[T]], delay: float) -> None:
        self._func = func
        self._delay = delay
        self._task: asyncio.Task[T] | None = None

    def schedule(self) -> asyncio.Task[T]:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None

        async def _func_with_timer() -> T:
            await asyncio.sleep(self._delay)
            return await self._func()

        self._task = asyncio.create_task(_func_with_timer())
        return self._task

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None

    def is_running(self) -> bool:
        return (
            self._task is not None
            and not self._task.done()
            and not self._task.cancelled()
            and not self._task.cancelling()
        )

    def __call__(self) -> asyncio.Task[T]:
        return self.schedule()


def debounce(delay: float) -> Callable[[Callable[[], Awaitable[T]]], Debounce[T]]:
    def decorator(func: Callable[[], Awaitable[T]]) -> Debounce[T]:
        return Debounce(func, delay)

    return decorator
