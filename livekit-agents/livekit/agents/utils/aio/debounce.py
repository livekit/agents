import asyncio
from collections.abc import Coroutine
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class Debounced(Generic[T]):
    def __init__(self, func: Callable[[], Coroutine[Any, Any, T]], delay: float) -> None:
        self._func = func
        self._delay = delay
        self._task: Optional[asyncio.Task[T]] = None

    def schedule(self) -> asyncio.Task[T]:
        self.cancel()

        async def _func_with_timer() -> T:
            await asyncio.sleep(self._delay)
            return await self._func()

        self._task = asyncio.create_task(_func_with_timer())
        return self._task

    def run(self) -> asyncio.Task[T]:
        self.cancel()

        self._task = asyncio.create_task(self._func())
        return self._task

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self._task = None

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done() and not self._task.cancelled()

    def __call__(self) -> asyncio.Task[T]:
        return self.run()


def debounced(delay: float) -> Callable[[Callable[[], Coroutine[Any, Any, T]]], Debounced[T]]:
    def decorator(func: Callable[[], Coroutine[Any, Any, T]]) -> Debounced[T]:
        return Debounced(func, delay)

    return decorator
