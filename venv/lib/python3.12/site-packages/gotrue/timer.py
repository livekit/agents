import asyncio
from threading import Timer as _Timer
from typing import Any, Callable, Coroutine, Union, cast


class Timer:
    def __init__(
        self,
        seconds: float,
        function: Callable[[], Union[Coroutine[Any, Any, None], None]],
    ) -> None:
        self._milliseconds = seconds
        self._function = function
        self._task: Union[asyncio.Task, None] = None
        self._timer: Union[_Timer, None] = None

    def start(self) -> None:
        if asyncio.iscoroutinefunction(self._function):

            async def schedule():
                await asyncio.sleep(self._milliseconds / 1000)
                await cast(Coroutine[Any, Any, None], self._function())

            def cleanup(_):
                self._task = None

            self._task = asyncio.create_task(schedule())
            self._task.add_done_callback(cleanup)
        else:
            self._timer = _Timer(self._milliseconds / 1000, self._function)
            self._timer.start()

    def cancel(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def is_alive(self) -> bool:
        return self._task is not None or (
            self._timer is not None and self._timer.is_alive()
        )
