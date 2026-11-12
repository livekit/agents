import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        """
        Accumulate values and invoke ``callback`` with the total after ``duration``
        seconds have elapsed since the last flush, provided at least one value was
        pushed.

        Args:
            duration: Time in seconds between callback invocations.
            callback: Function to call with the accumulated total.
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()
