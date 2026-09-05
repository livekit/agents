from __future__ import annotations

import time
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        """Accumulate values and hand the total to `callback` every `duration` seconds.

        Args:
            callback: Called with the accumulated total when `duration` elapses.
            duration: Seconds between callback invocations.
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value to the accumulator."""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Report the current total, if any, and start a new period."""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()
