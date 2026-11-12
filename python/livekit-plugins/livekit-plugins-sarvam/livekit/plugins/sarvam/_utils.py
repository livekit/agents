from __future__ import annotations

import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    """Accumulates pushed values and reports the total on a fixed interval."""

    def __init__(
        self,
        callback: Callable[[T], None],
        *,
        duration: float,
    ) -> None:
        """Create a new periodic collector.

        Args:
            callback: Function to call with the accumulated value once
                the duration expires.
            duration: Time in seconds between callback invocations.
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value, flushing once the duration has elapsed."""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore[operator]

        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Invoke the callback with the total, if anything is pending."""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


def looks_like_error_text(value: object) -> bool:
    """Heuristic to detect server-side error hints in text payloads/reasons."""
    if not isinstance(value, str):
        return False

    lowered = value.lower()
    error_hints = (
        "error",
        "invalid",
        "failed",
        "forbidden",
        "unauthorized",
        "not found",
        "rate limit",
        "timeout",
    )
    return any(hint in lowered for hint in error_hints)
