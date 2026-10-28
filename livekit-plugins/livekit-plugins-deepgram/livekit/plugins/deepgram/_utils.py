import time
from collections.abc import Callable, Mapping, Sequence
from typing import Generic, TypeVar
from urllib.parse import urlencode

T = TypeVar("T")


class PeriodicCollector(Generic[T]):
    def __init__(self, callback: Callable[[T], None], *, duration: float) -> None:
        """
        Create a new periodic collector that accumulates values and calls the callback
        after the specified duration if there are values to report.

        Args:
            duration: Time in seconds between callback invocations
            callback: Function to call with accumulated value when duration expires
        """
        self._duration = duration
        self._callback = callback
        self._last_flush_time = time.monotonic()
        self._total: T | None = None

    def push(self, value: T) -> None:
        """Add a value to the accumulator"""
        if self._total is None:
            self._total = value
        else:
            self._total += value  # type: ignore
        if time.monotonic() - self._last_flush_time >= self._duration:
            self.flush()

    def flush(self) -> None:
        """Force callback to be called with current total if non-zero"""
        if self._total is not None:
            self._callback(self._total)
            self._total = None
        self._last_flush_time = time.monotonic()


def _to_deepgram_url(
    opts: Mapping[str, object],
    base_url: str,
    *,
    websocket: bool,
    keywords: Sequence[tuple[str, float]] | None = None,
    replace: Mapping[str, str] | None = None,
) -> str:
    query = {
        key: str(value).lower() if isinstance(value, bool) else value for key, value in opts.items()
    }
    if keywords:
        query["keywords"] = [f"{keyword}:{intensifier}" for keyword, intensifier in keywords]
    if replace:
        query["replace"] = [f"{term}:{replacement}" for term, replacement in replace.items()]

    if websocket and base_url.startswith("http"):
        base_url = base_url.replace("http", "ws", 1)

    elif not websocket and base_url.startswith("ws"):
        base_url = base_url.replace("ws", "http", 1)
    return f"{base_url}?{urlencode(query, doseq=True)}"
