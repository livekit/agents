"""Qualification observations; no framework policy or production monkey patches."""

from __future__ import annotations

import asyncio
import math
import time
from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Distribution:
    count: int
    p99_ms: float
    p999_ms: float
    max_ms: float
    buckets_ms: dict[str, int]


def distribution(values: list[int]) -> Distribution:
    if not values:
        raise ValueError("empty measurement")
    ordered = sorted(values)
    buckets: Counter[str] = Counter()
    for value in values:
        limit = next((x for x in (0.01, 0.1, 0.5, 1, 2, 5, 10, 20) if value <= x * 1e6), None)
        buckets[str(limit) if limit is not None else ">20"] += 1
    return Distribution(
        len(values),
        ordered[math.ceil(len(values) * 0.99) - 1] / 1e6,
        ordered[math.ceil(len(values) * 0.999) - 1] / 1e6,
        ordered[-1] / 1e6,
        dict(buckets),
    )


class LagObservations:
    """Deadline observations stay numeric so the probe cannot induce cyclic GC pauses."""

    def __init__(self) -> None:
        self._values = array("Q")

    def append(self, observed_ns: int, lag_ns: int) -> None:
        self._values.append(observed_ns)
        self._values.append(lag_ns)

    def clear(self) -> None:
        del self._values[:]

    def __len__(self) -> int:
        return len(self._values) // 2

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for at in range(0, len(self._values), 2):
            yield self._values[at], self._values[at + 1]


class LoopLag:
    """Absolute deadlines preserve long stalls; overdue ticks aren't silently discarded."""

    def __init__(self, period_ms: float = 1) -> None:
        if not math.isfinite(period_ms) or period_ms <= 0:
            raise ValueError("period must be finite and positive")
        self.period = period_ms / 1000
        self.samples: list[int] = []
        self.observations = LagObservations()
        self.ready = asyncio.Event()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.period
        self.ready.set()
        while True:
            await asyncio.sleep(max(0, deadline - loop.time()))
            now = loop.time()
            lag = max(0, int((now - deadline) * 1e9))
            self.samples.append(lag)
            self.observations.append(time.monotonic_ns(), lag)
            # One observation per wake. Missed periods are represented by the lag itself.
            deadline += (max(0, int((now - deadline) / self.period)) + 1) * self.period
