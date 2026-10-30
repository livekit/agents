"""Qualification observations; no framework policy or production monkey patches."""

from __future__ import annotations

import asyncio
import math
import time
from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal


class CallbackSlices:
    """Attribute event-loop callback wall and thread CPU without retaining callback objects."""

    def __init__(self) -> None:
        self.values = array("Q")
        self.names: list[str] = []
        self._name_ids: dict[str, int] = {}

    def __enter__(self) -> CallbackSlices:
        self._original = asyncio.Handle._run

        def run(handle: asyncio.Handle) -> None:
            callback = handle._callback
            owner = getattr(callback, "__self__", None)
            if isinstance(owner, asyncio.Task):
                callback = owner.get_coro()
            name = getattr(callback, "__qualname__", type(callback).__name__)
            if name not in self._name_ids:
                self._name_ids[name] = len(self.names)
                self.names.append(name)
            name_id = self._name_ids[name]
            started = time.monotonic_ns()
            cpu = time.thread_time_ns()
            try:
                self._original(handle)
            finally:
                self.values.extend(
                    (started, time.monotonic_ns() - started, time.thread_time_ns() - cpu, name_id)
                )

        asyncio.Handle._run = run
        return self

    def __exit__(self, *exc: object) -> None:
        asyncio.Handle._run = self._original

    @property
    def cpu(self) -> list[int]:
        return list(self.values[2::4])

    @property
    def wall(self) -> list[int]:
        return list(self.values[1::4])

    def largest(self, by: Literal["cpu", "wall"], count: int = 10) -> list[dict[str, object]]:
        field = 2 if by == "cpu" else 1
        offsets = sorted(
            range(0, len(self.values), 4), key=lambda at: self.values[at + field], reverse=True
        )
        return [
            {
                "callback": self.names[self.values[at + 3]],
                "started_ns": self.values[at],
                "wall_ms": self.values[at + 1] / 1e6,
                "cpu_ms": self.values[at + 2] / 1e6,
            }
            for at in offsets[:count]
        ]


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
