from __future__ import annotations


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self._hist: list[float] = [0] * window_size
        self._sum: float = 0
        self._count: int = 0

    def add_sample(self, sample: float) -> None:
        self._count += 1
        index = self._count % len(self._hist)
        if self._count > len(self._hist):
            self._sum -= self._hist[index]
        self._sum += sample
        self._hist[index] = sample

    def get_avg(self) -> float:
        if self._count == 0:
            return 0
        return self._sum / self.size()

    def reset(self):
        self._count = 0
        self._sum = 0

    def size(self) -> int:
        return min(self._count, len(self._hist))
