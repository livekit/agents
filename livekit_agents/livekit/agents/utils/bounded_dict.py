from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


class BoundedDict(OrderedDict[K, V]):
    def __init__(self, maxsize: int | None = None):
        super().__init__()
        self.maxsize = maxsize
        if self.maxsize is not None and self.maxsize <= 0:
            raise ValueError("maxsize must be greater than 0")

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)

        while self.maxsize is not None and len(self) > self.maxsize:
            self.popitem(last=False)
