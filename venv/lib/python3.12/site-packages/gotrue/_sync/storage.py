from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class SyncSupportedStorage(ABC):
    @abstractmethod
    def get_item(self, key: str) -> Optional[str]:
        ...  # pragma: no cover

    @abstractmethod
    def set_item(self, key: str, value: str) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def remove_item(self, key: str) -> None:
        ...  # pragma: no cover


class SyncMemoryStorage(SyncSupportedStorage):
    def __init__(self):
        self.storage: Dict[str, str] = {}

    def get_item(self, key: str) -> Optional[str]:
        if key in self.storage:
            return self.storage[key]

    def set_item(self, key: str, value: str) -> None:
        self.storage[key] = value

    def remove_item(self, key: str) -> None:
        if key in self.storage:
            del self.storage[key]
