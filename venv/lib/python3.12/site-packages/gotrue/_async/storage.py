from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class AsyncSupportedStorage(ABC):
    @abstractmethod
    async def get_item(self, key: str) -> Optional[str]:
        ...  # pragma: no cover

    @abstractmethod
    async def set_item(self, key: str, value: str) -> None:
        ...  # pragma: no cover

    @abstractmethod
    async def remove_item(self, key: str) -> None:
        ...  # pragma: no cover


class AsyncMemoryStorage(AsyncSupportedStorage):
    def __init__(self):
        self.storage: Dict[str, str] = {}

    async def get_item(self, key: str) -> Optional[str]:
        if key in self.storage:
            return self.storage[key]

    async def set_item(self, key: str, value: str) -> None:
        self.storage[key] = value

    async def remove_item(self, key: str) -> None:
        if key in self.storage:
            del self.storage[key]
