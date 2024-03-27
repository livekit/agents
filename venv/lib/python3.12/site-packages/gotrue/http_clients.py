from __future__ import annotations

from httpx import AsyncClient  # noqa: F401
from httpx import Client as BaseClient


class SyncClient(BaseClient):
    def aclose(self) -> None:
        self.close()
