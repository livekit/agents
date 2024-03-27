from httpx import AsyncClient as AsyncClient  # noqa: F401
from httpx import Client as BaseClient

__version__ = "0.5.5"


class SyncClient(BaseClient):
    def aclose(self) -> None:
        self.close()


class StorageException(Exception):
    """Error raised when an operation on the storage API fails."""
