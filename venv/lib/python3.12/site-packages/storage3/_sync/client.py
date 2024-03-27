from __future__ import annotations

from storage3.constants import DEFAULT_TIMEOUT

from ..utils import SyncClient, __version__
from .bucket import SyncStorageBucketAPI
from .file_api import SyncBucketProxy

__all__ = [
    "SyncStorageClient",
]


class SyncStorageClient(SyncStorageBucketAPI):
    """Manage storage buckets and files."""

    def __init__(
        self, url: str, headers: dict[str, str], timeout: int = DEFAULT_TIMEOUT
    ) -> None:
        headers = {
            "User-Agent": f"supabase-py/storage3 v{__version__}",
            **headers,
        }
        self.session = self._create_session(url, headers, timeout)
        super().__init__(self.session)

    def _create_session(
        self, base_url: str, headers: dict[str, str], timeout: int
    ) -> SyncClient:
        return SyncClient(base_url=base_url, headers=headers, timeout=timeout)

    def __enter__(self) -> SyncStorageClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.aclose()

    def aclose(self) -> None:
        self.session.aclose()

    def from_(self, id: str) -> SyncBucketProxy:
        """Run a storage file operation.

        Parameters
        ----------
        id
            The unique identifier of the bucket
        """
        return SyncBucketProxy(id, self._client)
