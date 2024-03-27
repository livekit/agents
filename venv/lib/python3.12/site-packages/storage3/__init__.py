from __future__ import annotations

from typing import Union, overload

from typing_extensions import Literal

from storage3._async import AsyncStorageClient
from storage3._sync import SyncStorageClient
from storage3.constants import DEFAULT_TIMEOUT
from storage3.utils import __version__

__all__ = ["create_client", "__version__"]


@overload
def create_client(
    url: str, headers: dict[str, str], *, is_async: Literal[True]
) -> AsyncStorageClient:
    ...


@overload
def create_client(
    url: str, headers: dict[str, str], *, is_async: Literal[False]
) -> SyncStorageClient:
    ...


def create_client(
    url: str, headers: dict[str, str], *, is_async: bool, timeout: int = DEFAULT_TIMEOUT
) -> Union[AsyncStorageClient, SyncStorageClient]:
    if is_async:
        return AsyncStorageClient(url, headers, timeout)
    else:
        return SyncStorageClient(url, headers, timeout)
