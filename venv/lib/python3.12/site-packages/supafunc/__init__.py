from __future__ import annotations

from typing import Literal, Union, overload

from ._async.functions_client import AsyncFunctionsClient
from ._sync.functions_client import SyncFunctionsClient

__all__ = ["create_client"]


@overload
def create_client(
    url: str, headers: dict[str, str], *, is_async: Literal[True]
) -> AsyncFunctionsClient:
    ...


@overload
def create_client(
    url: str, headers: dict[str, str], *, is_async: Literal[False]
) -> SyncFunctionsClient:
    ...


def create_client(
    url: str, headers: dict[str, str], *, is_async: bool
) -> Union[AsyncFunctionsClient, SyncFunctionsClient]:
    if is_async:
        return AsyncFunctionsClient(url, headers)
    else:
        return SyncFunctionsClient(url, headers)
