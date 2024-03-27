from __future__ import annotations

from typing import Any, Dict, Union, cast

from deprecation import deprecated
from httpx import Headers, QueryParams, Timeout

from .. import __version__
from ..base_client import BasePostgrestClient
from ..constants import (
    DEFAULT_POSTGREST_CLIENT_HEADERS,
    DEFAULT_POSTGREST_CLIENT_TIMEOUT,
)
from ..utils import SyncClient
from .request_builder import SyncRequestBuilder, SyncRPCFilterRequestBuilder

_TableT = Dict[str, Any]


class SyncPostgrestClient(BasePostgrestClient):
    """PostgREST client."""

    def __init__(
        self,
        base_url: str,
        *,
        schema: str = "public",
        headers: Dict[str, str] = DEFAULT_POSTGREST_CLIENT_HEADERS,
        timeout: Union[int, float, Timeout] = DEFAULT_POSTGREST_CLIENT_TIMEOUT,
    ) -> None:
        BasePostgrestClient.__init__(
            self,
            base_url,
            schema=schema,
            headers=headers,
            timeout=timeout,
        )
        self.session = cast(SyncClient, self.session)

    def create_session(
        self,
        base_url: str,
        headers: Dict[str, str],
        timeout: Union[int, float, Timeout],
    ) -> SyncClient:
        return SyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
        )

    def __enter__(self) -> SyncPostgrestClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.aclose()

    def aclose(self) -> None:
        """Close the underlying HTTP connections."""
        self.session.aclose()

    def from_(self, table: str) -> SyncRequestBuilder[_TableT]:
        """Perform a table operation.

        Args:
            table: The name of the table
        Returns:
            :class:`AsyncRequestBuilder`
        """
        return SyncRequestBuilder[_TableT](self.session, f"/{table}")

    def table(self, table: str) -> SyncRequestBuilder[_TableT]:
        """Alias to :meth:`from_`."""
        return self.from_(table)

    @deprecated("0.2.0", "1.0.0", __version__, "Use self.from_() instead")
    def from_table(self, table: str) -> SyncRequestBuilder:
        """Alias to :meth:`from_`."""
        return self.from_(table)

    def rpc(self, func: str, params: dict) -> SyncRPCFilterRequestBuilder[Any]:
        """Perform a stored procedure call.

        Args:
            func: The name of the remote procedure to run.
            params: The parameters to be passed to the remote procedure.
        Returns:
            :class:`AsyncRPCFilterRequestBuilder`
        Example:
            .. code-block:: python

                await client.rpc("foobar", {"arg": "value"}).execute()

        .. versionchanged:: 0.10.9
            This method now returns a :class:`AsyncRPCFilterRequestBuilder`.
        .. versionchanged:: 0.10.2
            This method now returns a :class:`AsyncFilterRequestBuilder` which allows you to
            filter on the RPC's resultset.
        """
        # the params here are params to be sent to the RPC and not the queryparams!
        return SyncRPCFilterRequestBuilder[Any](
            self.session, f"/rpc/{func}", "POST", Headers(), QueryParams(), json=params
        )
