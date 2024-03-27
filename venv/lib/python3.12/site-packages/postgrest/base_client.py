from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from httpx import BasicAuth, Timeout

from .utils import AsyncClient, SyncClient


class BasePostgrestClient(ABC):
    """Base PostgREST client."""

    def __init__(
        self,
        base_url: str,
        *,
        schema: str,
        headers: Dict[str, str],
        timeout: Union[int, float, Timeout],
    ) -> None:
        headers = {
            **headers,
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }
        self.session = self.create_session(base_url, headers, timeout)

    @abstractmethod
    def create_session(
        self,
        base_url: str,
        headers: Dict[str, str],
        timeout: Union[int, float, Timeout],
    ) -> Union[SyncClient, AsyncClient]:
        raise NotImplementedError()

    def auth(
        self,
        token: Optional[str],
        *,
        username: Union[str, bytes, None] = None,
        password: Union[str, bytes] = "",
    ):
        """
        Authenticate the client with either bearer token or basic authentication.

        Raises:
            `ValueError`: If neither authentication scheme is provided.

        .. note::
            Bearer token is preferred if both ones are provided.
        """
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        elif username:
            self.session.auth = BasicAuth(username, password)
        else:
            raise ValueError(
                "Neither bearer token or basic authentication scheme is provided"
            )
        return self

    def schema(self, schema: str):
        """Switch to another schema."""
        self.session.headers.update(
            {
                "Accept-Profile": schema,
                "Content-Profile": schema,
            }
        )
        return self
