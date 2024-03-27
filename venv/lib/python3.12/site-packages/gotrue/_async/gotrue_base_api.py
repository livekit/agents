from __future__ import annotations

from typing import Any, Callable, Dict, TypeVar, Union, overload

from httpx import Response
from pydantic import BaseModel
from typing_extensions import Literal, Self

from ..helpers import handle_exception, model_dump
from ..http_clients import AsyncClient

T = TypeVar("T")


class AsyncGoTrueBaseAPI:
    def __init__(
        self,
        *,
        url: str,
        headers: Dict[str, str],
        http_client: Union[AsyncClient, None],
    ):
        self._url = url
        self._headers = headers
        self._http_client = http_client or AsyncClient()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_t, exc_v, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http_client.aclose()

    @overload
    async def _request(
        self,
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        path: str,
        *,
        jwt: Union[str, None] = None,
        redirect_to: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        query: Union[Dict[str, str], None] = None,
        body: Union[Any, None] = None,
        no_resolve_json: Literal[False] = False,
        xform: Callable[[Any], T],
    ) -> T:
        ...  # pragma: no cover

    @overload
    async def _request(
        self,
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        path: str,
        *,
        jwt: Union[str, None] = None,
        redirect_to: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        query: Union[Dict[str, str], None] = None,
        body: Union[Any, None] = None,
        no_resolve_json: Literal[True],
        xform: Callable[[Response], T],
    ) -> T:
        ...  # pragma: no cover

    @overload
    async def _request(
        self,
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        path: str,
        *,
        jwt: Union[str, None] = None,
        redirect_to: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        query: Union[Dict[str, str], None] = None,
        body: Union[Any, None] = None,
        no_resolve_json: bool = False,
    ) -> None:
        ...  # pragma: no cover

    async def _request(
        self,
        method: Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"],
        path: str,
        *,
        jwt: Union[str, None] = None,
        redirect_to: Union[str, None] = None,
        headers: Union[Dict[str, str], None] = None,
        query: Union[Dict[str, str], None] = None,
        body: Union[Any, None] = None,
        no_resolve_json: bool = False,
        xform: Union[Callable[[Any], T], None] = None,
    ) -> Union[T, None]:
        url = f"{self._url}/{path}"
        headers = {**self._headers, **(headers or {})}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json;charset=UTF-8"
        if jwt:
            headers["Authorization"] = f"Bearer {jwt}"
        query = query or {}
        if redirect_to:
            query["redirect_to"] = redirect_to
        try:
            response = await self._http_client.request(
                method,
                url,
                headers=headers,
                params=query,
                json=model_dump(body) if isinstance(body, BaseModel) else body,
            )
            response.raise_for_status()
            result = response if no_resolve_json else response.json()
            if xform:
                return xform(result)
        except Exception as e:
            raise handle_exception(e)
