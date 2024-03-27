from __future__ import annotations

from json import JSONDecodeError
from typing import Any, Generic, Optional, TypeVar, Union

from httpx import Headers, QueryParams
from pydantic import ValidationError

from ..base_request_builder import (
    APIResponse,
    BaseFilterRequestBuilder,
    BaseRPCRequestBuilder,
    BaseSelectRequestBuilder,
    CountMethod,
    SingleAPIResponse,
    pre_delete,
    pre_insert,
    pre_select,
    pre_update,
    pre_upsert,
)
from ..exceptions import APIError, generate_default_error_message
from ..types import ReturnMethod
from ..utils import AsyncClient, get_origin_and_cast

_ReturnT = TypeVar("_ReturnT")


class AsyncQueryRequestBuilder(Generic[_ReturnT]):
    def __init__(
        self,
        session: AsyncClient,
        path: str,
        http_method: str,
        headers: Headers,
        params: QueryParams,
        json: dict,
    ) -> None:
        self.session = session
        self.path = path
        self.http_method = http_method
        self.headers = headers
        self.params = params
        self.json = json

    async def execute(self) -> APIResponse[_ReturnT]:
        """Execute the query.

        .. tip::
            This is the last method called, after the query is built.

        Returns:
            :class:`APIResponse`

        Raises:
            :class:`APIError` If the API raised an error.
        """
        r = await self.session.request(
            self.http_method,
            self.path,
            json=self.json,
            params=self.params,
            headers=self.headers,
        )
        try:
            if r.is_success:
                if self.http_method != "HEAD":
                    body = r.text
                    if self.headers.get("Accept") == "text/csv":
                        return body
                    if self.headers.get(
                        "Accept"
                    ) and "application/vnd.pgrst.plan" in self.headers.get("Accept"):
                        if not "+json" in self.headers.get("Accept"):
                            return body
                return APIResponse[_ReturnT].from_http_request_response(r)
            else:
                raise APIError(r.json())
        except ValidationError as e:
            raise APIError(r.json()) from e
        except JSONDecodeError as e:
            raise APIError(generate_default_error_message(r))


class AsyncSingleRequestBuilder(Generic[_ReturnT]):
    def __init__(
        self,
        session: AsyncClient,
        path: str,
        http_method: str,
        headers: Headers,
        params: QueryParams,
        json: dict,
    ) -> None:
        self.session = session
        self.path = path
        self.http_method = http_method
        self.headers = headers
        self.params = params
        self.json = json

    async def execute(self) -> SingleAPIResponse[_ReturnT]:
        """Execute the query.

        .. tip::
            This is the last method called, after the query is built.

        Returns:
            :class:`SingleAPIResponse`

        Raises:
            :class:`APIError` If the API raised an error.
        """
        r = await self.session.request(
            self.http_method,
            self.path,
            json=self.json,
            params=self.params,
            headers=self.headers,
        )
        try:
            if (
                200 <= r.status_code <= 299
            ):  # Response.ok from JS (https://developer.mozilla.org/en-US/docs/Web/API/Response/ok)
                return SingleAPIResponse[_ReturnT].from_http_request_response(r)
            else:
                raise APIError(r.json())
        except ValidationError as e:
            raise APIError(r.json()) from e
        except JSONDecodeError as e:
            raise APIError(generate_default_error_message(r))


class AsyncMaybeSingleRequestBuilder(AsyncSingleRequestBuilder[_ReturnT]):
    async def execute(self) -> Optional[SingleAPIResponse[_ReturnT]]:
        r = None
        try:
            r = await AsyncSingleRequestBuilder[_ReturnT].execute(self)
        except APIError as e:
            if e.details and "The result contains 0 rows" in e.details:
                return None
        if not r:
            raise APIError(
                {
                    "message": "Missing response",
                    "code": "204",
                    "hint": "Please check traceback of the code",
                    "details": "Postgrest couldn't retrieve response, please check traceback of the code. Please create an issue in `supabase-community/postgrest-py` if needed.",
                }
            )
        return r


# ignoring type checking as a workaround for https://github.com/python/mypy/issues/9319
class AsyncFilterRequestBuilder(BaseFilterRequestBuilder[_ReturnT], AsyncQueryRequestBuilder[_ReturnT]):  # type: ignore
    def __init__(
        self,
        session: AsyncClient,
        path: str,
        http_method: str,
        headers: Headers,
        params: QueryParams,
        json: dict,
    ) -> None:
        get_origin_and_cast(BaseFilterRequestBuilder[_ReturnT]).__init__(
            self, session, headers, params
        )
        get_origin_and_cast(AsyncQueryRequestBuilder[_ReturnT]).__init__(
            self, session, path, http_method, headers, params, json
        )


# this exists for type-safety. see https://gist.github.com/anand2312/93d3abf401335fd3310d9e30112303bf
class AsyncRPCFilterRequestBuilder(
    BaseRPCRequestBuilder[_ReturnT], AsyncSingleRequestBuilder[_ReturnT]
):
    def __init__(
        self,
        session: AsyncClient,
        path: str,
        http_method: str,
        headers: Headers,
        params: QueryParams,
        json: dict,
    ) -> None:
        get_origin_and_cast(BaseFilterRequestBuilder[_ReturnT]).__init__(
            self, session, headers, params
        )
        get_origin_and_cast(AsyncSingleRequestBuilder[_ReturnT]).__init__(
            self, session, path, http_method, headers, params, json
        )


# ignoring type checking as a workaround for https://github.com/python/mypy/issues/9319
class AsyncSelectRequestBuilder(BaseSelectRequestBuilder[_ReturnT], AsyncQueryRequestBuilder[_ReturnT]):  # type: ignore
    def __init__(
        self,
        session: AsyncClient,
        path: str,
        http_method: str,
        headers: Headers,
        params: QueryParams,
        json: dict,
    ) -> None:
        get_origin_and_cast(BaseSelectRequestBuilder[_ReturnT]).__init__(
            self, session, headers, params
        )
        get_origin_and_cast(AsyncQueryRequestBuilder[_ReturnT]).__init__(
            self, session, path, http_method, headers, params, json
        )

    def single(self) -> AsyncSingleRequestBuilder[_ReturnT]:
        """Specify that the query will only return a single row in response.

        .. caution::
            The API will raise an error if the query returned more than one row.
        """
        self.headers["Accept"] = "application/vnd.pgrst.object+json"
        return AsyncSingleRequestBuilder[_ReturnT](
            headers=self.headers,
            http_method=self.http_method,
            json=self.json,
            params=self.params,
            path=self.path,
            session=self.session,  # type: ignore
        )

    def maybe_single(self) -> AsyncMaybeSingleRequestBuilder[_ReturnT]:
        """Retrieves at most one row from the result. Result must be at most one row (e.g. using `eq` on a UNIQUE column), otherwise this will result in an error."""
        self.headers["Accept"] = "application/vnd.pgrst.object+json"
        return AsyncMaybeSingleRequestBuilder[_ReturnT](
            headers=self.headers,
            http_method=self.http_method,
            json=self.json,
            params=self.params,
            path=self.path,
            session=self.session,  # type: ignore
        )

    def text_search(
        self, column: str, query: str, options: dict[str, Any] = {}
    ) -> AsyncFilterRequestBuilder[_ReturnT]:
        type_ = options.get("type")
        type_part = ""
        if type_ == "plain":
            type_part = "pl"
        elif type_ == "phrase":
            type_part = "ph"
        elif type_ == "web_search":
            type_part = "w"
        config_part = f"({options.get('config')})" if options.get("config") else ""
        self.params = self.params.add(column, f"{type_part}fts{config_part}.{query}")

        return AsyncQueryRequestBuilder[_ReturnT](
            headers=self.headers,
            http_method=self.http_method,
            json=self.json,
            params=self.params,
            path=self.path,
            session=self.session,  # type: ignore
        )

    def csv(self) -> AsyncSingleRequestBuilder[str]:
        """Specify that the query must retrieve data as a single CSV string."""
        self.headers["Accept"] = "text/csv"
        return AsyncSingleRequestBuilder[str](
            session=self.session,  # type: ignore
            path=self.path,
            http_method=self.http_method,
            headers=self.headers,
            params=self.params,
            json=self.json,
        )


class AsyncRequestBuilder(Generic[_ReturnT]):
    def __init__(self, session: AsyncClient, path: str) -> None:
        self.session = session
        self.path = path

    def select(
        self,
        *columns: str,
        count: Optional[CountMethod] = None,
    ) -> AsyncSelectRequestBuilder[_ReturnT]:
        """Run a SELECT query.

        Args:
            *columns: The names of the columns to fetch.
            count: The method to use to get the count of rows returned.
        Returns:
            :class:`AsyncSelectRequestBuilder`
        """
        method, params, headers, json = pre_select(*columns, count=count)
        return AsyncSelectRequestBuilder[_ReturnT](
            self.session, self.path, method, headers, params, json
        )

    def insert(
        self,
        json: Union[dict, list],
        *,
        count: Optional[CountMethod] = None,
        returning: ReturnMethod = ReturnMethod.representation,
        upsert: bool = False,
    ) -> AsyncQueryRequestBuilder[_ReturnT]:
        """Run an INSERT query.

        Args:
            json: The row to be inserted.
            count: The method to use to get the count of rows returned.
            returning: Either 'minimal' or 'representation'
            upsert: Whether the query should be an upsert.
        Returns:
            :class:`AsyncQueryRequestBuilder`
        """
        method, params, headers, json = pre_insert(
            json,
            count=count,
            returning=returning,
            upsert=upsert,
        )
        return AsyncQueryRequestBuilder[_ReturnT](
            self.session, self.path, method, headers, params, json
        )

    def upsert(
        self,
        json: Union[dict, list],
        *,
        count: Optional[CountMethod] = None,
        returning: ReturnMethod = ReturnMethod.representation,
        ignore_duplicates: bool = False,
        on_conflict: str = "",
    ) -> AsyncQueryRequestBuilder[_ReturnT]:
        """Run an upsert (INSERT ... ON CONFLICT DO UPDATE) query.

        Args:
            json: The row to be inserted.
            count: The method to use to get the count of rows returned.
            returning: Either 'minimal' or 'representation'
            ignore_duplicates: Whether duplicate rows should be ignored.
            on_conflict: Specified columns to be made to work with UNIQUE constraint.
        Returns:
            :class:`AsyncQueryRequestBuilder`
        """
        method, params, headers, json = pre_upsert(
            json,
            count=count,
            returning=returning,
            ignore_duplicates=ignore_duplicates,
            on_conflict=on_conflict,
        )
        return AsyncQueryRequestBuilder[_ReturnT](
            self.session, self.path, method, headers, params, json
        )

    def update(
        self,
        json: dict,
        *,
        count: Optional[CountMethod] = None,
        returning: ReturnMethod = ReturnMethod.representation,
    ) -> AsyncFilterRequestBuilder[_ReturnT]:
        """Run an UPDATE query.

        Args:
            json: The updated fields.
            count: The method to use to get the count of rows returned.
            returning: Either 'minimal' or 'representation'
        Returns:
            :class:`AsyncFilterRequestBuilder`
        """
        method, params, headers, json = pre_update(
            json,
            count=count,
            returning=returning,
        )
        return AsyncFilterRequestBuilder[_ReturnT](
            self.session, self.path, method, headers, params, json
        )

    def delete(
        self,
        *,
        count: Optional[CountMethod] = None,
        returning: ReturnMethod = ReturnMethod.representation,
    ) -> AsyncFilterRequestBuilder[_ReturnT]:
        """Run a DELETE query.

        Args:
            count: The method to use to get the count of rows returned.
            returning: Either 'minimal' or 'representation'
        Returns:
            :class:`AsyncFilterRequestBuilder`
        """
        method, params, headers, json = pre_delete(
            count=count,
            returning=returning,
        )
        return AsyncFilterRequestBuilder[_ReturnT](
            self.session, self.path, method, headers, params, json
        )
