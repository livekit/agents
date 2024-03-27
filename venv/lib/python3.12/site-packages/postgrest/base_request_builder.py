from __future__ import annotations

import json
from json import JSONDecodeError
from re import search
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from httpx import Headers, QueryParams
from httpx import Response as RequestResponse
from pydantic import BaseModel

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

try:
    # >= 2.0.0
    from pydantic import field_validator
except ImportError:
    # < 2.0.0
    from pydantic import validator as field_validator

from .types import CountMethod, Filters, RequestMethod, ReturnMethod
from .utils import AsyncClient, SyncClient, get_origin_and_cast, sanitize_param


class QueryArgs(NamedTuple):
    # groups the method, json, headers and params for a query in a single object
    method: RequestMethod
    params: QueryParams
    headers: Headers
    json: Dict[Any, Any]


def pre_select(
    *columns: str,
    count: Optional[CountMethod] = None,
) -> QueryArgs:
    if columns:
        method = RequestMethod.GET
        params = QueryParams({"select": ",".join(columns)})
    else:
        method = RequestMethod.HEAD
        params = QueryParams()
    headers = Headers({"Prefer": f"count={count}"}) if count else Headers()
    return QueryArgs(method, params, headers, {})


def pre_insert(
    json: dict,
    *,
    count: Optional[CountMethod],
    returning: ReturnMethod,
    upsert: bool,
) -> QueryArgs:
    prefer_headers = [f"return={returning}"]
    if count:
        prefer_headers.append(f"count={count}")
    if upsert:
        prefer_headers.append("resolution=merge-duplicates")
    headers = Headers({"Prefer": ",".join(prefer_headers)})
    return QueryArgs(RequestMethod.POST, QueryParams(), headers, json)


def pre_upsert(
    json: dict,
    *,
    count: Optional[CountMethod],
    returning: ReturnMethod,
    ignore_duplicates: bool,
    on_conflict: str = "",
) -> QueryArgs:
    query_params = {}
    prefer_headers = [f"return={returning}"]
    if count:
        prefer_headers.append(f"count={count}")
    resolution = "ignore" if ignore_duplicates else "merge"
    prefer_headers.append(f"resolution={resolution}-duplicates")
    headers = Headers({"Prefer": ",".join(prefer_headers)})
    if on_conflict:
        query_params["on_conflict"] = on_conflict
    return QueryArgs(RequestMethod.POST, QueryParams(query_params), headers, json)


def pre_update(
    json: dict,
    *,
    count: Optional[CountMethod],
    returning: ReturnMethod,
) -> QueryArgs:
    prefer_headers = [f"return={returning}"]
    if count:
        prefer_headers.append(f"count={count}")
    headers = Headers({"Prefer": ",".join(prefer_headers)})
    return QueryArgs(RequestMethod.PATCH, QueryParams(), headers, json)


def pre_delete(
    *,
    count: Optional[CountMethod],
    returning: ReturnMethod,
) -> QueryArgs:
    prefer_headers = [f"return={returning}"]
    if count:
        prefer_headers.append(f"count={count}")
    headers = Headers({"Prefer": ",".join(prefer_headers)})
    return QueryArgs(RequestMethod.DELETE, QueryParams(), headers, {})


_ReturnT = TypeVar("_ReturnT")


# the APIResponse.data is marked as _ReturnT instead of list[_ReturnT]
# as it is also returned in the case of rpc() calls; and rpc calls do not
# necessarily return lists.
# https://github.com/supabase-community/postgrest-py/issues/200
class APIResponse(BaseModel, Generic[_ReturnT]):
    data: List[_ReturnT]
    """The data returned by the query."""
    count: Optional[int] = None
    """The number of rows returned."""

    @field_validator("data")
    @classmethod
    def raise_when_api_error(cls: Type[Self], value: Any) -> Any:
        if isinstance(value, dict) and value.get("message"):
            raise ValueError("You are passing an API error to the data field.")
        return value

    @staticmethod
    def _get_count_from_content_range_header(
        content_range_header: str,
    ) -> Optional[int]:
        content_range = content_range_header.split("/")
        return None if len(content_range) < 2 else int(content_range[1])

    @staticmethod
    def _is_count_in_prefer_header(prefer_header: str) -> bool:
        pattern = f"count=({'|'.join([cm.value for cm in CountMethod])})"
        return bool(search(pattern, prefer_header))

    @classmethod
    def _get_count_from_http_request_response(
        cls: Type[Self],
        request_response: RequestResponse,
    ) -> Optional[int]:
        prefer_header: Optional[str] = request_response.request.headers.get("prefer")
        if not prefer_header:
            return None
        is_count_in_prefer_header = cls._is_count_in_prefer_header(prefer_header)
        content_range_header: Optional[str] = request_response.headers.get(
            "content-range"
        )
        return (
            cls._get_count_from_content_range_header(content_range_header)
            if (is_count_in_prefer_header and content_range_header)
            else None
        )

    @classmethod
    def from_http_request_response(
        cls: Type[Self], request_response: RequestResponse
    ) -> Self:
        try:
            data = request_response.json()
        except JSONDecodeError as e:
            return cls(data=[], count=0)
        count = cls._get_count_from_http_request_response(request_response)
        # the type-ignore here is as pydantic needs us to pass the type parameter
        # here explicitly, but pylance already knows that cls is correctly parametrized
        return cls[_ReturnT](data=data, count=count)  # type: ignore

    @classmethod
    def from_dict(cls: Type[Self], dict: Dict[str, Any]) -> Self:
        keys = dict.keys()
        assert len(keys) == 3 and "data" in keys and "count" in keys and "error" in keys
        return cls[_ReturnT](  # type: ignore
            data=dict.get("data"), count=dict.get("count"), error=dict.get("error")
        )


class SingleAPIResponse(APIResponse[_ReturnT], Generic[_ReturnT]):
    data: _ReturnT  # type: ignore
    """The data returned by the query."""

    @classmethod
    def from_http_request_response(
        cls: Type[Self], request_response: RequestResponse
    ) -> Self:
        count = cls._get_count_from_http_request_response(request_response)
        try:
            data = request_response.json()
        except JSONDecodeError:
            data = request_response.text if len(request_response.text) > 0 else []
        return cls[_ReturnT](data=data, count=count)  # type: ignore

    @classmethod
    def from_dict(cls: Type[Self], dict: Dict[str, Any]) -> Self:
        keys = dict.keys()
        assert len(keys) == 3 and "data" in keys and "count" in keys and "error" in keys
        return cls[_ReturnT](  # type: ignore
            data=dict.get("data"), count=dict.get("count"), error=dict.get("error")
        )


class BaseFilterRequestBuilder(Generic[_ReturnT]):
    def __init__(
        self,
        session: Union[AsyncClient, SyncClient],
        headers: Headers,
        params: QueryParams,
    ) -> None:
        self.session = session
        self.headers = headers
        self.params = params
        self.negate_next = False

    @property
    def not_(self: Self) -> Self:
        """Whether the filter applied next should be negated."""
        self.negate_next = True
        return self

    def filter(self: Self, column: str, operator: str, criteria: str) -> Self:
        """Apply filters on a query.

        Args:
            column: The name of the column to apply a filter on
            operator: The operator to use while filtering
            criteria: The value to filter by
        """
        if self.negate_next is True:
            self.negate_next = False
            operator = f"{Filters.NOT}.{operator}"
        key, val = sanitize_param(column), f"{operator}.{criteria}"
        self.params = self.params.add(key, val)
        return self

    def eq(self: Self, column: str, value: Any) -> Self:
        """An 'equal to' filter.

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.EQ, value)

    def neq(self: Self, column: str, value: Any) -> Self:
        """A 'not equal to' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.NEQ, value)

    def gt(self: Self, column: str, value: Any) -> Self:
        """A 'greater than' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.GT, value)

    def gte(self: Self, column: str, value: Any) -> Self:
        """A 'greater than or equal to' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.GTE, value)

    def lt(self: Self, column: str, value: Any) -> Self:
        """A 'less than' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.LT, value)

    def lte(self: Self, column: str, value: Any) -> Self:
        """A 'less than or equal to' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.LTE, value)

    def is_(self: Self, column: str, value: Any) -> Self:
        """An 'is' filter

        Args:
            column: The name of the column to apply a filter on
            value: The value to filter by
        """
        return self.filter(column, Filters.IS, value)

    def like(self: Self, column: str, pattern: str) -> Self:
        """A 'LIKE' filter, to use for pattern matching.

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """
        return self.filter(column, Filters.LIKE, pattern)

    def like_all_of(self: Self, column: str, pattern: str) -> Self:
        """A 'LIKE' filter, to use for pattern matching.

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """

        return self.filter(column, Filters.LIKE_ALL, f"{{{pattern}}}")

    def like_any_of(self: Self, column: str, pattern: str) -> Self:
        """A 'LIKE' filter, to use for pattern matching.

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """

        return self.filter(column, Filters.LIKE_ANY, f"{{{pattern}}}")

    def ilike_all_of(self: Self, column: str, pattern: str) -> Self:
        """A 'ILIKE' filter, to use for pattern matching (case insensitive).

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """

        return self.filter(column, Filters.ILIKE_ALL, f"{{{pattern}}}")

    def ilike_any_of(self: Self, column: str, pattern: str) -> Self:
        """A 'ILIKE' filter, to use for pattern matching (case insensitive).

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """

        return self.filter(column, Filters.ILIKE_ANY, f"{{{pattern}}}")

    def ilike(self: Self, column: str, pattern: str) -> Self:
        """An 'ILIKE' filter, to use for pattern matching (case insensitive).

        Args:
            column: The name of the column to apply a filter on
            pattern: The pattern to filter by
        """
        return self.filter(column, Filters.ILIKE, pattern)

    def or_(self: Self, filters: str, reference_table: Union[str, None] = None) -> Self:
        """An 'or' filter

        Args:
            filters: The filters to use, following PostgREST syntax
            reference_table: Set this to filter on referenced tables instead of the parent table
        """
        key = f"{sanitize_param(reference_table)}.or" if reference_table else "or"
        self.params = self.params.add(key, f"({filters})")
        return self

    def fts(self: Self, column: str, query: Any) -> Self:
        return self.filter(column, Filters.FTS, query)

    def plfts(self: Self, column: str, query: Any) -> Self:
        return self.filter(column, Filters.PLFTS, query)

    def phfts(self: Self, column: str, query: Any) -> Self:
        return self.filter(column, Filters.PHFTS, query)

    def wfts(self: Self, column: str, query: Any) -> Self:
        return self.filter(column, Filters.WFTS, query)

    def in_(self: Self, column: str, values: Iterable[Any]) -> Self:
        values = map(sanitize_param, values)
        values = ",".join(values)
        return self.filter(column, Filters.IN, f"({values})")

    def cs(self: Self, column: str, values: Iterable[Any]) -> Self:
        values = ",".join(values)
        return self.filter(column, Filters.CS, f"{{{values}}}")

    def cd(self: Self, column: str, values: Iterable[Any]) -> Self:
        values = ",".join(values)
        return self.filter(column, Filters.CD, f"{{{values}}}")

    def contains(
        self: Self, column: str, value: Union[Iterable[Any], str, Dict[Any, Any]]
    ) -> Self:
        if isinstance(value, str):
            # range types can be inclusive '[', ']' or exclusive '(', ')' so just
            # keep it simple and accept a string
            return self.filter(column, Filters.CS, value)
        if not isinstance(value, dict) and isinstance(value, Iterable):
            # Expected to be some type of iterable
            stringified_values = ",".join(value)
            return self.filter(column, Filters.CS, f"{{{stringified_values}}}")

        return self.filter(column, Filters.CS, json.dumps(value))

    def contained_by(
        self: Self, column: str, value: Union[Iterable[Any], str, Dict[Any, Any]]
    ) -> Self:
        if isinstance(value, str):
            # range
            return self.filter(column, Filters.CD, value)
        if not isinstance(value, dict) and isinstance(value, Iterable):
            stringified_values = ",".join(value)
            return self.filter(column, Filters.CD, f"{{{stringified_values}}}")
        return self.filter(column, Filters.CD, json.dumps(value))

    def ov(self: Self, column: str, values: Iterable[Any]) -> Self:
        values = ",".join(values)
        return self.filter(column, Filters.OV, f"{{{values}}}")

    def sl(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.filter(column, Filters.SL, f"({range[0]},{range[1]})")

    def sr(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.filter(column, Filters.SR, f"({range[0]},{range[1]})")

    def nxl(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.filter(column, Filters.NXL, f"({range[0]},{range[1]})")

    def nxr(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.filter(column, Filters.NXR, f"({range[0]},{range[1]})")

    def adj(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.filter(column, Filters.ADJ, f"({range[0]},{range[1]})")

    def range_gt(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.sr(column, range)

    def range_gte(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.nxl(column, range)

    def range_lt(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.sl(column, range)

    def range_lte(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.nxr(column, range)

    def range_adjacent(self: Self, column: str, range: Tuple[int, int]) -> Self:
        return self.adj(column, range)

    def overlaps(self: Self, column: str, values: Iterable[Any]) -> Self:
        return self.ov(column, values)

    def match(self: Self, query: Dict[str, Any]) -> Self:
        updated_query = self

        if not query:
            raise ValueError(
                "query dictionary should contain at least one key-value pair"
            )

        for key, value in query.items():
            updated_query = self.eq(key, value)

        return updated_query


class BaseSelectRequestBuilder(BaseFilterRequestBuilder[_ReturnT]):
    def __init__(
        self,
        session: Union[AsyncClient, SyncClient],
        headers: Headers,
        params: QueryParams,
    ) -> None:
        # Generic[T] is an instance of typing._GenericAlias, so doing Generic[T].__init__
        # tries to call _GenericAlias.__init__ - which is the wrong method
        # The __origin__ attribute of the _GenericAlias is the actual class
        get_origin_and_cast(BaseFilterRequestBuilder[_ReturnT]).__init__(
            self, session, headers, params
        )

    def explain(
        self: Self,
        analyze: bool = False,
        verbose: bool = False,
        settings: bool = False,
        buffers: bool = False,
        wal: bool = False,
        format: Literal["text", "json"] = "text",
    ) -> Self:
        options = [
            key
            for key, value in locals().items()
            if key not in ["self", "format"] and value
        ]
        options_str = "|".join(options)
        self.headers[
            "Accept"
        ] = f"application/vnd.pgrst.plan+{format}; options={options_str}"
        return self

    def order(
        self: Self,
        column: str,
        *,
        desc: bool = False,
        nullsfirst: bool = False,
        foreign_table: Optional[str] = None,
    ) -> Self:
        """Sort the returned rows in some specific order.

        Args:
            column: The column to order by
            desc: Whether the rows should be ordered in descending order or not.
            nullsfirst: nullsfirst
            foreign_table: Foreign table name whose results are to be ordered.
        .. versionchanged:: 0.10.3
           Allow ordering results for foreign tables with the foreign_table parameter.
        """
        self.params = self.params.add(
            f"{foreign_table}.order" if foreign_table else "order",
            f"{column}{'.desc' if desc else ''}{'.nullsfirst' if nullsfirst else ''}",
        )
        return self

    def limit(self: Self, size: int, *, foreign_table: Optional[str] = None) -> Self:
        """Limit the number of rows returned by a query.

        Args:
            size: The number of rows to be returned
            foreign_table: Foreign table name to limit
        .. versionchanged:: 0.10.3
           Allow limiting results returned for foreign tables with the foreign_table parameter.
        """
        self.params = self.params.add(
            f"{foreign_table}.limit" if foreign_table else "limit",
            size,
        )
        return self

    def offset(self: _FilterT, size: int) -> _FilterT:
        """Set the starting row index returned by a query.
        Args:
            size: The number of the row to start at
        """
        self.params = self.params.add(
            "offset",
            size,
        )
        return self

    def range(
        self: Self, start: int, end: int, foreign_table: Optional[str] = None
    ) -> Self:
        self.params = self.params.add(
            f"{foreign_table}.offset" if foreign_table else "offset", start
        )
        self.params = self.params.add(
            f"{foreign_table}.limit" if foreign_table else "limit",
            end - start + 1,
        )
        return self


class BaseRPCRequestBuilder(BaseSelectRequestBuilder[_ReturnT]):
    def __init__(
        self,
        session: Union[AsyncClient, SyncClient],
        headers: Headers,
        params: QueryParams,
    ) -> None:
        # Generic[T] is an instance of typing._GenericAlias, so doing Generic[T].__init__
        # tries to call _GenericAlias.__init__ - which is the wrong method
        # The __origin__ attribute of the _GenericAlias is the actual class
        get_origin_and_cast(BaseSelectRequestBuilder[_ReturnT]).__init__(
            self, session, headers, params
        )

    def select(
        self,
        *columns: str,
    ) -> Self:
        """Run a SELECT query.

        Args:
            *columns: The names of the columns to fetch.
        Returns:
            :class:`BaseSelectRequestBuilder`
        """
        method, params, headers, json = pre_select(*columns, count=None)
        self.params = self.params.add("select", params.get("select"))
        self.headers["Prefer"] = "return=representation"
        return self

    def single(self) -> Self:
        """Specify that the query will only return a single row in response.

        .. caution::
            The API will raise an error if the query returned more than one row.
        """
        self.headers["Accept"] = "application/vnd.pgrst.object+json"
        return self

    def maybe_single(self) -> Self:
        """Retrieves at most one row from the result. Result must be at most one row (e.g. using `eq` on a UNIQUE column), otherwise this will result in an error."""
        self.headers["Accept"] = "application/vnd.pgrst.object+json"
        return self

    def csv(self) -> Self:
        """Specify that the query must retrieve data as a single CSV string."""
        self.headers["Accept"] = "text/csv"
        return self
