"""Toolsets that keep a large tool collection out of the model's context: ``ToolSearchToolset``
loads the tools a search finds, ``ToolProxyToolset`` keeps two fixed tools (search and call).
The ranking, the arguments' checks and the replies to the model are the core's; a custom
``SearchStrategy`` (an embedding search, say) can stand in for the built-in ones."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol

from typing_extensions import Self

from .. import _core
from ..llm.tool_context import (
    FunctionTool,
    ProviderTool,
    RawFunctionTool,
    Tool,
    ToolContext,
    Toolset,
    function_tool,
)
from ..llm.utils import function_arguments_to_pydantic_model, prepare_function_arguments
from ..types import NOT_GIVEN, NotGivenOr
from ..voice.events import RunContext

__all__ = [
    "BM25SearchStrategy",
    "KeywordSearchStrategy",
    "SearchItem",
    "SearchStrategy",
    "ToolProxyToolset",
    "ToolSearchToolset",
]


@dataclass
class SearchItem:
    """A search candidate derived from a single tool at index time."""

    source: Tool | Toolset
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # {name: description}
    index_data: Any = field(default=None, repr=False)


class SearchStrategy(Protocol):
    def build_index(self, items: list[SearchItem]) -> None | Awaitable[None]: ...
    def search(
        self, query: str, items: list[SearchItem], max_results: int
    ) -> list[SearchItem] | Awaitable[list[SearchItem]]: ...
    def cleanup(self) -> None | Awaitable[None]: ...


_DEFAULT_SEARCH_DESCRIPTION = (
    "Search for available tools by describing what you need. "
    "The matching tools will become available for use after calling this tool. "
    "Call this before attempting to use a tool that isn't "
    "currently available."
)

_DEFAULT_QUERY_DESCRIPTION = (
    "keywords to search for in the tool names and descriptions, split by spaces"
)


class ToolSearchToolset(Toolset):
    """Wraps tools/toolsets and exposes a tool_search function for dynamic loading.

    Instead of loading all tool definitions into LLM context, this exposes a single
    ``tool_search`` function. When the LLM calls it, matching tools are dynamically
    loaded into the context.

    Each tool (FunctionTool, RawFunctionTool, ProviderTool) is indexed as its own
    SearchItem. If a matched tool belongs to a Toolset, the entire Toolset is loaded
    atomically.
    """

    def __init__(
        self,
        *,
        id: str,
        tools: list[Tool | Toolset] | None = None,
        max_results: int = 5,
        search_strategy: NotGivenOr[SearchStrategy] = NOT_GIVEN,
        search_description: NotGivenOr[str] = NOT_GIVEN,
        query_description: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(id=id, tools=tools)
        self._strategy = search_strategy or BM25SearchStrategy()
        self._max_results = max_results
        self._loaded_tools: list[Tool | Toolset] = []

        self._search_items: list[SearchItem] = []
        self._initialized = False
        self._lock = asyncio.Lock()

        search_description = search_description or _DEFAULT_SEARCH_DESCRIPTION
        query_description = query_description or _DEFAULT_QUERY_DESCRIPTION
        self._search_tool = function_tool(
            self._handle_search,
            raw_schema={
                "name": "tool_search",
                "description": search_description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": query_description}},
                    "required": ["query"],
                },
            },
        )

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [self._search_tool, *self._loaded_tools]

    async def setup(self, *, reload: bool = False) -> Self:
        await super().setup()
        async with self._lock:
            if not reload and self._initialized:
                return self

            # setup wrapped toolsets
            toolsets = [t for t in self._tools if isinstance(t, Toolset)]
            if toolsets:
                await asyncio.gather(*(ts.setup() for ts in toolsets))

            self._search_items = []

            def _index_tool(tool: Tool | Toolset, source: Tool | Toolset) -> None:
                if isinstance(tool, Toolset):
                    tool_ctx = ToolContext([tool])
                    for tool in tool_ctx.flatten():
                        _index_tool(tool, source)
                elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                    self._search_items.append(
                        SearchItem(
                            name=tool.id,
                            description=_get_tool_description(tool),
                            parameters=_get_tool_params(tool),
                            source=source,
                        )
                    )
                elif isinstance(tool, ProviderTool):
                    self._search_items.append(
                        SearchItem(name=tool.id, description="", parameters={}, source=source)
                    )
                else:
                    raise ValueError(f"Unsupported tool type: {type(tool)}")

            for tool in self._tools:
                _index_tool(tool, tool)

            result = self._strategy.build_index(self._search_items)
            if inspect.isawaitable(result):
                await result
            self._initialized = True
            return self

    async def _handle_search(self, raw_arguments: dict[str, object]) -> str:
        query = str(raw_arguments.get("query", ""))
        tools = await self._search_tools(query)
        loaded: str = _core.tool_search_loaded(query, len(tools))
        self._loaded_tools = tools
        return loaded

    async def _search_tools(self, query: str) -> list[Tool | Toolset]:
        _core.tool_search_validate(query)
        results = self._strategy.search(query, self._search_items, self._max_results)
        if inspect.isawaitable(results):
            results = await results

        return list(dict.fromkeys(result.source for result in results))

    async def aclose(self) -> None:
        await super().aclose()
        self._initialized = False
        self._search_items.clear()
        self._loaded_tools.clear()

        result = self._strategy.cleanup()
        if inspect.isawaitable(result):
            await result


def _get_tool_description(tool: FunctionTool | RawFunctionTool) -> str:
    if isinstance(tool, FunctionTool):
        return tool.info.description or ""
    return str(tool.info.raw_schema.get("description", ""))


def _get_tool_params(tool: FunctionTool | RawFunctionTool) -> dict[str, str]:
    if isinstance(tool, FunctionTool):
        model = function_arguments_to_pydantic_model(tool)
        return {name: field.description or "" for name, field in model.model_fields.items()}

    props = tool.info.raw_schema.get("parameters", {}).get("properties", {})
    return {
        name: prop.get("description", "") if isinstance(prop, dict) else ""
        for name, prop in props.items()
    }


def _document(item: SearchItem) -> _core.SearchDocument:
    return _core.SearchDocument(item.name, item.description, list(item.parameters.items()))


class KeywordSearchStrategy:
    """Keyword search using regex matching (the core's ``keyword_search``).

    Scoring: name match = 3pts, description match = 2pts, parameter name/desc match = 1pt each.
    """

    def build_index(self, items: list[SearchItem]) -> None:
        for item in items:
            item.index_data = _document(item)

    def search(self, query: str, items: list[SearchItem], max_results: int) -> list[SearchItem]:
        docs = [item.index_data or _document(item) for item in items]
        return [items[i] for i in _core.keyword_search(query, docs, max_results)]

    def cleanup(self) -> None:
        pass


class BM25SearchStrategy:
    """BM25-based search strategy (the core's ``BM25Index``).

    BM25 ranks items by term frequency, inverse document frequency, and document
    length normalization. Better than simple keyword matching for larger tool
    collections because it down-weights common terms and rewards rare, specific matches.

    Each SearchItem is treated as a document composed of its name (weighted 3x),
    description (weighted 2x), and parameter names/descriptions (weighted 1x).

    Args:
        k1: Term frequency saturation parameter. Higher values give more weight to
            repeated terms. Default 1.5.
        b: Length normalization parameter (0-1). Higher values penalize longer
            documents more. Default 0.75.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._index = _core.BM25Index(k1, b)

    def build_index(self, items: list[SearchItem]) -> None:
        for item in items:
            item.index_data = _document(item)
        self._index.build([item.index_data for item in items])

    def search(self, query: str, items: list[SearchItem], max_results: int) -> list[SearchItem]:
        docs = [item.index_data for item in items]
        assert all(d is not None for d in docs), "index data must be built before scoring"
        return [items[i] for i in self._index.search(query, docs, max_results)]

    def cleanup(self) -> None:
        self._index.clear()


_PROXY_SEARCH_DESCRIPTION = (
    "Search for available tools by describing what you need. "
    "Returns the schemas of matching tools. Use call_tool to invoke them."
)

_DEFAULT_CALL_DESCRIPTION = (
    "Call a tool by name with the given arguments. "
    "Use tool_search to discover available tools and their schemas if it isn't already loaded."
)


class ToolProxyToolset(ToolSearchToolset):
    """Exposes exactly two fixed tools: tool_search and call_tool.

    Unlike ToolSearchToolset which dynamically modifies the tool list,
    ToolProxyToolset keeps a constant tool list. ``tool_search`` returns
    tool schemas as text, and ``call_tool`` executes tools by name.

    This is useful for maximizing prompt cache hit rates with providers
    that cache based on tool definitions (e.g. Anthropic, OpenAI).
    """

    def __init__(
        self,
        *,
        id: str,
        tools: list[Tool | Toolset] | None = None,
        max_results: int = 5,
        search_strategy: NotGivenOr[SearchStrategy] = NOT_GIVEN,
        search_description: NotGivenOr[str] = NOT_GIVEN,
        query_description: NotGivenOr[str] = NOT_GIVEN,
        call_description: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            id=id,
            tools=tools,
            max_results=max_results,
            search_strategy=search_strategy,
            search_description=search_description or _PROXY_SEARCH_DESCRIPTION,
            query_description=query_description,
        )
        self._tool_ctx: ToolContext | None = None

        call_description = call_description or _DEFAULT_CALL_DESCRIPTION
        self._call_tool = function_tool(
            self._handle_call,
            raw_schema={
                "name": "call_tool",
                "description": call_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the tool to call",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "The parameters to pass to the tool",
                        },
                    },
                    "required": ["name", "parameters"],
                },
            },
        )

    @property
    def tools(self) -> list[Tool | Toolset]:
        # constant tool list — only tool_search and call_tool
        return [self._search_tool, self._call_tool]

    async def setup(self, *, reload: bool = False) -> Self:
        await super().setup(reload=reload)

        # build a ToolContext from all wrapped tools for call_tool execution
        self._tool_ctx = ToolContext(self._tools)
        return self

    async def _handle_search(self, raw_arguments: dict[str, object]) -> str:
        query = str(raw_arguments.get("query", ""))
        tools = await self._search_tools(query)
        _core.tool_search_loaded(query, len(tools))

        tool_ctx = ToolContext(tools)
        schemas = [_build_tool_schema(tool) for tool in tool_ctx.function_tools.values()]
        return "\n".join(json.dumps(schema) for schema in schemas)

    async def _handle_call(self, ctx: RunContext[Any], raw_arguments: dict[str, object]) -> Any:
        if self._tool_ctx is None:
            raise RuntimeError("toolset not initialized, call setup() first")
        name, parameters = _core.tool_proxy_arguments(
            raw_arguments, list(self._tool_ctx.function_tools)
        )
        fnc_tool = self._tool_ctx.function_tools[name]

        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=fnc_tool,
            json_arguments=parameters,
            call_ctx=ctx,
        )
        return await fnc_tool(*fnc_args, **fnc_kwargs)


def _build_tool_schema(tool: FunctionTool | RawFunctionTool) -> dict[str, Any]:
    """Build a JSON-serializable tool schema with full parameter type info."""
    if isinstance(tool, FunctionTool):
        model = function_arguments_to_pydantic_model(tool)
        return {
            "name": tool.info.name,
            "description": tool.info.description or "",
            "parameters": model.model_json_schema(),
        }

    # RawFunctionTool — use raw_schema directly
    raw = tool.info.raw_schema
    return {
        "name": raw.get("name", tool.id),
        "description": raw.get("description", ""),
        "parameters": raw.get("parameters", {}),
    }
