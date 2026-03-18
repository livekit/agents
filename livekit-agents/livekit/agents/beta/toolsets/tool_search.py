from __future__ import annotations

import asyncio
import inspect
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Protocol

from typing_extensions import Self

from ...llm.tool_context import (
    FunctionTool,
    ProviderTool,
    RawFunctionTool,
    Tool,
    ToolContext,
    Toolset,
    function_tool,
)
from ...types import NOT_GIVEN, NotGivenOr


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


class KeywordSearchStrategy:
    """Zero-dependency keyword search using regex matching.

    Scoring: name match = 3pts, description match = 2pts, parameter name/desc match = 1pt each.
    """

    def build_index(self, items: list[SearchItem]) -> None:
        for item in items:
            item.index_data = {
                "name": item.name.lower(),
                "description": item.description.lower(),
                "parameters": " ".join(f"{k} {v}" for k, v in item.parameters.items()).lower(),
            }

    def search(self, query: str, items: list[SearchItem], max_results: int) -> list[SearchItem]:
        keywords = list(set(query.lower().split()))
        if not keywords:
            return []

        scored: list[tuple[float, SearchItem]] = []
        for item in items:
            s = self._score(item, keywords)
            if s > 0:
                scored.append((s, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max_results]]

    def _score(self, item: SearchItem, keywords: list[str]) -> float:
        score = 0.0
        idx = item.index_data
        if idx is None:
            self.build_index([item])
            idx = item.index_data

        for kw in keywords:
            try:
                pattern = re.compile(kw)
            except re.error:
                pattern = re.compile(re.escape(kw))

            if pattern.search(idx["name"]):
                score += 3.0
            if pattern.search(idx["description"]):
                score += 2.0
            if pattern.search(idx["parameters"]):
                score += 1.0

        return score


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
        self._strategy = search_strategy or KeywordSearchStrategy()
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

            # Setup wrapped toolsets
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
        if not query:
            return "Please provide a search query."

        results = self._strategy.search(query, self._search_items, self._max_results)
        if inspect.isawaitable(results):
            results = await results

        if not results:
            return f"No tools found matching '{query}'."

        self._loaded_tools = list({result.source for result in results})
        return "Tools loaded successfully."

    async def aclose(self) -> None:
        await super().aclose()
        toolsets = [t for t in self._tools if isinstance(t, Toolset)]
        if toolsets:
            await asyncio.gather(*(ts.aclose() for ts in toolsets))


def _get_tool_description(tool: FunctionTool | RawFunctionTool) -> str:
    if isinstance(tool, FunctionTool):
        return tool.info.description or ""
    return str(tool.info.raw_schema.get("description", ""))


def _get_tool_params(tool: FunctionTool | RawFunctionTool) -> dict[str, str]:
    if isinstance(tool, FunctionTool):
        from ...llm.utils import function_arguments_to_pydantic_model

        model = function_arguments_to_pydantic_model(tool)
        return {name: field.description or "" for name, field in model.model_fields.items()}

    props = tool.info.raw_schema.get("parameters", {}).get("properties", {})
    return {
        name: prop.get("description", "") if isinstance(prop, dict) else ""
        for name, prop in props.items()
    }
