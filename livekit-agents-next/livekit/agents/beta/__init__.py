"""Prebuilt tools and toolsets. ``EndCallTool`` ends the call after the agent's goodbye and
``send_dtmf_events`` sends keypad (DTMF) tones; the core decides when the call closes and sends the
tones. ``ToolSearchToolset`` loads the tools a search finds and ``ToolProxyToolset`` keeps two fixed
tools (search and call), keeping a large tool collection out of the model's context: the ranking,
the arguments' checks and the replies to the model are the core's, and a custom ``SearchStrategy``
(an embedding search, say) can stand in for the built-in ones. ``workflows`` holds the building
blocks for capture flows: result records, capture checks, DTMF reception and warm transfer."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from typing_extensions import Self

from .. import _core
from ..job import get_job_context
from ..llm import ToolFlag, Toolset, function_tool
from ..llm.chat_context import Instructions
from ..llm.tool_context import (
    FunctionTool,
    ProviderTool,
    RawFunctionTool,
    Tool,
    ToolContext,
)
from ..llm.utils import prepare_function_arguments, tool_schema
from ..log import logger
from ..types import NOT_GIVEN, NotGivenOr
from ..voice.events import CloseEvent, RunContext
from . import workflows
from .workflows import DtmfEvent

__all__ = [
    "Instructions",
    "workflows",
    "EndCallTool",
    "send_dtmf_events",
    "BM25SearchStrategy",
    "KeywordSearchStrategy",
    "SearchItem",
    "SearchStrategy",
    "ToolProxyToolset",
    "ToolSearchToolset",
]


class EndCallTool(Toolset):
    def __init__(
        self,
        *,
        extra_description: str = "",
        delete_room: bool = True,
        end_instructions: str | None = "say goodbye to the user",
        ignore_on_enter: bool = False,
        on_tool_called: Callable[[Toolset.ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[Toolset.ToolCompletedEvent], Awaitable[None]] | None = None,
    ):
        """
        This tool allows the agent to end the call and disconnect from the room.

        Args:
            extra_description: Additional description to add to the end call tool.
            delete_room: Whether to delete the room when the user ends the call. deleting the room disconnects all remote users, including SIP callers.
            end_instructions: Tool output to the LLM for generating the tool response.
            ignore_on_enter: Hide the tool during ``on_enter`` so the model can't end the call while greeting.
            on_tool_called: Callback to call when the tool is called.
            on_tool_completed: Callback to call when the tool is completed.
        """
        end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{_core.END_CALL_DESCRIPTION}\n{extra_description}",
            flags=ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE,
        )
        super().__init__(id="end_call", tools=[end_call_tool])
        self._delete_room = delete_room
        self._end_instructions = end_instructions
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

    async def _end_call(self, ctx: RunContext) -> Any | None:
        logger.debug("end_call tool called")
        # the core closes the session once this speech (or a realtime model's own reply to the
        # tool result) is done
        session = ctx.live_session
        if session is None:
            raise RuntimeError("end_call ends a call; this tool runs outside one")
        session._core.end_call(ctx.speech_handle._core)
        session.once("close", self._on_session_close)

        if self._on_tool_called:
            await self._on_tool_called(Toolset.ToolCalledEvent(ctx=ctx, arguments={}))

        completed_ev = Toolset.ToolCompletedEvent(ctx=ctx, output=self._end_instructions)
        if self._on_tool_completed:
            await self._on_tool_completed(completed_ev)

        return completed_ev.output

    def _on_session_close(self, ev: CloseEvent) -> None:
        """The call is over: the job ends, deleting the room on its way out when asked."""
        job_ctx = get_job_context(required=False)
        if job_ctx is None:
            return
        if self._delete_room:

            async def _delete_room(reason: str) -> None:
                logger.info("deleting the room because the user ended the call")
                await job_ctx.delete_room()

            job_ctx.add_shutdown_callback(_delete_room)
        job_ctx.shutdown(reason=ev.reason.value)


@function_tool
async def send_dtmf_events(ctx: RunContext, events: list[DtmfEvent]) -> str:
    """
    Send a list of DTMF events to the telephony provider.

    Call when:
    - User wants to send DTMF events
    """
    session = ctx.live_session
    if session is None:
        raise RuntimeError("DTMF tones are sent on a call; this tool runs outside one")
    room = session._room or get_job_context().room
    return await _core.send_dtmf_events(room, [event.value for event in events])


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
                    _, description, schema = tool_schema(tool)
                    properties = schema.get("properties", {})
                    self._search_items.append(
                        SearchItem(
                            name=tool.id,
                            description=description,
                            parameters={
                                n: p.get("description", "") if isinstance(p, dict) else ""
                                for n, p in properties.items()
                            },
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

        schemas = []
        for tool in ToolContext(tools).function_tools.values():
            name, description, parameters = tool_schema(tool)
            schemas.append({"name": name, "description": description, "parameters": parameters})
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
