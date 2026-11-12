from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, Self, cast

from .. import _core
from .._generated import public_beta as _public
from .._generated.public_beta import *  # noqa: F403
from .._generated.workflow_prompts import (
    CALL_TOOL_DESCRIPTION,
    END_CALL_DESCRIPTION,
    TOOL_PROXY_SEARCH_DESCRIPTION,
    TOOL_SEARCH_DESCRIPTION,
    TOOL_SEARCH_QUERY_DESCRIPTION,
)
from ..job import get_job_context
from ..llm import ToolFlag, Toolset, function_tool
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
    "workflows",
    "EndCallTool",
    "send_dtmf_events",
    "BM25SearchStrategy",
    "KeywordSearchStrategy",
    "SearchStrategy",
    "ToolProxyToolset",
    "ToolSearchToolset",
]
__all__ += _public.__all__


class EndCallTool(Toolset):
    def __init__(
        self,
        *,
        extra_description: str = "",
        delete_room: bool = True,
        end_instructions: str | None = "say goodbye to the user",
        ignore_on_enter: bool = False,
        on_tool_called: Callable[[ToolCalledEvent], Awaitable[None]] | None = None,
        on_tool_completed: Callable[[ToolCompletedEvent], Awaitable[None]] | None = None,
    ):
        """Lets the agent end the call after its goodbye.

        Args:
            extra_description: Appended to the tool's description.
            delete_room: Delete the room when the call ends, disconnecting every remote
                participant, SIP callers included.
            end_instructions: The tool output the LLM words its goodbye from.
            ignore_on_enter: Hide the tool during ``on_enter`` so the model cannot end the call
                while greeting.
            on_tool_called: Called when the tool is called.
            on_tool_completed: Called when the tool completes; may change its output.
        """
        end_call_tool = function_tool(
            self._end_call,
            name="end_call",
            description=f"{END_CALL_DESCRIPTION}\n{extra_description}",
            flags=ToolFlag.IGNORE_ON_ENTER if ignore_on_enter else ToolFlag.NONE,
        )
        super().__init__(id="end_call", tools=[end_call_tool])
        self._delete_room = delete_room
        self._end_instructions = end_instructions
        self._on_tool_called = on_tool_called
        self._on_tool_completed = on_tool_completed

    async def _end_call(self, ctx: RunContext) -> Any | None:
        logger.debug("end_call tool called")
        # the session closes once this speech (or a realtime model's reply to the result) is done
        session = ctx.live_session
        if session is None:
            raise RuntimeError("end_call ends a call; this tool runs outside one")
        ctx.speech_handle._native.end_call()
        session.once("close", self._on_session_close)

        if self._on_tool_called:
            await self._on_tool_called(ToolCalledEvent(ctx=ctx, arguments={}))

        completed_ev = ToolCompletedEvent(ctx=ctx, output=self._end_instructions)
        if self._on_tool_completed:
            await self._on_tool_completed(completed_ev)

        return completed_ev.output

    def _on_session_close(self, ev: CloseEvent) -> None:
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
    return await _core.send_dtmf_events(_core._core_create_room_session(room), events)


class SearchStrategy(Protocol):
    def build_index(self, items: list[SearchItem]) -> None | Awaitable[None]: ...
    def search(
        self, query: str, items: list[SearchItem], max_results: int
    ) -> list[SearchItem] | Awaitable[list[SearchItem]]: ...
    def cleanup(self) -> None | Awaitable[None]: ...


class ToolSearchToolset(Toolset):
    """Exposes a single ``tool_search`` tool; the tools it finds are loaded into the context.

    Each tool is indexed as its own ``SearchItem``. A matched tool that belongs to a Toolset
    loads the whole Toolset.
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

        self._search_tool = function_tool(
            self._handle_search,
            raw_schema=_core.tool_search_schema(
                search_description or TOOL_SEARCH_DESCRIPTION,
                query_description or TOOL_SEARCH_QUERY_DESCRIPTION,
            ),
        )

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [self._search_tool, *self._loaded_tools]

    def _toolset_dependencies(self) -> list[Toolset]:
        return [tool for tool in self._tools if isinstance(tool, Toolset)]

    async def setup(self) -> Self:
        await super().setup()
        self._search_items = []

        def _index_tool(tool: Tool | Toolset, source: Tool | Toolset) -> None:
            if isinstance(tool, Toolset):
                tool_ctx = ToolContext([tool])
                for tool in tool_ctx.flatten():
                    _index_tool(tool, source)
            elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                _, description, schema = tool_schema(tool)
                properties = cast(dict[str, object], schema.get("properties", {}))
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
        self._search_items.clear()
        self._loaded_tools.clear()

        result = self._strategy.cleanup()
        if inspect.isawaitable(result):
            await result


class KeywordSearchStrategy:
    """Keyword search. A name match scores 3, a description match 2, and each parameter name or
    description match 1."""

    def build_index(self, items: list[SearchItem]) -> None:
        for item in items:
            item.index_data = _core.SearchDocument(item)

    def search(self, query: str, items: list[SearchItem], max_results: int) -> list[SearchItem]:
        docs = [item.index_data or _core.SearchDocument(item) for item in items]
        return [items[i] for i in _core.keyword_search(query, docs, max_results)]

    def cleanup(self) -> None:
        pass


class BM25SearchStrategy:
    """BM25 search, better than keyword matching for large tool collections: common terms
    weigh less and rare, specific matches more. An item's name weighs 3x, its description 2x
    and its parameters 1x.

    Args:
        k1: Term frequency saturation; higher gives repeated terms more weight.
        b: Length normalization (0-1); higher penalizes longer documents more.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._index = _core.BM25Index(k1, b)

    def build_index(self, items: list[SearchItem]) -> None:
        for item in items:
            item.index_data = _core.SearchDocument(item)
        self._index.build([item.index_data for item in items])

    def search(self, query: str, items: list[SearchItem], max_results: int) -> list[SearchItem]:
        docs = [item.index_data for item in items]
        assert all(d is not None for d in docs), "index data must be built before scoring"
        return [items[i] for i in self._index.search(query, docs, max_results)]

    def cleanup(self) -> None:
        self._index.clear()


class ToolProxyToolset(ToolSearchToolset):
    """Exposes two fixed tools: ``tool_search`` returns tool schemas as text and ``call_tool``
    runs a tool by name. The constant tool list keeps provider prompt caches warm.
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
            search_description=search_description or TOOL_PROXY_SEARCH_DESCRIPTION,
            query_description=query_description,
        )
        self._tool_ctx: ToolContext | None = None

        self._call_tool = function_tool(
            self._handle_call,
            raw_schema=_core.call_tool_schema(call_description or CALL_TOOL_DESCRIPTION),
        )

    @property
    def tools(self) -> list[Tool | Toolset]:
        return [self._search_tool, self._call_tool]

    async def setup(self) -> Self:
        await super().setup()
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
        call = _core.tool_proxy_arguments(raw_arguments, list(self._tool_ctx.function_tools))
        fnc_tool = self._tool_ctx.function_tools[call.name]

        fnc_args, fnc_kwargs = prepare_function_arguments(
            fnc=fnc_tool,
            json_arguments=cast(str | dict[str, object], call.parameters),
            call_ctx=ctx,
        )
        result = fnc_tool(*fnc_args, **fnc_kwargs)
        return await result if inspect.isawaitable(result) else result
