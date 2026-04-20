from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from typing_extensions import Self

from ...llm.tool_context import (
    FunctionTool,
    RawFunctionTool,
    Tool,
    ToolContext,
    ToolError,
    Toolset,
    function_tool,
)
from ...llm.utils import function_arguments_to_pydantic_model, prepare_function_arguments
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.events import RunContext
from .tool_search import SearchStrategy, ToolSearchToolset

_DEFAULT_SEARCH_DESCRIPTION = (
    "Search for available tools by describing what you need. "
    "Returns the schemas of matching tools. Use call_tool to invoke them."
)

_DEFAULT_CALL_DESCRIPTION = (
    "Call a tool by name with the given arguments. "
    "Use search_tools to discover available tools and their schemas if it isn't already loaded."
)


class ToolProxyToolset(ToolSearchToolset):
    """Exposes exactly two fixed tools: search_tools and call_tool.

    Unlike ToolSearchToolset which dynamically modifies the tool list,
    ToolProxyToolset keeps a constant tool list. ``search_tools`` returns
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
            search_description=search_description or _DEFAULT_SEARCH_DESCRIPTION,
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
        # constant tool list — only search_tools and call_tool
        return [self._search_tool, self._call_tool]

    async def setup(self, *, reload: bool = False) -> Self:
        await super().setup(reload=reload)

        # build a ToolContext from all wrapped tools for call_tool execution
        self._tool_ctx = ToolContext(self._tools)
        return self

    async def _handle_search(self, raw_arguments: dict[str, object]) -> str:
        query = str(raw_arguments.get("query", ""))
        if not query:
            raise ToolError("query cannot be empty")

        tools = await self._search_tools(query)
        if not tools:
            raise ToolError(f"No tools found matching '{query}'.")

        tool_ctx = ToolContext(tools)
        schemas = [_build_tool_schema(tool) for tool in tool_ctx.function_tools.values()]
        return "\n".join(json.dumps(schema) for schema in schemas)

    async def _handle_call(self, ctx: RunContext[Any], raw_arguments: dict[str, object]) -> Any:
        name = str(raw_arguments.get("name", ""))
        parameters = raw_arguments.get("parameters")

        if not name:
            raise ToolError("tool name cannot be empty")

        if parameters is None:
            raise ToolError("parameters is required")

        if self._tool_ctx is None:
            raise RuntimeError("toolset not initialized, call setup() first")

        fnc_tool = self._tool_ctx.get_function_tool(name)
        if fnc_tool is None:
            raise ToolError(f"unknown tool '{name}', use search_tools to discover available tools")

        try:
            json_args = json.dumps(parameters) if isinstance(parameters, dict) else str(parameters)
            fnc_args, fnc_kwargs = prepare_function_arguments(
                fnc=fnc_tool,
                json_arguments=json_args,
                call_ctx=ctx,
            )
        except ValidationError as e:
            raise ToolError(
                f"invalid parameters for tool '{name}': {e.json(include_url=False)}"
            ) from e
        except ToolError:
            raise
        except Exception as e:
            logger.exception(
                f"error parsing arguments for tool '{name}'",
                extra={"tool": name, "arguments": parameters},
            )
            raise ToolError(f"error calling '{name}': {e}") from e

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
