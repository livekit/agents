# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

from typing_extensions import NotRequired, Required, TypedDict, TypeGuard


# Used by ToolChoice
class Function(TypedDict, total=False):
    name: Required[str]


class NamedToolChoice(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[Function]


ToolChoice = Union[NamedToolChoice, Literal["auto", "required", "none"]]


class ToolError(Exception):
    def __init__(self, message: str) -> None:
        """
        Exception raised within AI functions.

        This exception should be raised by users when an error occurs
        in the context of AI operations. The provided message will be
        visible to the LLM, allowing it to understand the context of
        the error during FunctionOutput generation.
        """
        super().__init__(message)
        self._message = message

    @property
    def message(self) -> str:
        return self._message


class StopResponse(Exception):
    def __init__(self) -> None:
        """
        Exception raised within AI functions.

        This exception can be raised by the user to indicate that
        the agent should not generate a response for the current
        function call.
        """
        super().__init__()


@dataclass
class _FunctionToolInfo:
    name: str
    description: str | None


@runtime_checkable
class FunctionTool(Protocol):
    __livekit_tool_info: _FunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class RawFunctionDescription(TypedDict):
    """
    Represents the raw function schema format used in LLM function calling APIs.

    This structure directly maps to OpenAI's function definition format as documented at:
    https://platform.openai.com/docs/guides/function-calling?api-mode=responses

    It is also compatible with other LLM providers that support raw JSON Schema-based
    function definitions.
    """

    name: str
    description: NotRequired[str | None]
    parameters: dict[str, object]


@dataclass
class _RawFunctionToolInfo:
    name: str
    raw_schema: dict[str, Any]


@runtime_checkable
class RawFunctionTool(Protocol):
    __livekit_raw_tool_info: _RawFunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])
Raw_F = TypeVar("Raw_F", bound=Callable[..., Awaitable[Any]])


@overload
def function_tool(
    f: Raw_F, *, raw_schema: RawFunctionDescription | dict[str, Any]
) -> RawFunctionTool: ...


@overload
def function_tool(
    f: None = None, *, raw_schema: RawFunctionDescription | dict[str, Any]
) -> Callable[[Raw_F], RawFunctionTool]: ...


@overload
def function_tool(
    f: F, *, name: str | None = None, description: str | None = None
) -> FunctionTool: ...


@overload
def function_tool(
    f: None = None, *, name: str | None = None, description: str | None = None
) -> Callable[[F], FunctionTool]: ...


def function_tool(
    f: F | Raw_F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
) -> (
    FunctionTool
    | RawFunctionTool
    | Callable[[F], FunctionTool]
    | Callable[[Raw_F], RawFunctionTool]
):
    def deco_raw(func: Raw_F) -> RawFunctionTool:
        assert raw_schema is not None

        if not raw_schema.get("name"):
            raise ValueError("raw function name cannot be empty")

        if "parameters" not in raw_schema:
            # support empty parameters
            raise ValueError("raw function description must contain a parameters key")

        info = _RawFunctionToolInfo(raw_schema={**raw_schema}, name=raw_schema["name"])
        setattr(func, "__livekit_raw_tool_info", info)
        return cast(RawFunctionTool, func)

    def deco_func(func: F) -> FunctionTool:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        info = _FunctionToolInfo(
            name=name or func.__name__,
            description=description or docstring.description,
        )
        setattr(func, "__livekit_tool_info", info)
        return cast(FunctionTool, func)

    if f is not None:
        return deco_raw(cast(Raw_F, f)) if raw_schema is not None else deco_func(cast(F, f))

    return deco_raw if raw_schema is not None else deco_func


def is_function_tool(f: Callable[..., Any]) -> TypeGuard[FunctionTool]:
    return hasattr(f, "__livekit_tool_info")


def get_function_info(f: FunctionTool) -> _FunctionToolInfo:
    return cast(_FunctionToolInfo, getattr(f, "__livekit_tool_info"))


def is_raw_function_tool(f: Callable[..., Any]) -> TypeGuard[RawFunctionTool]:
    return hasattr(f, "__livekit_raw_tool_info")


def get_raw_function_info(f: RawFunctionTool) -> _RawFunctionToolInfo:
    return cast(_RawFunctionToolInfo, getattr(f, "__livekit_raw_tool_info"))


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool | RawFunctionTool]:
    methods: list[FunctionTool | RawFunctionTool] = []
    for _, member in inspect.getmembers(cls_or_obj):
        if is_function_tool(member) or is_raw_function_tool(member):
            methods.append(member)
    return methods


class ToolContext:
    """Stateless container for a set of AI functions"""

    def __init__(self, tools: list[FunctionTool | RawFunctionTool]) -> None:
        self.update_tools(tools)

    @classmethod
    def empty(cls) -> ToolContext:
        return cls([])

    @property
    def function_tools(self) -> dict[str, FunctionTool | RawFunctionTool]:
        return self._tools_map.copy()

    def update_tools(self, tools: list[FunctionTool | RawFunctionTool]) -> None:
        self._tools = tools.copy()

        for method in find_function_tools(self):
            tools.append(method)

        self._tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        info: _FunctionToolInfo | _RawFunctionToolInfo
        for tool in tools:
            if is_raw_function_tool(tool):
                info = get_raw_function_info(tool)
            elif is_function_tool(tool):
                info = get_function_info(tool)
            else:
                # TODO(theomonnom): MCP servers & other tools
                raise ValueError(f"unknown tool type: {type(tool)}")

            if info.name in self._tools_map:
                raise ValueError(f"duplicate function name: {info.name}")

            self._tools_map[info.name] = tool

    def copy(self) -> ToolContext:
        return ToolContext(self._tools.copy())
