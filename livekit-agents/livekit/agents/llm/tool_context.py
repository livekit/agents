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

from typing_extensions import Required, TypedDict, TypeGuard


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
    name: str | None
    description: str | None


class _RawFunctionToolInfo:
    raw: dict[str, Any]


@runtime_checkable
class FunctionTool(Protocol):
    __livekit_agents_function_tool: _FunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class RawFunctionTool(Protocol):
    __livekit_agents_raw_function_tool: _RawFunctionToolInfo

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


@overload
def function_tool(
    f: F, *, name: str | None = None, description: str | None = None
) -> FunctionTool: ...


@overload
def function_tool(
    f: None = None, *, name: str | None = None, description: str | None = None
) -> Callable[[F], FunctionTool]: ...


@overload
def function_tool(f: F, *, raw: dict) -> RawFunctionTool: ...


@overload
def function_tool(f: None = None, *, raw: dict) -> Callable[[F], RawFunctionTool]: ...


def function_tool(
    f: F | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw: dict[str, Any] | None = None,
) -> FunctionTool | RawFunctionTool | Callable[[F], FunctionTool | RawFunctionTool]:
    def deco(func: F) -> FunctionTool:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        if raw is not None:
            info = _FunctionToolInfo(
                name=name or func.__name__,
                description=description or docstring.description,
            )
        setattr(func, "__livekit_agents_function_tool", info)
        return cast(FunctionTool, func)

    if f is not None:
        return deco(f)

    return deco


def is_function_tool(f: Callable) -> TypeGuard[FunctionTool]:
    return hasattr(f, "__livekit_agents_function_tool")


def get_function_info(f: FunctionTool) -> _FunctionToolInfo:
    return getattr(f, "__livekit_agents_function_tool")


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool]:
    methods: list[FunctionTool] = []
    for _, member in inspect.getmembers(cls_or_obj):
        if is_function_tool(member):
            methods.append(member)
    return methods


class ToolContext:
    """Stateless container for a set of AI functions"""

    def __init__(self, tools: list[FunctionTool]) -> None:
        self.update_tools(tools)

    @classmethod
    def empty(cls) -> ToolContext:
        return cls([])

    @property
    def function_tools(self) -> dict[str, FunctionTool]:
        return self._tools_map.copy()

    def update_tools(self, tools: list[FunctionTool]) -> None:
        self._tools = tools

        for method in find_function_tools(self):
            tools.append(method)

        self._tools_map = {}
        for tool in tools:
            info = get_function_info(tool)
            if info.name in self._tools_map:
                raise ValueError(f"duplicate function name: {info.name}")

            self._tools_map[info.name] = tool

    def copy(self) -> ToolContext:
        return ToolContext(self._tools.copy())
