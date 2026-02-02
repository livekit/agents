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

import functools
import inspect
import itertools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, Union, overload

from typing_extensions import NotRequired, ParamSpec, Required, Self, TypedDict, TypeGuard

from ..log import logger
from . import _provider_format

if TYPE_CHECKING:
    from ..voice.events import RunContext


class Tool(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...


class ProviderTool(Tool):
    def __init__(self, *, id: str) -> None:
        self._id = id

    @property
    def id(self) -> str:
        return self._id


class Toolset(ABC):
    @dataclass
    class ToolCalledEvent:
        ctx: RunContext
        arguments: dict[str, Any]

    @dataclass
    class ToolCompletedEvent:
        ctx: RunContext
        output: Any | Exception | None

    def __init__(self, *, id: str) -> None:
        self._id = id

    @property
    def id(self) -> str:
        return self._id

    @property
    @abstractmethod
    def tools(self) -> list[Tool]: ...


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


class ToolFlag(Flag):
    NONE = 0
    IGNORE_ON_ENTER = auto()
    DURABLE = auto()


@dataclass
class FunctionToolInfo:
    name: str
    description: str | None
    flags: ToolFlag


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
class RawFunctionToolInfo:
    name: str
    raw_schema: dict[str, Any]
    flags: ToolFlag


_InfoT = TypeVar("_InfoT", FunctionToolInfo, RawFunctionToolInfo)
_P = ParamSpec("_P")
_R = TypeVar("_R", bound=Awaitable[Any])


class _BaseFunctionTool(Tool, Generic[_InfoT, _P, _R]):
    """Base class for function tool wrappers with descriptor support."""

    def __init__(self, func: Callable[_P, _R], info: _InfoT, instance: Any = None) -> None:
        if info.flags & ToolFlag.DURABLE and instance is None:
            # only wrap if instance is none, to avoid wrapping the same function multiple times
            from livekit.durable import durable

            func = durable(func)

        functools.update_wrapper(self, func)
        self._func = func
        self._info: _InfoT = info
        self._instance = instance

    @property
    def id(self) -> str:
        return self._info.name

    @property
    def info(self) -> _InfoT:
        return self._info

    def __get__(self, obj: Any, objtype: type | None = None) -> Self:
        if obj is None:
            return self

        # bind the tool to an instance
        bound_tool = self.__class__(self._func, self._info, instance=obj)
        sig = inspect.signature(self._func)
        # skip the instance parameter (e.g. usually the 'self')
        params = list(sig.parameters.values())[1:]
        bound_tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        return bound_tool

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        if self._instance is not None:
            return self._func(self._instance, *args, **kwargs)
        return self._func(*args, **kwargs)


class FunctionTool(_BaseFunctionTool[FunctionToolInfo, _P, _R]):
    """Wrapper for a function decorated with @function_tool"""

    def __init__(
        self, func: Callable[_P, _R], info: FunctionToolInfo, instance: Any = None
    ) -> None:
        super().__init__(func, info, instance)
        setattr(self, "__livekit_tool_info", self._info)


class RawFunctionTool(_BaseFunctionTool[RawFunctionToolInfo, _P, _R]):
    """Wrapper for a function decorated with @function_tool(raw_schema=...)"""

    def __init__(
        self, func: Callable[_P, _R], info: RawFunctionToolInfo, instance: Any = None
    ) -> None:
        super().__init__(func, info, instance)
        setattr(self, "__livekit_raw_tool_info", self._info)


@overload
def function_tool(
    f: Callable[_P, _R],
    *,
    raw_schema: RawFunctionDescription | dict[str, Any],
    flags: ToolFlag = ToolFlag.NONE,
) -> RawFunctionTool[_P, _R]: ...


@overload
def function_tool(
    f: None = None,
    *,
    raw_schema: RawFunctionDescription | dict[str, Any],
    flags: ToolFlag = ToolFlag.NONE,
) -> Callable[[Callable[_P, _R]], RawFunctionTool[_P, _R]]: ...


@overload
def function_tool(
    f: Callable[_P, _R],
    *,
    name: str | None = None,
    description: str | None = None,
    flags: ToolFlag = ToolFlag.NONE,
) -> FunctionTool[_P, _R]: ...


@overload
def function_tool(
    f: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    flags: ToolFlag = ToolFlag.NONE,
) -> Callable[[Callable[_P, _R]], FunctionTool[_P, _R]]: ...


def function_tool(
    f: Callable[_P, _R] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    raw_schema: RawFunctionDescription | dict[str, Any] | None = None,
    flags: ToolFlag = ToolFlag.NONE,
) -> (
    FunctionTool[_P, _R]
    | RawFunctionTool[_P, _R]
    | Callable[[Callable[_P, _R]], FunctionTool[_P, _R]]
    | Callable[[Callable[_P, _R]], RawFunctionTool[_P, _R]]
):
    def deco_raw(
        func: Callable[_P, _R],
    ) -> RawFunctionTool[_P, _R]:
        assert raw_schema is not None

        if not raw_schema.get("name"):
            raise ValueError("raw function name cannot be empty")

        if "parameters" not in raw_schema:
            # support empty parameters
            raise ValueError("raw function description must contain a parameters key")

        info = RawFunctionToolInfo(
            name=raw_schema["name"],
            raw_schema={**raw_schema},
            flags=flags,
        )
        return RawFunctionTool(func, info)

    def deco_func(func: Callable[_P, _R]) -> FunctionTool[_P, _R]:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        info = FunctionToolInfo(
            name=name or func.__name__,
            description=description or docstring.description,
            flags=flags,
        )
        return FunctionTool(func, info)

    if f is not None:
        return deco_raw(f) if raw_schema is not None else deco_func(f)

    return deco_raw if raw_schema is not None else deco_func


def is_function_tool(f: Any) -> TypeGuard[FunctionTool]:
    # TODO(long): for backward compatibility, deprecate in future versions?
    return isinstance(f, FunctionTool)


def get_function_info(f: FunctionTool) -> FunctionToolInfo:
    return f.info


def is_raw_function_tool(f: Any) -> TypeGuard[RawFunctionTool]:
    return isinstance(f, RawFunctionTool)


def get_raw_function_info(f: RawFunctionTool) -> RawFunctionToolInfo:
    return f.info


def _resolve_wrapped_tool(tool: Any) -> FunctionTool | RawFunctionTool | None:
    """Convert a wrapped tool to a FunctionTool or RawFunctionTool with a warning."""
    if not callable(tool):
        return None

    if isinstance(tool, (FunctionTool, RawFunctionTool)):
        return tool

    resolved_tool: FunctionTool | RawFunctionTool | None = None
    if (
        hasattr(tool, "__wrapped__")  # automatically added by functools.wraps
        and isinstance(tool.__wrapped__, (FunctionTool, RawFunctionTool))
    ):
        wrapped = tool.__wrapped__
        resolved_tool = wrapped.__class__(tool, wrapped.info)  # type: ignore

    elif (info := getattr(tool, "__livekit_tool_info", None)) and isinstance(
        info, FunctionToolInfo
    ):
        resolved_tool = FunctionTool(tool, info)

    elif (info := getattr(tool, "__livekit_raw_tool_info", None)) and isinstance(
        info, RawFunctionToolInfo
    ):
        resolved_tool = RawFunctionTool(tool, info)

    if resolved_tool:
        tool_name = resolved_tool.info.name
        logger.warning(
            f"function tool {tool_name} is wrapped, this may cause unexpected behavior and not be supported in future versions, "
            "please wrap the original function before converting to a function tool.",
            extra={
                "function_tool": tool_name,
            },
        )

    return resolved_tool


def find_function_tools(cls_or_obj: Any) -> list[FunctionTool | RawFunctionTool]:
    methods: list[FunctionTool | RawFunctionTool] = []
    for _, member in inspect.getmembers(cls_or_obj):
        if isinstance(member, (FunctionTool, RawFunctionTool)):
            methods.append(member)
        elif normalized_tool := _resolve_wrapped_tool(member):
            methods.append(normalized_tool)

    return methods


def get_fnc_tool_names(tools: Sequence[Tool | Toolset]) -> list[str]:
    """Get names of all function and raw function tools in the list, unwrapping tool sets."""
    names = []
    for tool in tools:
        if isinstance(tool, (FunctionTool, RawFunctionTool)):
            names.append(tool.info.name)
        elif isinstance(tool, Toolset):
            names.extend(get_fnc_tool_names(tool.tools))

    return names


class ToolContext:
    """Stateless container for a set of AI functions"""

    def __init__(self, tools: Sequence[Tool | Toolset]) -> None:
        self.update_tools(tools)

    @classmethod
    def empty(cls) -> ToolContext:
        return cls([])

    @property
    def function_tools(self) -> dict[str, FunctionTool | RawFunctionTool]:
        """A copy of all function tools in the tool context, including those in tool sets."""
        return self._fnc_tools_map.copy()

    @property
    def provider_tools(self) -> list[ProviderTool]:
        """A copy of all provider tools in the tool context, including those in tool sets."""
        return self._provider_tools

    @property
    def toolsets(self) -> list[Toolset]:
        """A copy of all tool sets in the tool context."""
        return self._tool_sets

    def flatten(self) -> list[Tool]:
        """Flatten the tool context to a list of tools."""
        tools: list[Tool] = []
        tools.extend(list(self._fnc_tools_map.values()))
        tools.extend(self._provider_tools)
        return tools

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolContext):
            return False

        if self._fnc_tools_map.keys() != other._fnc_tools_map.keys():
            return False

        for name in self._fnc_tools_map:
            if self._fnc_tools_map[name] is not other._fnc_tools_map[name]:
                return False

        if len(self._provider_tools) != len(other._provider_tools):
            return False

        self_provider_ids = {id(tool) for tool in self._provider_tools}
        other_provider_ids = {id(tool) for tool in other._provider_tools}
        if self_provider_ids != other_provider_ids:
            return False

        self_tool_set_ids = {id(tool_set) for tool_set in self._tool_sets}
        other_tool_set_ids = {id(tool_set) for tool_set in other._tool_sets}
        if self_tool_set_ids != other_tool_set_ids:
            return False

        return True

    def update_tools(self, tools: Sequence[Tool | Toolset]) -> None:
        self._tools = list(tools)
        self._fnc_tools_map: dict[str, FunctionTool | RawFunctionTool] = {}
        self._provider_tools: list[ProviderTool] = []
        self._tool_sets: list[Toolset] = []

        def add_tool(tool: Tool | Toolset) -> None:
            if isinstance(tool, ProviderTool):
                self._provider_tools.append(tool)

            elif isinstance(tool, (FunctionTool, RawFunctionTool)):
                if tool.info.name in self._fnc_tools_map:
                    raise ValueError(f"duplicate function name: {tool.info.name}")
                self._fnc_tools_map[tool.info.name] = tool

            elif isinstance(tool, Toolset):
                for t in tool.tools:
                    add_tool(t)
                self._tool_sets.append(tool)

            elif normalized_tool := _resolve_wrapped_tool(tool):
                add_tool(normalized_tool)

            elif callable(tool):
                raise ValueError(
                    "Expected an instance of FunctionTool or RawFunctionTool, got a callable object. "
                    "If it's a wrapped tool, please consider wrapping the original function before converting to a function tool."
                )

            else:
                raise ValueError(f"unknown tool type: {type(tool)}")

        for tool in itertools.chain(tools, find_function_tools(self)):
            add_tool(tool)

    def copy(self) -> ToolContext:
        return ToolContext(self._tools.copy())

    @overload
    def parse_function_tools(
        self, format: Literal["openai", "openai.responses"], *, strict: bool = True
    ) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(
        self,
        format: Literal["google"],
        *,
        tool_behavior: _provider_format.google.TOOL_BEHAVIOR | None = None,
    ) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(self, format: Literal["aws"]) -> list[dict[str, Any]]: ...

    @overload
    def parse_function_tools(self, format: Literal["anthropic"]) -> list[dict[str, Any]]: ...

    def parse_function_tools(
        self,
        format: Literal["openai", "google", "aws", "anthropic"] | str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Parse the function tools to a provider-specific schema."""
        if format == "openai":
            return _provider_format.openai.to_fnc_ctx(self, **kwargs)
        elif format == "openai.responses":
            return _provider_format.openai.to_responses_fnc_ctx(self, **kwargs)
        elif format == "google":
            return _provider_format.google.to_fnc_ctx(self, **kwargs)
        elif format == "anthropic":
            return _provider_format.anthropic.to_fnc_ctx(self, **kwargs)
        elif format == "aws":
            return _provider_format.aws.to_fnc_ctx(self, **kwargs)

        raise ValueError(f"Unsupported provider format: {format}")
