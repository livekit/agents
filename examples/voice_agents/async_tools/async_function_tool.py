# Copyright 2025 LiveKit, Inc.
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

"""@async_function_tool decorator and supporting types.

Defines async generator-based tools that produce an immediate result (first yield)
and background notifications (subsequent yields). Used with AsyncAgent.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

from typing_extensions import ParamSpec

ToolReplyMode = Literal["when_idle", "interrupt", "silent"]


@dataclass
class AgentNotification:
    """Wraps an intermediate yield value with a delivery mode.

    When yielded from an @async_function_tool generator, the AsyncAgent
    will deliver this notification to the LLM based on the specified mode.
    """

    value: Any
    mode: ToolReplyMode


def notify(value: Any, mode: ToolReplyMode = "when_idle") -> AgentNotification:
    """Helper to create an AgentNotification for use inside @async_function_tool generators."""
    return AgentNotification(value=value, mode=mode)


@dataclass
class AsyncFunctionToolInfo:
    name: str
    description: str | None
    reply_mode: ToolReplyMode


_P = ParamSpec("_P")
_R = TypeVar("_R")


class AsyncFunctionTool:
    """Wrapper for an async generator function decorated with @async_function_tool.

    Stores the original function and its metadata. AsyncAgent discovers these
    and converts them into regular FunctionTools at construction time.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        info: AsyncFunctionToolInfo,
        instance: Any = None,
    ) -> None:
        self._func = func
        self._info = info
        self._instance = instance

    @property
    def info(self) -> AsyncFunctionToolInfo:
        return self._info

    def __get__(self, obj: Any, objtype: type | None = None) -> AsyncFunctionTool:
        if obj is None:
            return self
        # Bind the tool to an instance (descriptor protocol for class methods)
        bound = AsyncFunctionTool(self._func, self._info, instance=obj)
        sig = inspect.signature(self._func)
        params = list(sig.parameters.values())[1:]  # skip 'self'
        bound.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
        return bound

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._instance is not None:
            return self._func(self._instance, *args, **kwargs)
        return self._func(*args, **kwargs)


# --- Overloads for @async_function_tool ---


@overload
def async_function_tool(
    f: Callable[_P, _R],
    *,
    name: str | None = None,
    description: str | None = None,
    reply_mode: ToolReplyMode = "when_idle",
) -> AsyncFunctionTool: ...


@overload
def async_function_tool(
    f: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    reply_mode: ToolReplyMode = "when_idle",
) -> Callable[[Callable[_P, _R]], AsyncFunctionTool]: ...


def async_function_tool(
    f: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    reply_mode: ToolReplyMode = "when_idle",
) -> AsyncFunctionTool | Callable[[Callable[..., Any]], AsyncFunctionTool]:
    def deco(func: Callable[..., Any]) -> AsyncFunctionTool:
        from docstring_parser import parse_from_object

        docstring = parse_from_object(func)
        info = AsyncFunctionToolInfo(
            name=name or func.__name__,
            description=description or docstring.description,
            reply_mode=reply_mode,
        )
        return AsyncFunctionTool(func, info)

    if f is not None:
        return deco(f)
    return deco


def find_async_function_tools(obj: Any) -> list[AsyncFunctionTool]:
    """Discover @async_function_tool-decorated methods on a class or instance."""
    tools: list[AsyncFunctionTool] = []
    for _, member in inspect.getmembers(obj):
        if isinstance(member, AsyncFunctionTool):
            tools.append(member)
    return tools
