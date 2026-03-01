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

"""`@async_function_tool` decorator and supporting types for use with `AsyncAgent`.

Async function tools are async generator functions that support long-running operations
with intermediate notifications. The key concept:

- **First yield** = immediate tool return sent to the LLM as the normal
  ``FunctionCallOutput``. This is always returned synchronously as the tool result
  and is **not** affected by ``reply_mode``.
- **Subsequent yields** (second yield onward) = background notifications. These are
  scheduled as agent speech based on the tool's ``reply_mode``.

``reply_mode`` controls how notifications (second yield onward) are scheduled as speech:

- ``"when_idle"`` (default): Wait until the agent finishes speaking, then trigger a new reply.
- ``"interrupt"``: Interrupt the agent's current speech and trigger an immediate reply.
- ``"silent"``: Update the chat context without triggering a reply.

Usage::

    from async_agent import AsyncAgent
    from async_function_tool import async_function_tool, notify

    class MyAgent(AsyncAgent):
        def __init__(self):
            super().__init__(instructions="You are helpful.")

        @async_function_tool(reply_mode="when_idle")
        async def search_web(self, query: str):
            \"\"\"Search the web for information.
            Args:
                query: What to search for.
            \"\"\"
            # First yield = immediate return to the LLM
            yield f"Searching for '{query}'..."

            await asyncio.sleep(5)  # simulate slow API

            # Final yield = delivered when agent is idle
            yield f"Found results for '{query}'"

        @async_function_tool(reply_mode="when_idle")
        async def long_task(self, topic: str):
            \"\"\"Run a long task with progress updates.
            Args:
                topic: The task topic.
            \"\"\"
            yield f"Starting task on '{topic}'..."

            for i in range(3):
                await asyncio.sleep(2)
                # Use notify() to override the reply mode per-yield
                yield notify(f"Progress: {i+1}/3", "silent")

            yield f"Task on '{topic}' complete."
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

    When yielded from an ``@async_function_tool`` generator, ``AsyncAgent``
    will deliver this notification to the LLM based on the specified mode.
    """

    value: Any
    mode: ToolReplyMode


def notify(value: Any, mode: ToolReplyMode = "when_idle") -> AgentNotification:
    """Create an ``AgentNotification`` to override the per-yield delivery mode.

    Use this inside an ``@async_function_tool`` generator to deliver a specific yield
    with a different mode than the tool's default ``reply_mode``.

    Args:
        value: The notification payload (string or JSON-serializable object).
        mode: Delivery mode for this specific yield. Defaults to ``"when_idle"``.

    Example::

        @async_function_tool(reply_mode="when_idle")
        async def my_tool(self, query: str):
            yield "Starting..."
            # This update is silent — just updates context, agent won't speak
            yield notify("50% done", "silent")
            # This final yield uses the tool's default "when_idle" mode
            yield "Done!"
    """
    return AgentNotification(value=value, mode=mode)


@dataclass
class AsyncFunctionToolInfo:
    name: str
    description: str | None
    reply_mode: ToolReplyMode


_P = ParamSpec("_P")
_R = TypeVar("_R")


class AsyncFunctionTool:
    """Wrapper produced by the ``@async_function_tool`` decorator.

    This object acts as a descriptor — when defined as a class attribute on an
    ``AsyncAgent`` subclass, accessing it on an instance automatically binds ``self``.
    ``AsyncAgent`` discovers these at construction time and converts each into a
    regular ``FunctionTool`` that the framework can call.

    You should not instantiate this directly; use the ``@async_function_tool`` decorator.
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
    """Decorator that marks an async generator method as an async function tool.

    The decorated function must be an async generator (use ``yield``, not ``return``).
    It can be used bare or with arguments::

        @async_function_tool
        async def my_tool(self, query: str):
            yield "immediate result"       # first yield = tool return (not affected by reply_mode)
            yield "background result"      # second yield onward = notification (scheduled by reply_mode)

        @async_function_tool(reply_mode="interrupt")
        async def urgent_tool(self, item: str):
            yield "processing..."          # immediate tool return
            yield "done!"                  # interrupts agent speech

    Args:
        f: The function to decorate (when used without parentheses).
        name: Override the tool name (defaults to the function name).
        description: Override the description (defaults to the docstring).
        reply_mode: How notifications (second yield onward) are scheduled as speech.
            The first yield is always returned as the immediate tool result and is
            not affected by this setting.

            - ``"when_idle"`` (default) — wait for agent to finish speaking.
            - ``"interrupt"`` — interrupt agent speech immediately.
            - ``"silent"`` — update context without triggering a reply.
    """

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
    """Discover ``@async_function_tool``-decorated methods on a class or instance."""
    tools: list[AsyncFunctionTool] = []
    for _, member in inspect.getmembers(obj):
        if isinstance(member, AsyncFunctionTool):
            tools.append(member)
    return tools
