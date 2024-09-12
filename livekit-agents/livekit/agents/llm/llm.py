from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .. import utils
from . import function_context
from .chat_context import ChatContext, ChatRole


@dataclass
class ChoiceDelta:
    role: ChatRole
    content: str | None = None
    tool_calls: list[function_context.FunctionCallInfo] | None = None


@dataclass
class Choice:
    delta: ChoiceDelta
    index: int = 0


@dataclass
class ChatChunk:
    choices: list[Choice] = field(default_factory=list)


class LLM(abc.ABC):
    @abc.abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: function_context.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> "LLMStream": ...


class LLMStream(abc.ABC):
    def __init__(
        self, *, chat_ctx: ChatContext, fnc_ctx: function_context.FunctionContext | None
    ) -> None:
        self._function_calls_info: list[function_context.FunctionCallInfo] = []
        self._tasks = set[asyncio.Task[Any]]()
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx

    @property
    def function_calls(self) -> list[function_context.FunctionCallInfo]:
        """List of called functions from this stream."""
        return self._function_calls_info

    @property
    def chat_ctx(self) -> ChatContext:
        """The initial chat context of this stream."""
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> function_context.FunctionContext | None:
        """The function context of this stream."""
        return self._fnc_ctx

    def execute_functions(self) -> list[function_context.CalledFunction]:
        """Execute all functions concurrently of this stream."""
        called_functions: list[function_context.CalledFunction] = []
        for fnc_info in self._function_calls_info:
            called_fnc = fnc_info.execute()
            self._tasks.add(called_fnc.task)
            called_fnc.task.add_done_callback(self._tasks.remove)
            called_functions.append(called_fnc)

        return called_functions

    async def aclose(self) -> None:
        await utils.aio.gracefully_cancel(*self._tasks)

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...
