from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncIterator

from . import function_context
from .chat_context import ChatContext, ChatRole


@dataclass
class ChoiceDelta:
    role: ChatRole
    content: str | None = None
    tool_calls: list[function_context.CalledFunction] | None = None


@dataclass
class Choice:
    delta: ChoiceDelta
    index: int = 0


@dataclass
class ChatChunk:
    choices: list[Choice] = field(default_factory=list)


class LLM(abc.ABC):
    @abc.abstractmethod
    async def chat(
        self,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: function_context.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream": ...


class LLMStream(abc.ABC):
    def __init__(self) -> None:
        self._called_functions: list[function_context.CalledFunction] = []

    @property
    def called_functions(self) -> list[function_context.CalledFunction]:
        """List of called functions from this stream."""
        return self._called_functions

    @abc.abstractmethod
    async def gather_function_results(
        self,
    ) -> list[function_context.CalledFunction]: ...

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...

    @abc.abstractmethod
    async def aclose(self) -> None: ...
