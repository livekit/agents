from __future__ import annotations

import abc
from typing import Callable

from attrs import define

from .chat_context import ChatContext, ChatRole
from .function_context import FunctionContext


@define
class ChoiceDelta:
    role: ChatRole
    content: str | None = None


@define
class Choice:
    delta: ChoiceDelta
    index: int = 0


@define
class ChatChunk:
    choices: list[Choice] = []


class LLM(abc.ABC):
    @abc.abstractmethod
    async def chat(
        self,
        history: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream": ...


@define
class CalledFunction:
    fnc_name: str
    fnc: Callable
    args: dict


class LLMStream(abc.ABC):
    def __init__(self) -> None:
        # fnc_name, args..
        self._called_functions: list[CalledFunction] = []

    @property
    def called_functions(self) -> list[CalledFunction]:
        """List of called functions from this stream."""
        return self._called_functions

    @abc.abstractmethod
    def __aiter__(self) -> "LLMStream": ...

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...

    @abc.abstractmethod
    async def aclose(self, wait: bool = True) -> None: ...
