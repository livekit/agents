from __future__ import annotations

import abc
import enum

from attrs import define

from . import function_context


class ChatRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@define
class ChatMessage:
    role: ChatRole
    text: str


@define
class ChatContext:
    messages: list[ChatMessage] = []


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


@define
class ChoiceDelta:
    content: str | None = None
    role: ChatRole | None = None


@define
class Choice:
    delta: ChoiceDelta
    index: int = 0


@define
class ChatChunk:
    choices: list[Choice] = []


class LLMStream(abc.ABC):
    def __init__(self) -> None:
        # fnc_name, args..
        self._called_functions: list[function_context.CalledFunction] = []

    @property
    def called_functions(self) -> list[function_context.CalledFunction]:
        """List of called functions from this stream."""
        return self._called_functions

    @abc.abstractmethod
    def __aiter__(self) -> "LLMStream": ...

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...

    @abc.abstractmethod
    async def aclose(self, wait: bool = True) -> None: ...
