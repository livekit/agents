import abc
import enum

from attrs import define

from .function_context import FunctionContext


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


class LLM(abc.ABC):
    @abc.abstractmethod
    async def chat(
        self,
        history: ChatContext,
        fnc_ctx: FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
    ) -> "LLMStream": ...


class LLMStream(abc.ABC):
    @abc.abstractmethod
    def __aiter__(self) -> "LLMStream": ...

    @abc.abstractmethod
    async def __anext__(self) -> ChatChunk: ...

    @abc.abstractmethod
    async def aclose(self, wait: bool = True) -> None: ...
