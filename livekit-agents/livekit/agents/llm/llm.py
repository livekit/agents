from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Generic,
    Literal,
    Optional,
    TypedDict,
    TypeVar,
    Union,
)

from livekit import rtc
from livekit.agents._exceptions import APIConnectionError, APIError
from pydantic import BaseModel, Field
from typing_extensions import Required

from .. import utils
from ..log import logger
from ..metrics import LLMMetrics
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio
from .chat_context import ChatContext, ChatRole
from .tool_context import FunctionTool


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    total_tokens: int


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str


class ChoiceDelta(BaseModel):
    role: Optional[ChatRole] = None
    content: Optional[str] = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)


class ChatChunk(BaseModel):
    id: str
    delta: Optional[ChoiceDelta] = None
    usage: Optional[CompletionUsage] = None


# Used by ToolChoice
class Function(TypedDict, total=False):
    name: Required[str]


class ToolChoice(TypedDict, total=False):
    type: Required[Literal["function"]]
    function: Required[Function]


TEvent = TypeVar("TEvent")


class LLM(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected"], TEvent]],
    Generic[TEvent],
):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[Union[ToolChoice, Literal["auto", "required", "none"]]] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LLMStream": ...

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class LLMStream(ABC):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[FunctionTool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options

        self._event_ch = aio.Chan[ChatChunk]()
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="LLM._metrics_task"
        )

        self._task = asyncio.create_task(self._main_task())
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                if self._conn_options.max_retry == 0 or not e.retryable:
                    raise
                elif i == self._conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to generate LLM completion after {self._conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to generate LLM completion, retrying in {self._conn_options.retry_interval}s",
                        exc_info=e,
                        extra={
                            "llm": self._llm._label,
                            "attempt": i + 1,
                        },
                    )

                await asyncio.sleep(self._conn_options.retry_interval)

    @utils.log_exceptions(logger=logger)
    async def _metrics_monitor_task(self, event_aiter: AsyncIterable[ChatChunk]) -> None:
        start_time = time.perf_counter()
        ttft = -1.0
        request_id = ""
        usage: CompletionUsage | None = None

        async for ev in event_aiter:
            request_id = ev.id
            if ttft == -1.0:
                ttft = time.perf_counter() - start_time

            if ev.usage is not None:
                usage = ev.usage

        duration = time.perf_counter() - start_time
        metrics = LLMMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttft=ttft,
            duration=duration,
            cancelled=self._task.cancelled(),
            label=self._llm._label,
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            tokens_per_second=usage.completion_tokens / duration if usage else 0.0,
            error=None,
        )
        self._llm.emit("metrics_collected", metrics)

    @property
    def chat_ctx(self) -> ChatContext:
        """The chat context of this stream."""
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> list[FunctionTool]:
        """The function context of this stream."""
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        await self._metrics_task

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __aenter__(self) -> LLMStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
