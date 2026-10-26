from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from types import TracebackType
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from livekit import rtc

from .._exceptions import APIError, APIStatusError
from .._generated.llm import (
    ChatChunk,
    ChoiceDelta as ChoiceDelta,
    CompletionUsage,
    FunctionToolCall,
)
from ..types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from ..utils import aio
from .chat_context import ChatContext, MetricsMetadata
from .tool_context import Tool, ToolChoice


class CollectedResponse(BaseModel):
    text: str = ""
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)
    usage: CompletionUsage | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific extra data accumulated across chunks
    (e.g., xAI encrypted reasoning, Google thought signatures)."""


class LLMError(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["llm_error"] = "llm_error"
    timestamp: float
    label: str
    error: Exception = Field(..., exclude=True)
    recoverable: bool


TEvent = TypeVar("TEvent")


class LLM(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._prewarm_task: asyncio.Task[None] | None = None

    @property
    def label(self) -> str:
        return self._label

    @property
    def model(self) -> str:
        """Get the model name/identifier for this LLM instance.

        Returns:
            The model name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their model information.
        """
        return "unknown"

    @property
    def provider(self) -> str:
        """Get the provider name/identifier for this LLM instance.

        Returns:
            The provider name if available, "unknown" otherwise.

        Note:
            Plugins should override this property to provide their provider information.
        """
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """Metadata used to label turn metrics emitted for this LLM instance."""
        return {"model_name": self.model, "model_provider": self.provider}

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> LLMStream: ...

    def prewarm(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Pre-warm connection to the LLM service.

        Establishes DNS resolution and the TLS connection to the provider before the
        first inference request, reducing time-to-first-token on the initial reply.
        It is called automatically when an ``LiveSession`` is constructed and when an
        agent activity starts, but can also be called directly.

        Non-blocking (fire-and-forget) and idempotent. Providers enable it by
        overriding ``_prewarm_impl``.

        Args:
            loop: Event loop to schedule the prewarm request on. Defaults to the
                running event loop.
        """
        if type(self)._prewarm_impl is LLM._prewarm_impl:
            return  # no provider-specific prewarm implemented

        if self._prewarm_task is not None:
            return

        if loop is None:
            loop = asyncio.get_event_loop()

        async def _prewarm() -> None:
            try:
                await self._prewarm_impl()
            except Exception:
                pass

        self._prewarm_task = loop.create_task(_prewarm())

    async def _prewarm_impl(self) -> None:
        """Provider-specific prewarm request, overriding it enables ``prewarm()``.

        Implementations should perform a cheap, token-free request (e.g. listing
        models) using the same client that serves chat requests, so DNS + TLS are
        established and a keep-alive connection is left in the pool. Exceptions
        are swallowed by ``prewarm()``; subclasses overriding ``aclose`` must call
        ``await super().aclose()`` to cancel an in-flight prewarm."""

    async def aclose(self) -> None:
        if self._prewarm_task is not None:
            await aio.cancel_and_wait(self._prewarm_task)

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class _LLMEventChannel(aio.Chan[ChatChunk]):
    def __init__(self) -> None:
        super().__init__()
        self.output_sent = False

    def send_nowait(self, value: ChatChunk) -> None:
        super().send_nowait(value)
        # a provider can raise before the consumer reads the chunk
        self.output_sent = self.output_sent or value.has_response()


class LLMStream(ABC):
    """One completion: ``_run`` sends its chunks to ``_event_ch``. One attempt; the core retries
    a failed request."""

    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._tools = tools
        self._conn_options = conn_options
        self._event_ch = _LLMEventChannel()
        self._retry_on_chunk_sent = True
        self._task = asyncio.create_task(self._main_task(), name="LLM._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        try:
            await self._run()
        except APIError as e:
            # 499 (Client Closed Request) - close gracefully without raising
            if isinstance(e, APIStatusError) and e.status_code == 499:
                return
            if not self._retry_on_chunk_sent and self._event_ch.output_sent:
                e.retryable = False
            self._emit_error(e, recoverable=False)
            raise
        except Exception as e:
            self._emit_error(e, recoverable=False)
            raise

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._llm.emit(
            "error",
            LLMError(
                timestamp=time.time(),
                label=self._llm._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def tools(self) -> list[Tool]:
        return self._tools

    async def aclose(self) -> None:
        await aio.cancel_and_wait(self._task)
        self._event_ch.close()

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_ch.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc  # noqa: B904

            raise StopAsyncIteration from None

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

    def to_str_iterable(self) -> AsyncIterable[str]:
        """
        Convert the LLMStream to an async iterable of strings.
        This assumes the stream will not call any tools.
        """

        async def _iterable() -> AsyncIterable[str]:
            async with self:
                async for chunk in self:
                    if chunk.delta and chunk.delta.content:
                        yield chunk.delta.content

        return _iterable()

    async def collect(self) -> CollectedResponse:
        """Collect the entire stream into a single response.

        Example:
            ```python
            from livekit.agents import llm

            response = await my_llm.chat(chat_ctx=ctx, tools=tools).collect()

            for tc in response.tool_calls:
                result = await llm.execute_function_call(tc, tool_ctx)
                ctx.insert(result.fnc_call)
                ctx.insert(result.fnc_call_out)
            ```
        """
        text_parts: list[str] = []
        tool_calls: list[FunctionToolCall] = []
        usage: CompletionUsage | None = None
        extra: dict[str, Any] = {}

        async with self:
            async for chunk in self:
                if chunk.delta:
                    if chunk.delta.content:
                        text_parts.append(chunk.delta.content)
                    if chunk.delta.tool_calls:
                        tool_calls.extend(chunk.delta.tool_calls)
                    if chunk.delta.extra:
                        extra.update(chunk.delta.extra)
                if chunk.usage is not None:
                    usage = chunk.usage

        return CollectedResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=usage,
            extra=extra,
        )
