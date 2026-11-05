# GENERATED FILE -- DO NOT EDIT.
# From ABI declarations and authored plugin bases:
#     rust/crates/lk-agents-abi/src/llm/llm.py

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing_extensions import TypeVar

from livekit import rtc

from .. import _bindings, _core
from ..types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr
from ..utils import aio
from .chat_items import ChatRole, MetricsMetadata
from .types import ProviderId
from .usage import TokenUsage

if TYPE_CHECKING:
    from ..llm.chat_context import ChatContext
    from ..llm.tool_context import Tool, ToolChoice
    from .chat_items import InferenceDetail


# Record declarations:
#     rust/crates/lk-agents-abi/src/llm/events.rs
#     rust/crates/lk-agents-abi/src/llm/provider_format/image.rs


class FunctionToolCall(BaseModel):
    type: Literal["function"] = "function"
    name: str
    arguments: str
    call_id: str
    extra: dict[str, JsonValue] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class ChoiceDelta(BaseModel):
    role: ChatRole | None = None
    content: str | None = None
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)
    extra: dict[str, JsonValue] | None = None
    """Provider-specific extra data (e.g., Google thought signatures)."""


class CollectedResponse(BaseModel):
    text: str = ""
    tool_calls: list[FunctionToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    extra: dict[str, JsonValue] = Field(default_factory=dict)
    """Provider-specific extra data accumulated across chunks
    (e.g., xAI encrypted reasoning, Google thought signatures).
    """


class ChatChunk(BaseModel):
    id: str
    delta: ChoiceDelta | None = None
    usage: TokenUsage | None = None

    def has_response(self) -> bool:
        """Whether this chunk delivered generation the caller can see.

        Token counts and provider metadata (a gateway deployment stamp, a thought
        signature) reach the caller without being output: they neither start the
        clock on time-to-first-token nor give a retry anything to duplicate.
        """
        return _bindings.chat_chunk_has_response(self)


class LLMError(BaseModel):
    """The public provider error event; its Python exception is retained but never serialized."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["llm_error"] = "llm_error"
    timestamp: float
    label: str
    error: Annotated[Exception, Field(exclude=True)]
    recoverable: bool


@dataclass(kw_only=True, slots=True)
class SerializedImage:
    inference_detail: InferenceDetail
    mime_type: str | None
    data_bytes: bytes | None = None
    external_url: str | None = None


FunctionToolCall.__module__ = "livekit.agents.llm"
ChoiceDelta.__module__ = "livekit.agents.llm"
CollectedResponse.__module__ = "livekit.agents.llm"
ChatChunk.__module__ = "livekit.agents.llm"
LLMError.__module__ = "livekit.agents.llm"
SerializedImage.__module__ = "livekit.agents.llm.utils"


TEvent = TypeVar("TEvent", default=Never, contravariant=True)


class LLM(
    ABC,
    rtc.EventEmitter[Literal["metrics_collected", "error"] | TEvent],
    Generic[TEvent],
):
    """A language model.

    Plugins implement ``chat()``, returning an ``LLMStream`` whose ``_run`` sends the provider's
    chunks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._label = f"{type(self).__module__}.{type(self).__name__}"
        self._prewarm_task: asyncio.Task[None] | None = None

    def _core_leg(self) -> object:
        """The Python provider implementation used by the native runtime's declared proxy."""
        return self

    @property
    def label(self) -> str:
        """The name logs and metrics give this instance: its class path."""
        return self._label

    @property
    def model(self) -> str:
        """The model name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def provider(self) -> str:
        """The provider name; ``"unknown"`` until a plugin overrides it."""
        return "unknown"

    @property
    def metrics_metadata(self) -> MetricsMetadata:
        """The model and provider labelling this instance's turn metrics."""
        return {"model_name": self.model, "model_provider": self.provider}

    def prewarm(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Pre-warm connection to the LLM service.

        Establishes DNS resolution and the TLS connection to the provider before the first inference
        request, reducing time-to-first-token on the initial reply. It is called automatically when
        a ``VoiceSession`` is constructed and when an agent activity starts, but can also be called
        directly.

        Non-blocking (fire-and-forget) and idempotent. Providers enable it by overriding
        ``_prewarm_impl``.

        Args:
            loop: Event loop to schedule the prewarm request on. Defaults to the running event loop.
        """
        if type(self)._prewarm_impl is LLM._prewarm_impl:
            return  # nothing to warm
        if self._prewarm_task is not None:
            return
        if loop is None:
            loop = asyncio.get_event_loop()

        async def _run() -> None:
            try:
                await self._prewarm_impl()
            except Exception:
                pass

        self._prewarm_task = loop.create_task(_run())

    async def _prewarm_impl(self) -> None:
        """The provider's own ``prewarm``: overriding it enables ``prewarm()``.

        A cheap, token-free request (e.g. listing models) on the client that serves the
        requests, so DNS and TLS are established and a connection is left in the pool. Its
        exceptions are swallowed; a subclass overriding ``aclose`` must call
        ``await super().aclose()`` to cancel one in flight.
        """

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
    ) -> LLMStream:
        """Open a completion of ``chat_ctx``, its attempts retried per ``conn_options``.

        Args:
            chat_ctx: The conversation to complete.
            tools: The tools the model may call.
            conn_options: The timeout and retries of the request.
            parallel_tool_calls: Whether the model may call several tools at once.
            tool_choice: Whether and which tool the model must call.
            extra_kwargs: Provider-specific request options.
        """

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self.emit(
            "error",
            LLMError(
                timestamp=time.time(),
                label=self._label,
                error=api_error,
                recoverable=recoverable,
            ),
        )

    async def aclose(self) -> None:
        """Close the provider and everything it opened."""
        if self._prewarm_task is not None:
            await aio.cancel_and_wait(self._prewarm_task)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    def _core_instance_id(self) -> int | None:
        return id(self)

    def _core_label(self) -> str:
        return self.label

    def _core_model(self) -> str:
        return self.model

    def _core_id(self) -> ProviderId:
        return {"label": self.label, "model": self.model, "provider": self.provider}

    def _core_prewarm(self) -> None:
        self.prewarm()

    def _core_failed(self, error: Exception, recoverable: bool) -> None:
        self._emit_error(error, recoverable)

    async def _core_chat(
        self, req: _core.ChatRequest, conn: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> LLMStream:
        from .._bindings import chat_request

        chat_ctx, tools = chat_request(req)
        out = self.chat(
            chat_ctx=chat_ctx,
            conn_options=conn,
            tools=tools,
            tool_choice=NOT_GIVEN if req.tool_choice is None else req.tool_choice,
            parallel_tool_calls=(
                NOT_GIVEN if req.parallel_tool_calls is None else req.parallel_tool_calls
            ),
            extra_kwargs=req.extra or NOT_GIVEN,
        )
        out._core_attempts.core = True
        return out


class _LLMEventChannel(aio.Chan[ChatChunk]):
    def __init__(self) -> None:
        super().__init__()
        self.output_sent = False

    def send_nowait(self, value: ChatChunk) -> None:
        super().send_nowait(value)
        # a provider can raise before the consumer reads the chunk
        self.output_sent = self.output_sent or value.has_response()


class LLMStream(ABC):
    """One completion: ``ChatChunk`` out.

    Plugins implement ``_run``, sending the provider's chunks to ``_event_ch``; its attempts are
    retried per ``conn_options``.
    """

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
        self._metrics = self._new_metrics()
        self._core_attempts = _bindings.AttemptLoop("llm", conn_options, self._emit_error)
        self._task = asyncio.create_task(self._main_task(), name="LLM._main_task")
        self._task.add_done_callback(lambda _: self._event_ch.close())

    @abstractmethod
    async def _run(self) -> None: ...

    def _new_metrics(self) -> _core.LlmMetricState:
        llm = self._llm
        return _core.LlmMetricState(time.perf_counter(), llm.label, llm.model, llm.provider)

    async def _attempt(self) -> None:
        self._metrics = self._new_metrics()
        await self._run()

    async def _main_task(self) -> None:
        # a failure after output reached the reader is final: the core never retries it
        await self._core_attempts.run(self._attempt, produced=lambda: self._event_ch.output_sent)
        record = self._metrics.finish(time.perf_counter(), False, time.time())
        _bindings.emit_metrics(self._llm, record)

    def _emit_error(self, api_error: Exception, recoverable: bool) -> None:
        self._llm._emit_error(api_error, recoverable)

    def send(self, chunk: ChatChunk) -> None:
        """Hand ``chunk`` to the reader."""
        self._metrics.observe(time.perf_counter(), chunk)
        self._event_ch.send_nowait(chunk)

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

    def to_str_iterable(self) -> AsyncIterable[str]:
        """The stream as its text, for a completion that calls no tool."""

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
            response = await my_llm.chat(chat_ctx=ctx, tools=tools).collect()
            print(response.text)
            ```

        Tool calls are available in ``response.tool_calls``. Use ``Agent.run()``
        when the framework should execute tools and continue the conversation.
        """
        text_parts: list[str] = []
        tool_calls: list[FunctionToolCall] = []
        usage: TokenUsage | None = None
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

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()




# The public module identity is assigned after all classes are built.
LLM.__module__ = "livekit.agents.llm"
LLMStream.__module__ = "livekit.agents.llm"
