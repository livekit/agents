from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    CompletionUsage,
    FallbackAdapter,
    FunctionToolCall,
    LLMError,
    LLMStream,
    Tool,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]


class _RetryLLM(FakeLLM):
    def __init__(
        self,
        chunk: ChatChunk | None,
        error: Exception | None,
        *,
        success_chunk: ChatChunk | None = None,
    ) -> None:
        super().__init__()
        self.chunk = chunk
        self.error = error
        self.success_chunk = success_chunk or chunk or _TEXT_CHUNK
        self.requests = 0
        self.attempts = 0

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        self.requests += 1
        return _RetryLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _RetryLLMStream(LLMStream):
    async def _run(self) -> None:
        assert isinstance(self._llm, _RetryLLM)
        self._llm.attempts += 1
        if self._llm.attempts == 1:
            if self._llm.chunk is not None:
                self._event_ch.send_nowait(self._llm.chunk)
            if self._llm.error is not None:
                raise self._llm.error
        else:
            self._event_ch.send_nowait(self._llm.success_chunk)


async def _close_retry_adapter(adapter: FallbackAdapter) -> None:
    await asyncio.gather(
        *(status.recovering_task for status in adapter._status if status.recovering_task)
    )
    await adapter.aclose()


_TEXT_CHUNK = ChatChunk(id="text", delta=ChoiceDelta(content="The answer."))
_TOOL_CHUNK = ChatChunk(
    id="tool",
    delta=ChoiceDelta(
        tool_calls=[FunctionToolCall(call_id="call-1", name="transfer_call", arguments="{}")]
    ),
)


# Based on @dtran26's diagnosis and reproducer: livekit/agents-js#2477.
@pytest.mark.parametrize("chunk", [_TEXT_CHUNK, _TOOL_CHUNK], ids=["text", "tool"])
@pytest.mark.parametrize("with_fallback", [False, True], ids=["outer", "fallback"])
@pytest.mark.parametrize("error_kind", ["timeout", "status", "nonretryable", "unexpected"])
@pytest.mark.parametrize("child_retries", [0, 1])
async def test_no_retry_after_output(
    chunk: ChatChunk, with_fallback: bool, error_kind: str, child_retries: int
) -> None:
    error: Exception
    if error_kind == "status":
        error = APIStatusError("After output", status_code=503, request_id="request-1")
    elif error_kind == "unexpected":
        error = ValueError("After output")
    else:
        error = APITimeoutError("After output", retryable=error_kind != "nonretryable")
    primary = _RetryLLM(chunk, error)
    fallback = _RetryLLM(chunk, None)
    adapter = FallbackAdapter(
        [primary, fallback] if with_fallback else [primary], max_retry_per_llm=child_retries
    )
    errors: list[LLMError] = []
    adapter.on("error", errors.append)
    chunks: list[ChatChunk] = []
    try:
        with pytest.raises(type(error)) as exc_info:
            async with adapter.chat(
                chat_ctx=ChatContext.empty(),
                conn_options=APIConnectOptions(max_retry=3, retry_interval=0),
            ) as stream:
                async for result in stream:
                    chunks.append(result)

        assert chunks == [chunk]
        assert primary.requests == primary.attempts == 1
        assert fallback.requests == 0
        assert exc_info.value is error
        assert len(errors) == 1
        assert errors[0].error is error
        assert not errors[0].recoverable
        if isinstance(error, APIError):
            assert not error.retryable
    finally:
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize(
    "chunk",
    [
        None,
        ChatChunk(id="role", delta=ChoiceDelta(role="assistant")),
        ChatChunk(id="empty", delta=ChoiceDelta(content="")),
        ChatChunk(
            id="usage",
            usage=CompletionUsage(completion_tokens=0, prompt_tokens=1, total_tokens=1),
        ),
    ],
    ids=["no-output", "role", "empty-text", "usage"],
)
@pytest.mark.parametrize("retry_path", ["child", "fallback", "outer"])
async def test_retries_before_output(chunk: ChatChunk | None, retry_path: str) -> None:
    error = APITimeoutError("Before output")
    primary = _RetryLLM(chunk, error, success_chunk=_TEXT_CHUNK)
    fallback = _RetryLLM(_TEXT_CHUNK, None)
    adapter = FallbackAdapter(
        [primary, fallback] if retry_path == "fallback" else [primary],
        max_retry_per_llm=1 if retry_path == "child" else 0,
        retry_interval=0,
    )
    errors: list[LLMError] = []
    adapter.on("error", errors.append)
    try:
        async with adapter.chat(
            chat_ctx=ChatContext.empty(),
            conn_options=APIConnectOptions(
                max_retry=3 if retry_path == "outer" else 0, retry_interval=0
            ),
        ) as stream:
            chunks = [result async for result in stream]

        expected = ([chunk] if chunk is not None else []) + [_TEXT_CHUNK]
        assert chunks == expected
        assert error.retryable
        if retry_path == "child":
            assert primary.requests == 1
            assert primary.attempts == 2
        elif retry_path == "fallback":
            assert fallback.requests == 1
        else:
            assert primary.requests >= 2
        assert len(errors) == (1 if retry_path == "outer" else 0)
        assert all(event.recoverable for event in errors)
    finally:
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("chunk", [_TEXT_CHUNK, _TOOL_CHUNK], ids=["text", "tool"])
@pytest.mark.parametrize("with_fallback", [False, True], ids=["outer", "fallback"])
@pytest.mark.parametrize("child_retries", [0, 1])
async def test_retry_after_output_when_enabled(
    chunk: ChatChunk, with_fallback: bool, child_retries: int
) -> None:
    error = APITimeoutError("After output")
    primary = _RetryLLM(chunk, error)
    fallback = _RetryLLM(chunk, None)
    adapter = FallbackAdapter(
        [primary, fallback] if with_fallback else [primary],
        retry_on_chunk_sent=True,
        max_retry_per_llm=child_retries,
    )
    errors: list[LLMError] = []
    adapter.on("error", errors.append)
    try:
        async with adapter.chat(
            chat_ctx=ChatContext.empty(),
            conn_options=APIConnectOptions(max_retry=3, retry_interval=0),
        ) as stream:
            chunks = [result async for result in stream]

        assert chunks == [chunk, chunk]
        assert error.retryable
        assert len(errors) == (0 if with_fallback or child_retries else 1)
        assert all(event.recoverable for event in errors)
        if child_retries:
            assert primary.requests == 1
            assert primary.attempts == 2
            assert fallback.requests == 0
        elif with_fallback:
            assert fallback.requests == 1
        else:
            assert primary.requests >= 2
    finally:
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("chunk", [_TEXT_CHUNK, _TOOL_CHUNK], ids=["text", "tool"])
async def test_direct_provider_retries_after_output(chunk: ChatChunk) -> None:
    provider = _RetryLLM(chunk, APITimeoutError("After output"))
    try:
        async with provider.chat(
            chat_ctx=ChatContext.empty(), conn_options=APIConnectOptions(max_retry=1)
        ) as stream:
            chunks = [result async for result in stream]
        assert chunks == [chunk, chunk]
        assert provider.requests == 1
        assert provider.attempts == 2
    finally:
        await provider.aclose()


class PrewarmableLLM(FakeLLM):
    """FakeLLM that opts into prewarming by overriding ``_prewarm_impl``."""

    def __init__(self) -> None:
        super().__init__()
        self.prewarmed = asyncio.Event()

    async def _prewarm_impl(self) -> None:
        self.prewarmed.set()


class RecordingLLM(FakeLLM):
    """FakeLLM that records the event loop it was asked to prewarm on."""

    def __init__(self) -> None:
        super().__init__()
        self.prewarm_loop: asyncio.AbstractEventLoop | None = None

    def prewarm(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.prewarm_loop = loop


async def test_prewarm_forwarded_to_primary_llm() -> None:
    primary = PrewarmableLLM()
    fallback = PrewarmableLLM()

    fallback_adapter = FallbackAdapter([primary, fallback])
    try:
        fallback_adapter.prewarm()

        await asyncio.wait_for(primary.prewarmed.wait(), timeout=5)
        assert not fallback.prewarmed.is_set(), (
            "expected only the primary LLM to be prewarmed, the fallbacks should stay cold"
        )
    finally:
        await fallback_adapter.aclose()
        await primary.aclose()
        await fallback.aclose()


class _NamedLLM(FakeLLM):
    """FakeLLM with a configurable model/provider so tests can tell instances apart."""

    def __init__(self, *, model: str, provider: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model_name = model
        self._provider_name = provider

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return self._provider_name


class _FailingLLMStream(LLMStream):
    async def _run(self) -> None:
        raise APIConnectionError("failing llm")


class _FailingLLM(_NamedLLM):
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        return _FailingLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


async def test_reports_active_instance_model_and_provider() -> None:
    primary = _FailingLLM(model="primary-model", provider="primary")
    fallback = _NamedLLM(
        model="fallback-model",
        provider="fallback",
        fake_responses=[
            FakeLLMResponse(input="hello", content="hi there", ttft=0.01, duration=0.02)
        ],
    )

    fallback_adapter = FallbackAdapter([primary, fallback])
    try:
        # before any traffic, the primary is reported
        assert fallback_adapter.metrics_metadata == {
            "model_name": "primary-model",
            "model_provider": "primary",
        }

        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        async with fallback_adapter.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass

        # the fallback served the request, so metrics must be labeled with it
        assert fallback_adapter.metrics_metadata == {
            "model_name": "fallback-model",
            "model_provider": "fallback",
        }
        # model and provider follow the instance that serves next, so spans and metrics name
        # the model that answered rather than the adapter; the label stays the adapter's own
        assert fallback_adapter.model == "fallback-model"
        assert fallback_adapter.provider == "fallback"
        assert "FallbackAdapter" in fallback_adapter.label
        # once the primary recovers (its recovery task flips it back to available) the next
        # request goes to it first, so that is what model and provider report
        fallback_adapter._status[0].available = True
        assert fallback_adapter.model == "primary-model"
        assert fallback_adapter.provider == "primary"
    finally:
        await fallback_adapter.aclose()


async def test_prewarm_forwards_event_loop() -> None:
    primary = RecordingLLM()

    fallback_adapter = FallbackAdapter([primary])
    # a loop distinct from the running one, so the assertion fails if `loop` is dropped
    # and the wrapped LLM falls back to the running loop
    supplied_loop = asyncio.new_event_loop()
    try:
        fallback_adapter.prewarm(loop=supplied_loop)

        assert primary.prewarm_loop is supplied_loop, (
            "expected the provided event loop to be forwarded to the primary LLM"
        )
    finally:
        supplied_loop.close()
        await fallback_adapter.aclose()
