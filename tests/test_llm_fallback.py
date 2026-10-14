from __future__ import annotations

import asyncio
from collections import deque
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
@pytest.mark.parametrize("sticky", [False, True])
async def test_no_retry_after_output(
    chunk: ChatChunk, with_fallback: bool, error_kind: str, child_retries: int, sticky: bool
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
        [primary, fallback] if with_fallback else [primary],
        max_retry_per_llm=child_retries,
        sticky=sticky,
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


class _ScriptedLLM(_NamedLLM):
    def __init__(self, model: str, responses: list[str | Exception | None]) -> None:
        super().__init__(model=model, provider=f"{model}-provider")
        self.responses = deque(responses)
        self.requests = 0
        self.finish: asyncio.Event | None = None

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        self.requests += 1
        response = self.responses.popleft() if self.responses else self.model
        return _ScriptedLLMStream(
            self,
            response=response,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _ScriptedLLMStream(LLMStream):
    def __init__(
        self, llm: _ScriptedLLM, *, response: str | Exception | None, **kwargs: Any
    ) -> None:
        super().__init__(llm, **kwargs)
        self._response = response
        self._finish = llm.finish

    async def _run(self) -> None:
        if isinstance(self._response, Exception):
            raise self._response
        if self._response is not None:
            self._event_ch.send_nowait(
                ChatChunk(id=self._llm.model, delta=ChoiceDelta(content=self._response))
            )
        if self._finish is not None:
            await self._finish.wait()


async def _wait_for_recovery(adapter: FallbackAdapter) -> None:
    await asyncio.wait_for(
        asyncio.gather(
            *(status.recovering_task for status in adapter._status if status.recovering_task)
        ),
        timeout=5,
    )


@pytest.mark.parametrize("sticky", [None, False, True], ids=["default", "priority", "sticky"])
async def test_recovered_primary_routing(sticky: bool | None) -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError(), "recovery"])
    fallback = _ScriptedLLM("fallback", [])
    adapter = FallbackAdapter(
        [primary, fallback], **({"sticky": sticky} if sticky is not None else {})
    )
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)
        assert adapter._status[0].available
        assert primary.requests == 2
        assert adapter.metrics_metadata == fallback.metrics_metadata

        expected = fallback if sticky else primary
        assert adapter.model == expected.model
        assert adapter.provider == expected.provider
        for _ in range(2):
            response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
            assert response.text == expected.model
        assert primary.requests == (2 if sticky else 4)
        assert fallback.requests == (3 if sticky else 1)
    finally:
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("primary_recovers", [False, True])
async def test_sticky_model_failure_uses_remaining_priority_order(primary_recovers: bool) -> None:
    primary = _ScriptedLLM(
        "primary",
        [APITimeoutError()] + ([] if primary_recovers else [APITimeoutError()] * 3),
    )
    fallback = _ScriptedLLM("fallback", ["fallback", APITimeoutError()])
    last = _ScriptedLLM("last", [])
    adapter = FallbackAdapter([primary, fallback, last], sticky=True)
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        expected = primary if primary_recovers else last
        assert response.text == expected.model
        await _wait_for_recovery(adapter)
        assert fallback.requests == 3  # served, failed, recovered
        assert last.requests == (0 if primary_recovers else 1)
        assert adapter.model == expected.model
        assert adapter.provider == expected.provider
        assert adapter.metrics_metadata == expected.metrics_metadata

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == expected.model
    finally:
        await _close_retry_adapter(adapter)


async def test_sticky_retries_from_primary_when_all_models_are_unavailable() -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError()] * 5)
    fallback = _ScriptedLLM("fallback", ["fallback"] + [APITimeoutError()] * 5)
    adapter = FallbackAdapter([primary, fallback], sticky=True)
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)

        with pytest.raises(APIConnectionError):
            await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        await _wait_for_recovery(adapter)
        assert all(not status.available for status in adapter._status)

        primary.responses.clear()
        fallback.responses.clear()
        fallback_requests = fallback.requests
        assert adapter.model == "primary"
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "primary"
        assert fallback.requests == fallback_requests
    finally:
        await _close_retry_adapter(adapter)


async def test_sticky_keeps_model_that_succeeds_after_all_models_failed() -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError()] * 3 + ["recovery"])
    fallback = _ScriptedLLM("fallback", [APITimeoutError()] * 2)
    adapter = FallbackAdapter([primary, fallback], sticky=True)
    try:
        with pytest.raises(APIConnectionError):
            await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        await _wait_for_recovery(adapter)
        assert all(not status.available for status in adapter._status)

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)
        assert adapter._status[0].available

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        assert primary.requests == 4
    finally:
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("primary_recovers_first", [False, True])
async def test_sticky_success_overrides_concurrent_failed_attempt(
    primary_recovers_first: bool,
) -> None:
    primary = _ScriptedLLM(
        "primary", [APITimeoutError(), "recovery", APITimeoutError(), "recovery"]
    )
    fallback = _ScriptedLLM(
        "fallback", ["fallback", "fallback", APITimeoutError(), APITimeoutError()]
    )
    adapter = FallbackAdapter([primary, fallback], sticky=True)
    fallback_finish = asyncio.Event()
    primary_recovery_finish = asyncio.Event()
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)

        fallback.finish = fallback_finish
        async with adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            chunk = await asyncio.wait_for(anext(stream), timeout=5)
            assert chunk.delta is not None and chunk.delta.content == "fallback"
            fallback.finish = None
            primary.finish = primary_recovery_finish

            with pytest.raises(APIConnectionError):
                await adapter.chat(chat_ctx=ChatContext.empty()).collect()

            if primary_recovers_first:
                primary_recovery_finish.set()
                await _wait_for_recovery(adapter)

            fallback_finish.set()
            await stream.collect()

        primary_recovery_finish.set()
        await _wait_for_recovery(adapter)
        assert adapter.model == "fallback"
        assert adapter.provider == "fallback-provider"

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        assert primary.requests == 4
        assert fallback.requests == 5
    finally:
        fallback_finish.set()
        primary_recovery_finish.set()
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("older_finishes_first", [False, True])
async def test_sticky_older_success_preserves_newer_successful_failover(
    older_finishes_first: bool,
) -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError(), "recovery", "primary"])
    fallback = _ScriptedLLM("fallback", ["fallback", "fallback", APITimeoutError(), "recovery"])
    adapter = FallbackAdapter([primary, fallback], sticky=True)
    fallback_finish = asyncio.Event()
    primary_finish = asyncio.Event()
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        await _wait_for_recovery(adapter)

        fallback.finish = fallback_finish
        async with adapter.chat(chat_ctx=ChatContext.empty()) as older_stream:
            chunk = await asyncio.wait_for(anext(older_stream), timeout=5)
            assert chunk.delta is not None and chunk.delta.content == "fallback"
            fallback.finish = None
            primary.finish = primary_finish

            async with adapter.chat(chat_ctx=ChatContext.empty()) as newer_stream:
                chunk = await asyncio.wait_for(anext(newer_stream), timeout=5)
                assert chunk.delta is not None and chunk.delta.content == "primary"
                await _wait_for_recovery(adapter)

                if older_finishes_first:
                    fallback_finish.set()
                    await older_stream.collect()

                primary_finish.set()
                await newer_stream.collect()

            if not older_finishes_first:
                fallback_finish.set()
                await older_stream.collect()

        assert adapter.model == "primary"
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "primary"
        assert primary.requests == fallback.requests == 4
    finally:
        fallback_finish.set()
        primary_finish.set()
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("newer_recovers", [False, True])
async def test_sticky_older_success_replaces_failed_newer_selection(newer_recovers: bool) -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError()] * 3 + ["recovery"])
    middle = _ScriptedLLM("middle", ["middle", "middle"] + [APITimeoutError()] * 3)
    last = _ScriptedLLM(
        "last", ["last", APITimeoutError(), "recovery" if newer_recovers else APITimeoutError()]
    )
    adapter = FallbackAdapter([primary, middle, last], sticky=True)
    finish = asyncio.Event()
    try:
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "middle"
        await _wait_for_recovery(adapter)

        middle.finish = finish
        async with adapter.chat(chat_ctx=ChatContext.empty()) as older_stream:
            chunk = await asyncio.wait_for(anext(older_stream), timeout=5)
            assert chunk.delta is not None and chunk.delta.content == "middle"
            middle.finish = None

            response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
            assert response.text == "last"
            await _wait_for_recovery(adapter)

            with pytest.raises(APIConnectionError):
                await adapter.chat(chat_ctx=ChatContext.empty()).collect()
            await _wait_for_recovery(adapter)
            assert [status.available for status in adapter._status] == [
                True,
                False,
                newer_recovers,
            ]

            finish.set()
            await older_stream.collect()

        assert adapter.model == "middle"
        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "middle"
    finally:
        finish.set()
        await _close_retry_adapter(adapter)


@pytest.mark.parametrize("empty_response", [False, True])
async def test_sticky_model_survives_stream_close(empty_response: bool) -> None:
    primary = _ScriptedLLM("primary", [APITimeoutError()])
    fallback = _ScriptedLLM("fallback", [None] if empty_response else [])
    finish = asyncio.Event()
    if not empty_response:
        fallback.finish = finish
    adapter = FallbackAdapter([primary, fallback], sticky=True)
    try:
        async with adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            if empty_response:
                assert [chunk async for chunk in stream] == []
            else:
                chunk = await asyncio.wait_for(anext(stream), timeout=5)
                assert chunk.delta is not None and chunk.delta.content == "fallback"
            await _wait_for_recovery(adapter)
        finish.set()

        response = await adapter.chat(chat_ctx=ChatContext.empty()).collect()
        assert response.text == "fallback"
        assert primary.requests == 2
        assert adapter.model == "fallback"
        assert adapter.provider == "fallback-provider"
    finally:
        finish.set()
        await _close_retry_adapter(adapter)


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
            "usage_source": "provider_plugin",
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
            "usage_source": "provider_plugin",
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
