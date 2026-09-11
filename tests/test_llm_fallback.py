from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import APIConnectionError, APITimeoutError
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FallbackAdapter,
    LLMStream,
    Tool,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit]


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
        # the adapter keeps its own stable identity for spans, logs, and error events
        assert fallback_adapter.model == "FallbackAdapter"
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


class _SlowFirstTokenStream(LLMStream):
    """Enforces ``conn_options.timeout`` on the first token, like provider plugins do."""

    async def _run(self) -> None:
        assert isinstance(self._llm, _SlowFirstTokenLLM)
        try:
            await asyncio.wait_for(asyncio.sleep(self._llm.ttft), self._conn_options.timeout)
        except asyncio.TimeoutError:
            raise APITimeoutError(
                f"{self._llm.model} exceeded the {self._conn_options.timeout}s attempt timeout"
            ) from None
        self._event_ch.send_nowait(
            ChatChunk(id=str(id(self)), delta=ChoiceDelta(role="assistant", content="hello"))
        )


class _SlowFirstTokenLLM(_NamedLLM):
    """FakeLLM with a fixed time-to-first-token, bounded by the attempt timeout."""

    def __init__(self, *, model: str, ttft: float) -> None:
        super().__init__(model=model, provider="fake")
        self.ttft = ttft

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        return _SlowFirstTokenStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


async def _collect_text(fallback_adapter: FallbackAdapter) -> str:
    text = ""
    async with fallback_adapter.chat(chat_ctx=ChatContext.empty()) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                text += chunk.delta.content
    return text


async def test_fallback_attempt_timeout_gives_fallbacks_a_longer_window() -> None:
    primary = _SlowFirstTokenLLM(model="primary", ttft=10.0)  # misses its timeout
    fallback = _SlowFirstTokenLLM(model="fallback", ttft=0.3)  # needs more than the primary's

    fallback_adapter = FallbackAdapter(
        [primary, fallback], attempt_timeout=0.15, fallback_attempt_timeout=0.5
    )
    try:
        assert await _collect_text(fallback_adapter) == "hello"
        # the primary was cut at its own timeout, not given the fallback's window
        assert [status.available for status in fallback_adapter._status] == [False, True]
        # let the primary's background recovery attempt finish
        await asyncio.sleep(0.3)
    finally:
        await fallback_adapter.aclose()


async def test_attempt_timeout_applies_to_all_llms_by_default() -> None:
    primary = _SlowFirstTokenLLM(model="primary", ttft=10.0)
    fallback = _SlowFirstTokenLLM(model="fallback", ttft=0.3)

    fallback_adapter = FallbackAdapter([primary, fallback], attempt_timeout=0.15)
    try:
        with pytest.raises(APIConnectionError):
            await _collect_text(fallback_adapter)
        await asyncio.sleep(0.3)
    finally:
        await fallback_adapter.aclose()
