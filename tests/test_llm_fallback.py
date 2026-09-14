from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.llm import (
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FallbackAdapter,
    LLMStream,
    ProviderToolCall,
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


class _ProviderToolLLMStream(LLMStream):
    async def _run(self) -> None:
        self.emit(
            "provider_tool_call",
            ProviderToolCall(
                phase="started", call_id="provider-call", name="web_search", arguments="{}"
            ),
        )
        self._event_ch.send_nowait(
            ChatChunk(id="response", delta=ChoiceDelta(role="assistant", content="done"))
        )
        self.emit(
            "provider_tool_call",
            ProviderToolCall(
                phase="done",
                status="done",
                call_id="provider-call",
                name="web_search",
                arguments="{}",
                result="result",
            ),
        )


class _ProviderToolLLM(_NamedLLM):
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        return _ProviderToolLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options
        )


class _RecoveringProviderToolLLMStream(LLMStream):
    def __init__(self, *args: Any, attempt: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._attempt = attempt

    async def _run(self) -> None:
        call_id = f"provider-call-{self._attempt}"
        self.emit(
            "provider_tool_call",
            ProviderToolCall(phase="started", call_id=call_id, name="web_search"),
        )
        self.emit(
            "provider_tool_call",
            ProviderToolCall(phase="done", status="done", call_id=call_id, name="web_search"),
        )
        if self._attempt == 1:
            raise APIConnectionError("primary failed")
        self._event_ch.send_nowait(
            ChatChunk(id="recovered", delta=ChoiceDelta(role="assistant", content="recovered"))
        )


class _RecoveringProviderToolLLM(_NamedLLM):
    def __init__(self) -> None:
        super().__init__(model="primary-model", provider="primary")
        self._attempt = 0

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        self._attempt += 1
        return _RecoveringProviderToolLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
            attempt=self._attempt,
        )


async def test_forwards_provider_tool_events_from_active_llm() -> None:
    child = _ProviderToolLLM(model="primary-model", provider="primary")
    fallback_adapter = FallbackAdapter([child])
    events: list[ProviderToolCall] = []

    try:
        async with fallback_adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            stream.on("provider_tool_call", events.append)
            async for _ in stream:
                pass

        assert [(event.phase, event.call_id) for event in events] == [
            ("started", "provider-call"),
            ("done", "provider-call"),
        ]
    finally:
        await fallback_adapter.aclose()


async def test_does_not_forward_provider_tool_events_from_recovery_probe() -> None:
    primary = _RecoveringProviderToolLLM()
    secondary = _NamedLLM(
        model="secondary-model",
        provider="secondary",
        fake_responses=[
            FakeLLMResponse(input="hello", content="fallback response", ttft=0.01, duration=0.02)
        ],
    )
    fallback_adapter = FallbackAdapter([primary, secondary])
    events: list[ProviderToolCall] = []

    try:
        chat_ctx = ChatContext.empty()
        chat_ctx.add_message(role="user", content="hello")
        async with fallback_adapter.chat(chat_ctx=chat_ctx) as stream:
            stream.on("provider_tool_call", events.append)
            async for _ in stream:
                pass

        recovery_task = fallback_adapter._status[0].recovering_task
        assert recovery_task is not None
        await recovery_task

        assert [(event.phase, event.call_id) for event in events] == [
            ("started", "provider-call-1"),
            ("done", "provider-call-1"),
        ]
    finally:
        await fallback_adapter.aclose()


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
