from __future__ import annotations

import asyncio
from typing import Any

import pytest

from livekit.agents import APIConnectionError
from livekit.agents.llm import (
    LLM,
    ChatContext,
    FallbackAdapter,
    LLMStream,
    Tool,
    ToolChoice,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

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


class FlakyLLM(LLM):
    """Fails the first request, then blocks on the recovery attempt until released.

    Modelled on a provider outage: the adapter falls back to the next LLM and kicks off a
    background recovery request against this one, which is still in flight at ``aclose``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests = 0
        self.recovery_started = asyncio.Event()
        self.release_recovery = asyncio.Event()
        self.recovery_completed = False

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
        self.requests += 1
        return FlakyLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class FlakyLLMStream(LLMStream):
    def __init__(
        self,
        llm: FlakyLLM,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool],
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._flaky_llm = llm

    async def _run(self) -> None:
        if self._flaky_llm.requests == 1:
            raise APIConnectionError("primary is down")

        self._flaky_llm.recovery_started.set()
        await self._flaky_llm.release_recovery.wait()
        self._flaky_llm.recovery_completed = True


async def test_aclose_cancels_in_flight_recovery() -> None:
    primary = FlakyLLM()
    fallback = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hello", content="hi", ttft=0.0, duration=0.0)]
    )

    fallback_adapter = FallbackAdapter([primary, fallback])
    availability: list[bool] = []
    fallback_adapter.on("llm_availability_changed", lambda ev: availability.append(ev.available))

    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    try:
        async with fallback_adapter.chat(chat_ctx=chat_ctx) as stream:
            async for _ in stream:
                pass

        await asyncio.wait_for(primary.recovery_started.wait(), timeout=5)
        recovering_task = fallback_adapter._status[0].recovering_task
        assert recovering_task is not None and not recovering_task.done()

        await fallback_adapter.aclose()

        assert recovering_task.done(), (
            "expected aclose to cancel the in-flight recovery request, "
            "it outlives the stream that started it"
        )

        # releasing it must not resurrect the attempt: a cancelled recovery can no longer
        # flip availability or emit on an adapter the caller has already closed
        primary.release_recovery.set()
        await asyncio.sleep(0)

        assert not primary.recovery_completed
        assert availability == [False]
        assert not fallback_adapter._status[0].available
    finally:
        primary.release_recovery.set()
        await fallback_adapter.aclose()
        await primary.aclose()
        await fallback.aclose()


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
