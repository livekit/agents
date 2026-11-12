"""Fallback crosses the public provider bridge without replaying partial output."""

from typing import Any

import pytest

from livekit.agents import APIError
from livekit.agents.llm import (
    AvailabilityChangedEvent,
    ChatChunk,
    ChatContext,
    ChoiceDelta,
    FallbackAdapter,
    LLMStream,
    Tool,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from .fake_llm import FakeLLM

pytestmark = pytest.mark.unit


class ScriptedLLM(FakeLLM):
    def __init__(self, text: str | None, *, fail: bool = False) -> None:
        super().__init__()
        self.text = text
        self.fail = fail
        self.requests = 0

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        tools: list[Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> LLMStream:
        self.requests += 1
        return ScriptedStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class ScriptedStream(LLMStream):
    def __init__(self, provider: ScriptedLLM, **kwargs: Any) -> None:
        super().__init__(provider, tools=kwargs.pop("tools", None) or [], **kwargs)
        self.provider = provider

    async def _run(self) -> None:
        if self.provider.text is not None:
            self.send(ChatChunk(id="chunk-1", delta=ChoiceDelta(content=self.provider.text)))
        if self.provider.fail:
            raise APIError("provider failed", retryable=True)


async def test_repeated_plugin_entries_share_one_health_transition() -> None:
    primary, backup = ScriptedLLM(None, fail=True), ScriptedLLM("Recovered.")
    adapter = FallbackAdapter([primary, primary, backup], retry_interval=0, sticky=True)
    changes: list[tuple[object, bool]] = []

    @adapter.on("llm_availability_changed")
    def changed(event: AvailabilityChangedEvent) -> None:
        changes.append((event.llm, event.available))

    try:
        async with adapter.chat(chat_ctx=ChatContext.empty()) as stream:
            response = await stream.collect()
        assert response.text == "Recovered."
    finally:
        await adapter.aclose()
    assert changes == [(primary, False)], "repeated references must share the provider's health"
