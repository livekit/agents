from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import llm
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _UsageLLM(llm.LLM):
    @property
    def model(self) -> str:
        return "test-model"

    @property
    def provider(self) -> str:
        return "test-provider"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _UsageLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )


class _UsageLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="request-id",
                delta=llm.ChoiceDelta(role="assistant", content="hello"),
            )
        )
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="request-id",
                usage=llm.CompletionUsage(
                    prompt_tokens=100,
                    prompt_cached_tokens=80,
                    completion_tokens=5,
                    total_tokens=105,
                ),
            )
        )


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_tracer_provider(provider)
    try:
        yield exporter
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


async def test_llm_span_reports_cached_input_tokens(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = _UsageLLM()

    response = await model.chat(chat_ctx=llm.ChatContext()).collect()

    assert response.usage is not None
    assert response.usage.prompt_cached_tokens == 80
    spans = [span for span in span_exporter.get_finished_spans() if span.name == "llm_request"]
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert spans[0].attributes["gen_ai.usage.cache_read.input_tokens"] == 80
    assert "gen_ai.usage.input_cached_tokens" not in spans[0].attributes
