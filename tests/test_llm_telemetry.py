from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

from livekit.agents import llm
from livekit.agents.telemetry import gen_ai, set_tracer_provider, trace_types, tracer
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.voice import generation
from livekit.agents.voice.io import ModelSettings

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
    _response_content: str = "hello"

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id="request-id",
                delta=llm.ChoiceDelta.model_construct(
                    role="assistant", content=self._response_content
                ),
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


class _CaptureToggleLLMStream(_UsageLLMStream):
    capture_content_during_run: bool = False

    async def _run(self) -> None:
        gen_ai.set_capture_content(self.capture_content_during_run)
        await super()._run()


class _ResponseContentProbe(str):
    was_accumulated: bool = False

    def __radd__(self, other: object) -> str:
        self.was_accumulated = True
        return f"{other}{self}"


def _custom_llm_node(
    chat_ctx: llm.ChatContext,
    tools: list[llm.Tool],
    model_settings: ModelSettings,
) -> str:
    return "hello"


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


@pytest.fixture
def nonrecording_tracer_provider() -> Iterator[None]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider(sampler=ALWAYS_OFF)
    set_tracer_provider(provider)
    try:
        yield
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


def _forbid_content_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_builder(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("content payload builder was called")

    for name in (
        "to_system_instructions",
        "to_input_messages",
        "to_output_messages",
        "to_tool_definitions",
    ):
        monkeypatch.setattr(gen_ai, name, unexpected_builder)


async def test_llm_span_reports_cached_input_tokens(
    span_exporter: InMemorySpanExporter,
) -> None:
    model = _UsageLLM()
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")

    response = await model.chat(chat_ctx=chat_ctx).collect()

    assert response.usage is not None
    assert response.usage.prompt_cached_tokens == 80
    spans = [span for span in span_exporter.get_finished_spans() if span.name == "llm_request"]
    assert len(spans) == 1
    assert spans[0].attributes["gen_ai.usage.input_tokens"] == 100
    assert spans[0].attributes["gen_ai.usage.cache_read.input_tokens"] == 80
    # both spellings: the registry one for the convention, and the unofficial one Langfuse
    # and Datadog key on. The realtime path emits both, so the two paths now agree.
    assert spans[0].attributes["gen_ai.usage.input_cached_tokens"] == 80
    assert json.loads(spans[0].attributes["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]}
    ]
    assert json.loads(spans[0].attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hello"}],
            "finish_reason": "stop",
        }
    ]


@pytest.mark.parametrize(
    ("capture_at_start", "capture_during_run"),
    [
        (False, True),
        (True, False),
    ],
)
async def test_llm_stream_capture_requires_enablement_at_start_and_completion(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
    capture_at_start: bool,
    capture_during_run: bool,
) -> None:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="hello")
    gen_ai.set_capture_content(capture_at_start)
    try:
        monkeypatch.setattr(
            _CaptureToggleLLMStream, "capture_content_during_run", capture_during_run
        )
        stream = _CaptureToggleLLMStream(
            _UsageLLM(),
            chat_ctx=chat_ctx,
            tools=[],
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        response = await stream.collect()
    finally:
        gen_ai.set_capture_content(True)

    assert response.text == "hello"
    spans = [span for span in span_exporter.get_finished_spans() if span.name == "llm_request"]
    assert len(spans) == 1
    assert (trace_types.ATTR_GEN_AI_INPUT_MESSAGES in spans[0].attributes) is capture_at_start
    assert (trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES in spans[0].attributes) is (
        capture_at_start and capture_during_run
    )


async def test_llm_stream_skips_content_builders_when_capture_is_disabled(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_content_builders(monkeypatch)
    gen_ai.set_capture_content(False)
    try:
        content = _ResponseContentProbe("hello")
        monkeypatch.setattr(_UsageLLMStream, "_response_content", content)
        response = await _UsageLLM().chat(chat_ctx=llm.ChatContext.empty()).collect()
    finally:
        gen_ai.set_capture_content(True)

    assert not content.was_accumulated
    assert response.text == "hello"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 100
    spans = [span for span in span_exporter.get_finished_spans() if span.name == "llm_request"]
    assert len(spans) == 1
    assert spans[0].attributes[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == "test-model"
    assert spans[0].attributes[trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 100
    assert trace_types.ATTR_GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes


async def test_llm_stream_skips_content_builders_for_nonrecording_span(
    nonrecording_tracer_provider: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_content_builders(monkeypatch)

    content = _ResponseContentProbe("hello")
    monkeypatch.setattr(_UsageLLMStream, "_response_content", content)
    response = await _UsageLLM().chat(chat_ctx=llm.ChatContext.empty()).collect()

    assert not content.was_accumulated
    assert response.text == "hello"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 100


async def test_llm_node_skips_payloads_for_nonrecording_span(
    nonrecording_tracer_provider: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_content_builders(monkeypatch)

    def unexpected_to_dict(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("chat context was serialized")

    monkeypatch.setattr(llm.ChatContext, "to_dict", unexpected_to_dict)

    task, data = generation.perform_llm_inference(
        node=_custom_llm_node,
        chat_ctx=llm.ChatContext.empty(),
        tool_ctx=llm.ToolContext([]),
        model_settings=ModelSettings(),
    )

    assert await task is True
    assert data.generated_text == "hello"


async def test_llm_node_preserves_noncontent_attributes_when_capture_is_disabled(
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_content_builders(monkeypatch)
    gen_ai.set_capture_content(False)
    try:
        task, _ = generation.perform_llm_inference(
            node=_custom_llm_node,
            chat_ctx=llm.ChatContext.empty(),
            tool_ctx=llm.ToolContext([]),
            model_settings=ModelSettings(),
        )
        assert await task is True
    finally:
        gen_ai.set_capture_content(True)

    spans = [span for span in span_exporter.get_finished_spans() if span.name == "llm_node"]
    assert len(spans) == 1
    assert trace_types.ATTR_CHAT_CTX in spans[0].attributes
    assert spans[0].attributes[trace_types.ATTR_GEN_AI_OPERATION_NAME] == "chat"
    assert trace_types.ATTR_GEN_AI_INPUT_MESSAGES not in spans[0].attributes
    assert trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES not in spans[0].attributes
