from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import llm
from livekit.agents.telemetry import gen_ai, set_tracer_provider, tracer

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


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
def limits() -> Iterator[None]:
    try:
        yield
    finally:
        gen_ai.set_capture_system_instructions(True)
        gen_ai.set_max_input_messages(0)


def _chat_ctx(turns: int) -> llm.ChatContext:
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="you are a claims assistant")
    for i in range(turns):
        chat_ctx.add_message(role="user", content=f"user {i}")
        chat_ctx.add_message(role="assistant", content=f"assistant {i}")
    return chat_ctx


def _record(chat_ctx: llm.ChatContext, **kwargs: bool) -> dict[str, object]:
    span = tracer.start_span("llm_request")
    gen_ai.set_content_attributes(
        span,
        system_instructions=gen_ai.to_system_instructions(chat_ctx),
        input_messages=gen_ai.to_input_messages(chat_ctx),
        **kwargs,
    )
    span.end()
    return dict(span.attributes or {})


def test_content_is_untruncated_by_default(
    span_exporter: InMemorySpanExporter, limits: None
) -> None:
    attributes = _record(_chat_ctx(4))

    assert json.loads(attributes["gen_ai.system_instructions"])
    assert len(json.loads(attributes["gen_ai.input.messages"])) == 8
    assert "lk.gen_ai.input.messages_dropped" not in attributes


def test_system_instructions_can_be_dropped(
    span_exporter: InMemorySpanExporter, limits: None
) -> None:
    gen_ai.set_capture_system_instructions(False)

    attributes = _record(_chat_ctx(4))

    assert "gen_ai.system_instructions" not in attributes
    assert len(json.loads(attributes["gen_ai.input.messages"])) == 8


def test_input_messages_keep_the_last_n(span_exporter: InMemorySpanExporter, limits: None) -> None:
    gen_ai.set_max_input_messages(3)

    attributes = _record(_chat_ctx(4))

    messages = json.loads(attributes["gen_ai.input.messages"])
    assert [message["parts"][0]["content"] for message in messages] == [
        "assistant 2",
        "user 3",
        "assistant 3",
    ]
    assert attributes["lk.gen_ai.input.messages_dropped"] == 5


def test_the_session_span_is_never_truncated(
    span_exporter: InMemorySpanExporter, limits: None
) -> None:
    gen_ai.set_capture_system_instructions(False)
    gen_ai.set_max_input_messages(3)

    attributes = _record(_chat_ctx(4), truncate=False)

    assert json.loads(attributes["gen_ai.system_instructions"])
    assert len(json.loads(attributes["gen_ai.input.messages"])) == 8
    assert "lk.gen_ai.input.messages_dropped" not in attributes


def test_a_malformed_message_limit_does_not_stop_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_MAX_INPUT_MESSAGES", "ten")
    assert gen_ai._env_max_input_messages() == 0
