from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from livekit.agents import AgentSession, APIError
from livekit.agents.llm import ChatContext
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer

from .test_typesafe_decisions import QUESTIONS, model_for, response_body

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


@pytest.mark.parametrize("session_bound", [False, True])
async def test_on_demand_decision_span_records_response_identity_and_usage(
    span_exporter, session_bound
) -> None:
    model, _ = model_for(response_body())
    evaluated_model = (
        AgentSession(decision_model=model, vad=None).decision_model if session_bound else model
    )
    with tracer.start_as_current_span("caller") as parent:
        await evaluated_model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    [span] = [s for s in span_exporter.get_finished_spans() if s.name == "decision_model.evaluate"]
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == model.model
    assert span.attributes[trace_types.ATTR_GEN_AI_RESPONSE_MODEL] == "typesafe/jev-1.13.0"
    assert span.attributes[trace_types.ATTR_GEN_AI_PROVIDER_NAME] == "openrouter"
    assert span.attributes[trace_types.ATTR_GEN_AI_RESPONSE_ID] == "request-1"
    assert span.attributes[trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 50
    assert span.attributes[trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert span.attributes["lk.decision_error_count"] == 0


@pytest.mark.parametrize("invalid_batch", [False, True])
async def test_failed_decision_evaluation_preserves_usage_and_marks_span_as_error(
    span_exporter,
    invalid_batch: bool,
) -> None:
    body = response_body()
    if invalid_batch:
        body["answers"]["extra"] = {"type": "noul", "noul": 0.5}
    else:
        body["answers"]["human"] = {"type": "noul", "noul": "invalid"}
    model, _ = model_for(body)
    with pytest.raises(APIError):
        await model.evaluate(chat_ctx=ChatContext.empty(), decisions=QUESTIONS)
    [span] = span_exporter.get_finished_spans()
    assert span.name == "decision_model.evaluate"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 50
    assert span.attributes[trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    if not invalid_batch:
        assert span.attributes["lk.decision_error_count"] == 1
