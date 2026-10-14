"""Context fields on error logs: the failed-turn inference ERROR and ``log_exceptions``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from livekit.agents import llm, utils
from livekit.agents.telemetry import set_tracer_provider, tracer
from livekit.agents.voice import generation
from livekit.agents.voice.agent_activity import _SpeechHandleContextVar
from livekit.agents.voice.io import ModelSettings
from livekit.agents.voice.speech_handle import SpeechHandle

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.fixture
def tracer_provider() -> Iterator[None]:
    original_provider = tracer._tracer_provider
    provider = TracerProvider()
    set_tracer_provider(provider)
    try:
        yield
    finally:
        set_tracer_provider(original_provider)
        provider.shutdown()


class _SpanCapturingHandler(logging.Handler):
    """Records the span that is current when each log is emitted, as the OTel handler sees it."""

    def __init__(self) -> None:
        super().__init__()
        self.span_ids: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        self.span_ids[record.getMessage()] = trace.get_current_span().get_span_context().span_id


async def test_llm_inference_error_log_has_speech_model_and_span(
    caplog: pytest.LogCaptureFixture, tracer_provider: None
) -> None:
    node_span_ids: list[int] = []

    async def failing_node(
        chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterator[str]:
        node_span_ids.append(trace.get_current_span().get_span_context().span_id)
        raise RuntimeError("boom")
        yield ""

    handler = _SpanCapturingHandler()
    agents_logger = logging.getLogger("livekit.agents")
    agents_logger.addHandler(handler)
    speech = SpeechHandle.create()
    token = _SpeechHandleContextVar.set(speech)
    try:
        task, _ = generation.perform_llm_inference(
            node=failing_node,
            chat_ctx=llm.ChatContext.empty(),
            tool_ctx=llm.ToolContext([]),
            model_settings=ModelSettings(),
            model="gpt-test",
            provider="openai",
        )
        with pytest.raises(RuntimeError):
            await task
    finally:
        _SpeechHandleContextVar.reset(token)
        agents_logger.removeHandler(handler)

    [record] = [r for r in caplog.records if r.getMessage() == "Error in _llm_inference_task"]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.speech_id == speech.id
    assert record.model == "gpt-test"
    assert record.provider == "openai"
    # the log is emitted inside the llm_node span, so it links to that span
    assert handler.span_ids["Error in _llm_inference_task"] == node_span_ids[0] != 0


async def test_log_exceptions_attaches_extra(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("livekit.agents.test_error_log_context")

    @utils.log_exceptions(logger=logger, extra={"agent_label": "my_agent"})
    async def _traceable_on_enter() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _traceable_on_enter()

    [record] = [r for r in caplog.records if r.getMessage() == "Error in _traceable_on_enter"]
    assert record.agent_label == "my_agent"
