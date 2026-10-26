"""The keyterm-detection LLM call is STT context for later turns, not part of any reply: it
gets its own ``keyterm_detection`` span under ``agent_session`` instead of landing under the
``agent_turn`` whose task fired the conversation event."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.llm import ChatContext
from livekit.agents.telemetry import set_tracer_provider, trace_types, tracer
from livekit.agents.voice import keyterm_detection
from livekit.agents.voice.keyterm_detection import KeytermDetector

from .fake_llm import FakeLLM

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


def _detector(monkeypatch: pytest.MonkeyPatch) -> tuple[KeytermDetector, object]:
    async def fake_detect(**kwargs: object) -> tuple[list[str], list[str], list[str]]:
        return [], ["Acme", "LiveKit"], []

    monkeypatch.setattr(keyterm_detection, "_detect_keyterms", fake_detect)
    detector = KeytermDetector(static_keyterms=["Zed"])
    detector._llm = FakeLLM()  # type: ignore[assignment]
    root = tracer.start_span("agent_session")
    detector._session = SimpleNamespace(  # type: ignore[assignment]
        _root_span_context=trace.set_span_in_context(root),
        history=ChatContext.empty(),
    )
    return detector, root


async def test_detection_pass_nests_under_the_replying_agent_turn(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The conversation event fires from the reply that answers the user message: the pass is
    that agent's work on the turn, so it nests under the agent_turn."""
    detector, root = _detector(monkeypatch)
    from livekit.agents.llm import ChatMessage
    from livekit.agents.voice.events import ConversationItemAddedEvent

    with tracer.start_as_current_span("agent_turn") as turn:
        detector._on_conversation_item_added(
            ConversationItemAddedEvent(item=ChatMessage(role="user", content=["order for Acme"]))
        )
        assert detector._detect_task is not None
        await detector._detect_task
    root.end()  # type: ignore[attr-defined]

    [span] = [s for s in span_exporter.get_finished_spans() if s.name == "keyterm_detection"]
    assert span.parent is not None and span.parent.span_id == turn.get_span_context().span_id
    attrs = span.attributes or {}
    assert attrs[trace_types.ATTR_KEYTERMS_COUNT] == 3  # Zed + Acme + LiveKit
    assert attrs[trace_types.ATTR_KEYTERMS_ADDED] == 2
    assert attrs[trace_types.ATTR_KEYTERMS_REMOVED] == 0
    assert attrs[trace_types.ATTR_GEN_AI_REQUEST_MODEL] == detector._llm.model
    assert detector.keyterms == ["Zed", "Acme", "LiveKit"]


async def test_detection_pass_without_a_reply_falls_back_to_the_session(
    span_exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector, root = _detector(monkeypatch)
    # no span current: user code edited the history, or the reply was skipped
    await detector._run_once(ChatContext.empty(), detector._session._root_span_context)  # type: ignore[union-attr]
    root.end()  # type: ignore[attr-defined]
    [span] = [s for s in span_exporter.get_finished_spans() if s.name == "keyterm_detection"]
    assert span.parent is not None
    assert span.parent.span_id == root.get_span_context().span_id  # type: ignore[attr-defined]
