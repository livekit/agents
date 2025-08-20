from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context, trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager
from opentelemetry.util.types import Attributes, AttributeValue

from . import trace_types

if TYPE_CHECKING:
    from livekit.agents.llm import ChatContext


class _DynamicTracer(Tracer):
    """A tracer that dynamically updates the tracer from the current trace provider.

    This ensures that when a new trace provider is configured in a function,
    subsequent tracing calls will use the new provider, rather than being bound to the
    provider that was active at import time.
    """

    def __init__(self, instrumenting_module_name: str) -> None:
        self._instrumenting_module_name = instrumenting_module_name
        self._tracer_provider = trace.get_tracer_provider()
        self._tracer = trace.get_tracer(instrumenting_module_name)

    def set_provider(self, tracer_provider: TracerProvider) -> None:
        self._tracer_provider = tracer_provider
        self._tracer = trace.get_tracer(
            self._instrumenting_module_name,
            tracer_provider=self._tracer_provider,
        )

    @property
    def current_tracer(self) -> Tracer:
        return self._tracer

    def start_span(self, *args: Any, **kwargs: Any) -> Span:
        """Start a span using the current tracer."""
        return self.current_tracer.start_span(*args, **kwargs)

    @_agnosticcontextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[Span]:
        """Start a span as current span using the current tracer."""
        with self.current_tracer.start_as_current_span(*args, **kwargs) as span:
            yield span


tracer: Tracer = _DynamicTracer("livekit-agents")


def set_tracer_provider(
    tracer_provider: TracerProvider, *, metadata: dict[str, AttributeValue] | None = None
) -> None:
    """Set the tracer provider for the livekit-agents.

    Args:
        tracer_provider (TracerProvider): The tracer provider to set.
        metadata (dict[str, AttributeValue] | None, optional): Metadata to set on all spans. Defaults to None.
    """
    assert isinstance(tracer, _DynamicTracer)

    class _MetadataSpanProcessor(SpanProcessor):
        def __init__(self, metadata: dict[str, AttributeValue]) -> None:
            self._metadata = metadata

        def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
            span.set_attributes(self._metadata)

    if metadata:
        tracer_provider.add_span_processor(_MetadataSpanProcessor(metadata))

    tracer.set_provider(tracer_provider)


def _chat_ctx_to_otel_events(chat_ctx: ChatContext) -> list[tuple[str, Attributes]]:
    role_to_event = {
        "system": trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE,
        "user": trace_types.EVENT_GEN_AI_USER_MESSAGE,
        "assistant": trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
    }

    events: list[tuple[str, Attributes]] = []
    for item in chat_ctx.items:
        if item.type == "message" and (event_name := role_to_event.get(item.role)):
            # only support text content for now
            events.append((event_name, {"content": item.text_content or ""}))
        elif item.type == "function_call":
            events.append(
                (
                    trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
                    {
                        "role": "assistant",
                        "tool_calls": [
                            json.dumps(
                                {
                                    "function": {"name": item.name, "arguments": item.arguments},
                                    "id": item.call_id,
                                    "type": "function",
                                }
                            )
                        ],
                    },
                )
            )
        elif item.type == "function_call_output":
            events.append(
                (
                    trace_types.EVENT_GEN_AI_TOOL_MESSAGE,
                    {"content": item.output, "name": item.name, "id": item.call_id},
                )
            )
    return events
