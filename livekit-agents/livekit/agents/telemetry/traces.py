from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer
from opentelemetry.util._decorator import _agnosticcontextmanager


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


def set_tracer_provider(tracer_provider: TracerProvider) -> None:
    assert isinstance(tracer, _DynamicTracer)
    tracer.set_provider(tracer_provider)
