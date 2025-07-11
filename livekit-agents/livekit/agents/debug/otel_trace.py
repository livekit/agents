import base64
import os
from collections.abc import Iterator, Sequence

from opentelemetry import trace
from opentelemetry.context.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Link, Span, SpanKind, Tracer
from opentelemetry.util import types
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

    def start_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> Span:
        """Start a span using the current tracer."""
        return self.current_tracer.start_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[Span]:
        """Start a span as current span using the current tracer."""
        with self.current_tracer.start_as_current_span(
            name=name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        ) as span:
            yield span


tracer = _DynamicTracer("livekit-agents")


def set_tracer_provider(tracer_provider: TracerProvider) -> None:
    tracer.set_provider(tracer_provider)


def use_trace_langfuse(
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_host: str | None = None,
) -> Tracer:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    langfuse_public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = langfuse_host or os.getenv("LANGFUSE_HOST")

    if not langfuse_public_key or not langfuse_secret_key or not langfuse_host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST must be set")

    langfuse_auth = base64.b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode()
    ).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{langfuse_host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    tracer.set_provider(trace_provider)

    return tracer
