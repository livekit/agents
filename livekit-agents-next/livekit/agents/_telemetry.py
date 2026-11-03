"""OpenTelemetry for the framework: the API's provider, recorded and exported by the Rust core
(docs/internals/telemetry.md "What Python keeps"). Becomes ``telemetry.py`` once the
``telemetry/`` package is gone."""

from __future__ import annotations

import traceback
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from opentelemetry import context as otel_context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, Status, TraceFlags
from opentelemetry.util._decorator import _agnosticcontextmanager

from . import _core
from ._generated import semconv

__all__ = ["configure", "diagnostics", "set_capture_content"]


def configure(
    *,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    protocol: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Send the process's spans and metrics to an OTLP/HTTP endpoint; arguments override the
    ``OTEL_EXPORTER_OTLP_*`` environment."""
    _core._core_telemetry_configure(
        endpoint=endpoint,
        headers=dict(headers) if headers else None,
        protocol=protocol,
        metadata=dict(metadata) if metadata else None,
        allow_pii=allow_pii,
    )


def set_capture_content(enabled: bool) -> None:
    """Whether model request and response messages and judge explanations are recorded."""
    _core._core_set_capture_content(enabled)


def diagnostics() -> dict[str, int]:
    """The process's export counters: dropped, failed, conflicts, incomplete_cleanup."""
    return dict(_core._core_telemetry_diagnostics())


def install() -> None:
    """Starts the exporter; takes the global tracer provider when the application set none,
    bridges an SDK provider, and reports any other as a conflict."""
    from .version import __version__

    _core._core_telemetry_install(__version__)
    provider = trace.get_tracer_provider()
    if isinstance(provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(TracerProvider())
    elif isinstance(provider, TracerProvider) or getattr(provider, "_lk_bridged", False):
        return
    elif _is_sdk(provider):
        provider.add_span_processor(_bridge())  # type: ignore[attr-defined]
        provider._lk_bridged = True  # type: ignore[attr-defined]
    else:
        _core._core_telemetry_conflict()


def _activate(trace_id: int, span_id: int, sampled: bool) -> None:
    """Makes a framework span current in the running context (HostInvoker's host calls)."""
    flags = TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)
    span = NonRecordingSpan(SpanContext(trace_id, span_id, is_remote=False, trace_flags=flags))
    otel_context.attach(trace.set_span_in_context(span))


def _current() -> tuple[int, int, bool]:
    """The current span, read at a ``_core`` entry; zeros when there is none."""
    ctx = trace.get_current_span().get_span_context()
    return (ctx.trace_id, ctx.span_id, ctx.trace_flags.sampled) if ctx.is_valid else (0, 0, False)


def _job_id() -> str | None:
    from .job import get_job_context

    job = get_job_context(required=False)
    return job.job.id if job is not None else None


def _value(value: Any) -> Any:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else value


class Span(trace.Span):
    def __init__(self, core: _core.TelemetrySpan) -> None:
        self._core = core
        trace_id, span_id, sampled = core.context
        flags = TraceFlags(TraceFlags.SAMPLED if sampled else TraceFlags.DEFAULT)
        self._context = SpanContext(trace_id, span_id, is_remote=False, trace_flags=flags)

    def get_span_context(self) -> SpanContext:
        return self._context

    def end(self, end_time: int | None = None) -> None:
        self._core.end(end_time)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        for key, value in attributes.items():
            self._core.set_attribute(key, _value(value))

    def set_attribute(self, key: str, value: Any) -> None:
        self._core.set_attribute(key, _value(value))

    def add_event(
        self, name: str, attributes: Mapping[str, Any] | None = None, timestamp: int | None = None
    ) -> None:
        attrs = {k: _value(v) for k, v in attributes.items()} if attributes else None
        self._core.add_event(name, attrs, timestamp)

    def update_name(self, name: str) -> None:
        self._core.update_name(name)

    def is_recording(self) -> bool:
        return self._core.is_recording()

    def set_status(self, status: Status | trace.StatusCode, description: str | None = None) -> None:
        if isinstance(status, Status):
            self._core.set_status(status.status_code.value, status.description)
        else:
            self._core.set_status(status.value, description)

    def record_exception(
        self,
        exception: BaseException,
        attributes: Mapping[str, Any] | None = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        stack = "".join(traceback.format_exception(type(exception), exception, None))
        event = {
            semconv.EXCEPTION_TYPE: type(exception).__name__,
            semconv.EXCEPTION_MESSAGE: str(exception),
            semconv.EXCEPTION_STACKTRACE: stack,
        }
        self.add_event(semconv.EVENT_EXCEPTION, {**event, **(attributes or {})}, timestamp)


class Tracer(trace.Tracer):
    def __init__(self, scope: str) -> None:
        self._scope = scope

    def start_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[trace.Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> trace.Span:
        parent = trace.get_current_span(context).get_span_context()
        core = _core.TelemetrySpan(
            name,
            self._scope,
            parent=(parent.trace_id, parent.span_id, parent.trace_flags.sampled)
            if parent.is_valid
            else None,
            kind=kind.value,
            start_time=start_time,
            job=_job_id(),
        )
        span = Span(core)
        if attributes:
            span.set_attributes(attributes)
        return span

    @_agnosticcontextmanager
    def start_as_current_span(
        self,
        name: str,
        context: otel_context.Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, Any] | None = None,
        links: Sequence[trace.Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[trace.Span]:
        span = self.start_span(name, context, kind, attributes, links, start_time)
        with trace.use_span(
            span, end_on_exit, record_exception, set_status_on_exception
        ) as current:
            yield current


class TracerProvider(trace.TracerProvider):
    def get_tracer(self, instrumenting_module_name: str, *args: Any, **kwargs: Any) -> Tracer:
        return Tracer(instrumenting_module_name)


def _is_sdk(provider: trace.TracerProvider) -> bool:
    try:
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
    except ImportError:
        return False
    return isinstance(provider, SdkTracerProvider)


def _bridge() -> Any:
    """A processor forwarding each ended SDK span to the Rust exporter."""
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    class Bridge(SpanProcessor):
        def on_end(self, span: ReadableSpan) -> None:
            ctx, parent = span.get_span_context(), span.parent
            if ctx is None:
                return
            core = _core.TelemetrySpan(
                span.name,
                span.instrumentation_scope.name if span.instrumentation_scope else "",
                parent=(parent.trace_id, parent.span_id, parent.trace_flags.sampled)
                if parent is not None
                else None,
                kind=span.kind.value,
                start_time=span.start_time,
                ids=(ctx.trace_id, ctx.span_id),
                job=_job_id(),
            )
            for key, value in (span.attributes or {}).items():
                core.set_attribute(key, _value(value))
            for event in span.events:
                attrs = {k: _value(v) for k, v in (event.attributes or {}).items()}
                core.add_event(event.name, attrs, event.timestamp)
            core.set_status(span.status.status_code.value, span.status.description)
            core.end(span.end_time)

    return Bridge()
