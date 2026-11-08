"""OpenTelemetry for the framework (docs/internals/telemetry.md "What Python keeps"): the API's
provider, recorded and exported by the Rust core; the bridge for an application's SDK provider;
the framework's spans around Python-only code."""

from __future__ import annotations

import asyncio
import threading
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, Status
from opentelemetry.util._decorator import _agnosticcontextmanager

from . import _core
from ._generated import facades_telemetry as _facades, public_telemetry as _public, semconv
from ._generated.public_telemetry import *  # noqa: F403

__all__: list[str] = []
__all__ += _public.__all__


async def _shutdown() -> None:
    """Flushes the process's exporter at exit, within its bound."""
    await _core._core_telemetry_shutdown()


_provider_adopted = False


def _install() -> None:
    """Starts the exporter. Once per process, takes the global tracer provider when the
    application set none, bridges an SDK provider, and reports any other as a conflict."""
    global _provider_adopted
    from .version import __version__

    _core._core_telemetry_install(__version__)
    if _provider_adopted:
        return
    _provider_adopted = True
    provider = trace.get_tracer_provider()
    if isinstance(provider, trace.ProxyTracerProvider):
        trace.set_tracer_provider(_TracerProvider())
    elif _is_sdk(provider):
        provider.add_span_processor(_bridge())  # type: ignore[attr-defined]
    elif not isinstance(provider, _TracerProvider):
        _core._core_telemetry_conflict()


class _LoopMonitor:
    """Beats the running loop's stall watch every 20 ms; the core's detector records a beat
    100 ms old as a stall, with the loop's stack (docs/internals/telemetry.md "Event-loop
    stalls"). The watch's stalls go under the span current at ``start()``."""

    _TICK = 0.02

    def __init__(self) -> None:
        self._handle: asyncio.TimerHandle | None = None
        self._watch: _core._core_LoopWatch | None = None

    def start(self, job: str | None, *, whole_process: bool) -> None:
        """``whole_process``: the process runs this job alone (the process executor), so the
        job's profile also holds the process's other running threads."""
        self._watch = _core._core_LoopWatch(
            threading.get_native_id(), parent=_current(), job=job, whole_process=whole_process
        )
        self._tick(asyncio.get_running_loop())

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._watch is not None:
            self._watch.close()
            self._watch = None

    def _tick(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._watch is not None:
            self._watch.beat()
        self._handle = loop.call_later(self._TICK, self._tick, loop)


def _context(span: SpanContext) -> otel_context.Context:
    """A context whose current span is the framework span ``span``."""
    return trace.set_span_in_context(NonRecordingSpan(span))


def _activate(span: SpanContext) -> None:
    """Makes a framework span current in the copied context one host call runs in
    (HostInvoker). Nothing detaches it: the copy ends with the call."""
    otel_context.attach(_context(span))


def _current() -> SpanContext:
    """The current span, read at a ``_core`` entry; the core ignores an invalid one."""
    return trace.get_current_span().get_span_context()


def _job_id() -> str | None:
    from .job import get_job_context

    job = get_job_context(required=False)
    return job.job.id if job is not None else None


class _Span(_facades._Span):
    def set_status(self, status: Status | trace.StatusCode, description: str | None = None) -> None:
        if isinstance(status, Status):
            self._native.set_status(status.status_code, status.description)
        else:
            self._native.set_status(status, description)

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


class _Tracer(trace.Tracer):
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
        core = _core.TelemetrySpan(
            name,
            self._scope,
            parent=trace.get_current_span(context).get_span_context(),
            kind=kind,
            start_time=start_time,
            job=_job_id(),
        )
        span = _Span(core)
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


class _TracerProvider(trace.TracerProvider):
    def get_tracer(self, instrumenting_module_name: str, *args: Any, **kwargs: Any) -> _Tracer:
        return _Tracer(instrumenting_module_name)


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
            _core._core_telemetry_export(span, job=_job_id())

    return Bridge()


# the framework's spans around Python-only code: always recorded in Rust, whatever the global
# provider is
_tracer = _Tracer("livekit-agents")


@contextmanager
def _detached_span(name: str, attributes: Mapping[str, Any]) -> Iterator[trace.Span]:
    """A span under the current one that is never made current itself, so the long-lived tasks
    started inside it (the room's) do not parent their spans under it."""
    span = _tracer.start_span(name, attributes=attributes)
    try:
        yield span
    except Exception as error:
        span.record_exception(error)
        span.set_status(trace.StatusCode.ERROR, type(error).__name__)
        raise
    finally:
        span.end()
