"""OpenTelemetry for the framework (docs/internals/telemetry.md "What Python keeps"): the API's
provider, recorded and exported by the Rust core; the bridge for an application's SDK provider;
the framework's spans around Python-only code."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, SpanKind, Status, TraceFlags
from opentelemetry.util._decorator import _agnosticcontextmanager

from livekit import rtc

from . import _core
from ._generated import semconv
from ._generated.telemetry import SpanContext as _SpanContext, TelemetryDiagnostics

__all__ = ["configure", "diagnostics", "set_capture_content"]


def configure(
    *,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_pii: bool | None = None,
) -> None:
    """Send the process's spans and metrics to an OTLP/HTTP protobuf endpoint; arguments override
    the ``OTEL_EXPORTER_OTLP_*`` environment."""
    _core._core_telemetry_configure(
        endpoint=endpoint,
        headers=dict(headers) if headers else None,
        metadata=dict(metadata) if metadata else None,
        allow_pii=allow_pii,
    )


def set_capture_content(enabled: bool) -> None:
    """Whether model request and response messages and judge explanations are recorded."""
    _core._core_set_capture_content(enabled)


def diagnostics() -> TelemetryDiagnostics:
    """The process's export counters."""
    return _core._core_telemetry_diagnostics()


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
    """Measures the running loop's stalls: a tick due every ``_TICK`` seconds that runs late by
    ``_THRESHOLD`` or more is a stall the core records (its measurement, and a rate-limited
    ``event_loop_blocked`` span under the span current at ``start()``)."""

    _TICK = 0.02
    _THRESHOLD = 0.1

    def __init__(self) -> None:
        self._handle: asyncio.TimerHandle | None = None
        self._parent: _SpanContext | None = None
        self._job: str | None = None

    def start(self, job: str | None) -> None:
        self._parent = _current()
        self._job = job
        self._arm(asyncio.get_running_loop())

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _arm(self, loop: asyncio.AbstractEventLoop) -> None:
        due = loop.time() + self._TICK
        self._handle = loop.call_at(due, self._tick, loop, due)

    def _tick(self, loop: asyncio.AbstractEventLoop, due: float) -> None:
        if (late := loop.time() - due) >= self._THRESHOLD:
            _core._core_event_loop_blocked(late, parent=self._parent, job=self._job)
        self._arm(loop)


def _context(span: _SpanContext) -> otel_context.Context:
    """A context whose current span is the framework span ``span``."""
    flags = TraceFlags(TraceFlags.SAMPLED if span.sampled else TraceFlags.DEFAULT)
    ids = SpanContext(span.trace_id, span.span_id, is_remote=False, trace_flags=flags)
    return trace.set_span_in_context(NonRecordingSpan(ids))


def _activate(trace_id: int, span_id: int, sampled: bool) -> None:
    """Makes a framework span current in the copied context one host call runs in
    (HostInvoker). Nothing detaches it: the copy ends with the call."""
    span = _SpanContext(trace_id=trace_id, span_id=span_id, sampled=sampled)
    otel_context.attach(_context(span))


def _current() -> _SpanContext | None:
    """The current span, read at a ``_core`` entry."""
    return _parent(trace.get_current_span().get_span_context())


def _parent(ctx: SpanContext | None) -> _SpanContext | None:
    if ctx is None or not ctx.is_valid:
        return None
    return _SpanContext(trace_id=ctx.trace_id, span_id=ctx.span_id, sampled=ctx.trace_flags.sampled)


def _job_id() -> str | None:
    from .job import get_job_context

    job = get_job_context(required=False)
    return job.job.id if job is not None else None


def _value(value: Any) -> Any:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str) else value


class _Span(trace.Span):
    def __init__(self, core: _core.TelemetrySpan) -> None:
        self._core = core
        ctx = core.context
        flags = TraceFlags(TraceFlags.SAMPLED if ctx.sampled else TraceFlags.DEFAULT)
        self._context = SpanContext(ctx.trace_id, ctx.span_id, is_remote=False, trace_flags=flags)

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
            self._core.set_status(_STATUS[status.status_code], status.description)
        else:
            self._core.set_status(_STATUS[status], description)

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


_KINDS = {
    SpanKind.INTERNAL: _core.SpanKind.Internal,
    SpanKind.SERVER: _core.SpanKind.Server,
    SpanKind.CLIENT: _core.SpanKind.Client,
    SpanKind.PRODUCER: _core.SpanKind.Producer,
    SpanKind.CONSUMER: _core.SpanKind.Consumer,
}
_STATUS = {
    trace.StatusCode.UNSET: _core.StatusCode.Unset,
    trace.StatusCode.OK: _core.StatusCode.Ok,
    trace.StatusCode.ERROR: _core.StatusCode.Error,
}


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
            parent=_parent(trace.get_current_span(context).get_span_context()),
            kind=_KINDS[kind],
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
            if (ctx := span.get_span_context()) is None:
                return
            core = _core.TelemetrySpan(
                span.name,
                span.instrumentation_scope.name if span.instrumentation_scope else "",
                parent=_parent(span.parent),
                kind=_KINDS[span.kind],
                start_time=span.start_time,
                ids=_parent(ctx),
                job=_job_id(),
            )
            for key, value in (span.attributes or {}).items():
                core.set_attribute(key, _value(value))
            for event in span.events:
                attrs = {k: _value(v) for k, v in (event.attributes or {}).items()}
                core.add_event(event.name, attrs, event.timestamp)
            core.set_status(_STATUS[span.status.status_code], span.status.description)
            core.end(span.end_time)

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


_RPC_PAYLOAD_CHARS = 1024
_CANCELLED = {
    rtc.RpcError.ErrorCode.RESPONSE_TIMEOUT: "response timeout",
    rtc.RpcError.ErrorCode.RECIPIENT_DISCONNECTED: "caller disconnected",
    rtc.RpcError.ErrorCode.APPLICATION_ERROR: "handler cancelled",
}


def _payload(size_key: str, key: str, payload: str | None) -> dict[str, Any]:
    payload = payload or ""
    attrs: dict[str, Any] = {size_key: len(payload.encode())}
    if payload:
        attrs[key] = payload[:_RPC_PAYLOAD_CHARS]
    return attrs


class _RpcTracing(rtc.RpcInterceptor):
    """``rpc_call`` and ``rpc_handler`` spans for the job's local participant."""

    async def intercept_outgoing(
        self, call: rtc.RpcCallInfo, next: Callable[[rtc.RpcCallInfo], Awaitable[str]]
    ) -> str:
        attrs = {
            semconv.RPC_METHOD: call.method,
            semconv.LK_RPC_DESTINATION_IDENTITY: call.destination_identity,
            **_payload(semconv.LK_RPC_PAYLOAD_SIZE, semconv.LK_PII_RPC_PAYLOAD, call.payload),
        }
        if call.response_timeout is not None:
            attrs[semconv.LK_RPC_RESPONSE_TIMEOUT] = call.response_timeout
        with _tracer.start_as_current_span(
            semconv.SPAN_RPC_CALL, kind=SpanKind.CLIENT, attributes=attrs
        ) as span:
            try:
                response = await next(call)
            except rtc.RpcError as e:
                span.set_attribute(semconv.LK_RPC_ERROR_CODE, int(e.code))
                raise
            span.set_attributes(
                _payload(semconv.LK_RPC_RESPONSE_SIZE, semconv.LK_PII_RPC_RESPONSE, response)
            )
            return response

    async def intercept_incoming(
        self,
        invocation: rtc.RpcInvocationData,
        next: Callable[[rtc.RpcInvocationData], Awaitable[str | None]],
    ) -> str | None:
        attrs = {
            semconv.RPC_METHOD: invocation.method,
            semconv.LK_RPC_REQUEST_ID: invocation.request_id,
            semconv.LK_RPC_CALLER_IDENTITY: invocation.caller_identity,
            semconv.LK_RPC_RESPONSE_TIMEOUT: invocation.response_timeout,
            semconv.LK_RPC_HANDLER_REGISTERED: True,
            **_payload(semconv.LK_RPC_PAYLOAD_SIZE, semconv.LK_PII_RPC_PAYLOAD, invocation.payload),
        }
        with _tracer.start_as_current_span(
            semconv.SPAN_RPC_HANDLER,
            context=_session_context(),
            kind=SpanKind.SERVER,
            attributes=attrs,
        ) as span:
            try:
                response = await next(invocation)
            except rtc.RpcError as e:
                span.set_attribute(semconv.LK_RPC_ERROR_CODE, int(e.code))
                if e.code == rtc.RpcError.ErrorCode.UNSUPPORTED_METHOD:
                    span.set_attribute(semconv.LK_RPC_HANDLER_REGISTERED, False)
                raise
            except asyncio.CancelledError:
                # not an Exception, so the span would end UNSET; the SDK answers a cancellation
                # from inside the chain (no cancel_reason) as APPLICATION_ERROR
                code = invocation.cancel_reason or rtc.RpcError.ErrorCode.APPLICATION_ERROR
                span.set_attribute(semconv.LK_RPC_ERROR_CODE, int(code))
                span.set_status(trace.StatusCode.ERROR, _CANCELLED.get(code, "cancelled"))
                raise
            span.set_attributes(
                _payload(semconv.LK_RPC_RESPONSE_SIZE, semconv.LK_PII_RPC_RESPONSE, response)
            )
            return response


def _session_context() -> otel_context.Context | None:
    """The job's primary session's span, which a remote participant's RPC is handled under."""
    from .job import get_job_context

    job = get_job_context(required=False)
    session = job._primary_agent_session if job is not None else None
    return session._root_span_context if session is not None else None


def _trace_rpcs(participant: rtc.LocalParticipant) -> None:
    participant.add_rpc_interceptor(_RpcTracing())
