from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents import utils
from livekit.agents.telemetry import trace_types, utils as telemetry_utils
from livekit.agents.telemetry.traces import _DynamicTracer, _TraceLevelLoggingHandler
from livekit.agents.types import NOT_GIVEN, NotGivenOr

pytestmark = pytest.mark.unit


class _FakeSpan:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.attributes: dict[str, Any] = {}
        self.recorded_exceptions: list[Exception] = []
        self.status: trace.Status | None = None

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def record_exception(self, exception: Exception) -> None:
        self.recorded_exceptions.append(exception)

    def set_status(self, status: trace.Status) -> None:
        self.status = status

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)


def _capture_exception(span: _FakeSpan, *, redacted: NotGivenOr[bool]) -> None:
    try:
        raise RuntimeError("secret transcript")
    except RuntimeError as exc:
        telemetry_utils.record_exception(span, exc, redacted=redacted)  # type: ignore[arg-type]


def test_record_exception_preserves_details_when_not_redacted() -> None:
    span = _FakeSpan()

    _capture_exception(span, redacted=False)

    assert len(span.recorded_exceptions) == 1
    assert span.attributes[trace_types.ATTR_EXCEPTION_TYPE] == "RuntimeError"
    # the GenAI/HTTP conventions' low-cardinality error identifier
    assert span.attributes[trace_types.ATTR_ERROR_TYPE] == "RuntimeError"
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == "secret transcript"
    assert "secret transcript" in span.attributes[trace_types.ATTR_EXCEPTION_TRACE]
    assert span.status is not None
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == "secret transcript"


def test_record_exception_omits_details_when_redacted() -> None:
    span = _FakeSpan()

    _capture_exception(span, redacted=True)

    assert span.recorded_exceptions == []
    # `error.type` names the exception class, never its message, so it survives redaction
    assert span.attributes == {
        trace_types.ATTR_ERROR_TYPE: "RuntimeError",
        trace_types.ATTR_EXCEPTION_TYPE: "RuntimeError",
        trace_types.ATTR_EXCEPTION_MESSAGE: telemetry_utils.REDACTED_EXCEPTION_MESSAGE,
    }
    assert trace_types.ATTR_EXCEPTION_TRACE not in span.attributes
    assert span.events == [
        (
            "exception",
            {
                trace_types.ATTR_EXCEPTION_TYPE: "RuntimeError",
                trace_types.ATTR_EXCEPTION_MESSAGE: telemetry_utils.REDACTED_EXCEPTION_MESSAGE,
            },
        )
    ]
    assert span.status is not None
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == telemetry_utils.REDACTED_EXCEPTION_MESSAGE


def test_record_exception_uses_resolved_redaction_state(monkeypatch: pytest.MonkeyPatch) -> None:
    span = _FakeSpan()

    def get_job_context(*, required: bool = True) -> SimpleNamespace:
        return SimpleNamespace(_redaction_enabled=True)

    monkeypatch.setattr("livekit.agents.job.get_job_context", get_job_context)
    _capture_exception(span, redacted=NOT_GIVEN)

    assert span.recorded_exceptions == []
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
        telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    )
    assert trace_types.ATTR_EXCEPTION_TRACE not in span.attributes


@pytest.mark.parametrize("use_span", [False, True], ids=["start_as_current_span", "use_span"])
@pytest.mark.parametrize("redaction_enabled", [False, True])
@pytest.mark.parametrize("record_exception", [False, True])
@pytest.mark.parametrize("set_status_on_exception", [False, True])
@pytest.mark.parametrize("end_on_exit", [False, True])
@pytest.mark.parametrize("call_style", ["keyword", "positional", "mixed"])
def test_dynamic_tracer_records_exceptions_with_caller_options(
    monkeypatch: pytest.MonkeyPatch,
    use_span: bool,
    redaction_enabled: bool,
    record_exception: bool,
    set_status_on_exception: bool,
    end_on_exit: bool,
    call_style: str,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-exception-redaction")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)
    options = {
        "record_exception": record_exception,
        "set_status_on_exception": set_status_on_exception,
        "end_on_exit": end_on_exit,
    }
    if call_style == "keyword":
        manager = (
            dynamic_tracer.use_span(dynamic_tracer.start_span("test-span"), **options)
            if use_span
            else dynamic_tracer.start_as_current_span("test-span", **options)
        )
    else:
        args: list[Any]
        if use_span:
            context_manager = dynamic_tracer.use_span
            args = [dynamic_tracer.start_span("test-span"), end_on_exit, record_exception]
            options = {"set_status_on_exception": set_status_on_exception}
        else:
            context_manager = dynamic_tracer.start_as_current_span
            args = ["test-span", None, trace.SpanKind.INTERNAL, None, None, None, record_exception]
            options = {
                "set_status_on_exception": set_status_on_exception,
                "end_on_exit": end_on_exit,
            }
        if call_style == "positional":
            args.extend(options.values())
            options = {}
        manager = context_manager(*args, **options)
    previous_span = trace.get_current_span()
    failure = RuntimeError("secret transcript")
    try:
        with pytest.raises(RuntimeError) as raised:
            with manager as span:
                assert trace.get_current_span() is span
                raise failure
        assert raised.value is failure
        assert trace.get_current_span() is previous_span
        assert span.is_recording() is (not end_on_exit)
        if not end_on_exit:
            span.end()

        (finished,) = exporter.get_finished_spans()
        events = [event for event in finished.events if event.name == "exception"]
        assert len(events) == int(record_exception)
        message = (
            telemetry_utils.REDACTED_EXCEPTION_MESSAGE if redaction_enabled else "secret transcript"
        )
        if record_exception:
            assert events[0].attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == message
        assert finished.status.status_code is (
            trace.StatusCode.ERROR if set_status_on_exception else trace.StatusCode.UNSET
        )
        assert finished.status.description == (message if set_status_on_exception else None)
        attrs = finished.attributes or {}
        assert attrs.get(trace_types.ATTR_ERROR_TYPE) == (
            "RuntimeError" if record_exception or set_status_on_exception else None
        )
        if not record_exception:
            assert trace_types.ATTR_EXCEPTION_MESSAGE not in attrs
        if redaction_enabled:
            assert "secret transcript" not in finished.to_json()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("use_span", [False, True], ids=["start_as_current_span", "use_span"])
@pytest.mark.parametrize("redaction_enabled", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_dynamic_tracer_decorator_preserves_logging_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    use_span: bool,
    redaction_enabled: bool,
    cancelled: bool,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-exception-decorator")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)
    manager = (
        dynamic_tracer.use_span(dynamic_tracer.start_span("test-span"), end_on_exit=True)
        if use_span
        else dynamic_tracer.start_as_current_span("test-span")
    )
    logger = logging.getLogger("test.exception-decorator")
    failure = asyncio.CancelledError() if cancelled else RuntimeError("secret transcript")

    @utils.log_exceptions(logger=logger)
    @manager
    async def failing_task() -> None:
        await asyncio.sleep(0)
        raise failure

    try:
        with pytest.raises(type(failure)) as raised:
            await failing_task()
        assert raised.value is failure
        (span,) = exporter.get_finished_spans()
        events = [event for event in span.events if event.name == "exception"]
        logs = [record for record in caplog.records if record.name == logger.name]
        assert len(events) == len(logs) == (0 if cancelled else 1)
        assert span.status.status_code is (
            trace.StatusCode.UNSET if cancelled else trace.StatusCode.ERROR
        )
        if not cancelled:
            assert logs[0].getMessage() == "Error in failing_task"
            assert logs[0].exc_info[1] is failure
    finally:
        provider.shutdown()


@pytest.mark.parametrize("redaction_enabled", [False, True])
def test_logging_handler_omits_automatic_exception_details_when_redacted(
    monkeypatch: pytest.MonkeyPatch, redaction_enabled: bool
) -> None:
    try:
        raise RuntimeError("secret transcript")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="livekit.agents.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="operation failed",
        args=(),
        exc_info=exc_info,
    )
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)

    translated = _TraceLevelLoggingHandler()._translate(record)
    assert translated.attributes is not None
    assert translated.attributes[trace_types.ATTR_EXCEPTION_TYPE] == "RuntimeError"
    if redaction_enabled:
        assert translated.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
            telemetry_utils.REDACTED_EXCEPTION_MESSAGE
        )
        assert trace_types.ATTR_EXCEPTION_TRACE not in translated.attributes
    else:
        assert translated.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == "secret transcript"
        assert "secret transcript" in translated.attributes[trace_types.ATTR_EXCEPTION_TRACE]


def test_redacted_exception_log_can_be_otlp_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify redacted exception logs remain encodable by OTLP."""
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    handler = _TraceLevelLoggingHandler(logger_provider=provider)

    try:
        raise RuntimeError("secret transcript")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="livekit.agents.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="operation failed",
        args=(),
        exc_info=exc_info,
    )
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: True)

    try:
        handler.emit(record)
        (exported_log,) = exporter.get_finished_logs()
        encoded_request = encode_logs([exported_log])
    finally:
        provider.shutdown()

    (encoded_log,) = encoded_request.resource_logs[0].scope_logs[0].log_records
    encoded_attributes = {
        attribute.key: attribute.value.string_value for attribute in encoded_log.attributes
    }
    assert encoded_attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
        telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    )
    assert trace_types.ATTR_EXCEPTION_TRACE not in encoded_attributes
    assert encoded_log.dropped_attributes_count == 0
    assert "secret transcript" not in str(encoded_log)
