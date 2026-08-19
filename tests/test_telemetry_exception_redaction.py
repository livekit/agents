from __future__ import annotations

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
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == "secret transcript"
    assert "secret transcript" in span.attributes[trace_types.ATTR_EXCEPTION_TRACE]
    assert span.status is not None
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == "secret transcript"


def test_record_exception_omits_details_when_redacted() -> None:
    span = _FakeSpan()

    _capture_exception(span, redacted=True)

    assert span.recorded_exceptions == []
    assert span.attributes == {
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


@pytest.mark.parametrize("redaction_enabled", [False, True])
def test_dynamic_tracer_omits_automatic_exception_details_when_redacted(
    monkeypatch: pytest.MonkeyPatch, redaction_enabled: bool
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-exception-redaction")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "_redaction_enabled", lambda: redaction_enabled)

    with pytest.raises(RuntimeError, match="secret transcript"):
        with dynamic_tracer.start_as_current_span("test-span"):
            raise RuntimeError("secret transcript")

    (span,) = exporter.get_finished_spans()
    exception_events = [event for event in span.events if event.name == "exception"]
    if redaction_enabled:
        assert exception_events == []
        assert span.status.status_code == trace.StatusCode.UNSET
        assert span.status.description is None
    else:
        assert len(exception_events) == 1
        assert exception_events[0].attributes is not None
        assert exception_events[0].attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
            "secret transcript"
        )
        assert span.status.status_code == trace.StatusCode.ERROR


@pytest.mark.parametrize("redaction_enabled", [False, True])
def test_dynamic_tracer_use_span_omits_automatic_exception_details_when_redacted(
    monkeypatch: pytest.MonkeyPatch, redaction_enabled: bool
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-use-span-exception-redaction")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "_redaction_enabled", lambda: redaction_enabled)

    span = dynamic_tracer.start_span("test-span")
    with pytest.raises(RuntimeError, match="secret transcript"):
        with dynamic_tracer.use_span(span):
            raise RuntimeError("secret transcript")
    span.end()

    (finished_span,) = exporter.get_finished_spans()
    exception_events = [event for event in finished_span.events if event.name == "exception"]
    if redaction_enabled:
        assert exception_events == []
        assert finished_span.status.status_code == trace.StatusCode.UNSET
        assert finished_span.status.description is None
    else:
        assert len(exception_events) == 1
        assert exception_events[0].attributes is not None
        assert exception_events[0].attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
            "secret transcript"
        )
        assert finished_span.status.status_code == trace.StatusCode.ERROR


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
    monkeypatch.setattr(telemetry_utils, "_redaction_enabled", lambda: redaction_enabled)

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
    monkeypatch.setattr(telemetry_utils, "_redaction_enabled", lambda: True)

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
