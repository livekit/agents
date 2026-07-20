from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry import trace

from livekit.agents.telemetry import trace_types, utils as telemetry_utils
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


def test_record_exception_uses_job_enable_redaction(monkeypatch: pytest.MonkeyPatch) -> None:
    span = _FakeSpan()

    def get_job_context(*, required: bool = True) -> SimpleNamespace:
        return SimpleNamespace(job=SimpleNamespace(enable_redaction=True))

    monkeypatch.setattr("livekit.agents.job.get_job_context", get_job_context)
    _capture_exception(span, redacted=NOT_GIVEN)

    assert span.recorded_exceptions == []
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
        telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    )
    assert trace_types.ATTR_EXCEPTION_TRACE not in span.attributes
