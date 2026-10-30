"""Queued telemetry owns primitive snapshots, including opted-in early logs."""

from __future__ import annotations

import logging

import pytest
from opentelemetry._logs import LogRecord
from opentelemetry.sdk._logs import LoggerProvider, ReadableLogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from livekit.agents import _core
from livekit.agents.telemetry import _context, _destination, _export, _redaction

pytestmark = pytest.mark.unit


def test_queued_log_body_cannot_grow_or_change_through_the_callers_dictionary() -> None:
    exporter = InMemoryLogRecordExporter()
    processor = _export.BoundedLogRecordProcessor(exporter)
    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(processor)
    values = ["captured"]
    body = {"values": values}
    try:
        provider.get_logger("snapshot").emit(
            body=body,
            attributes={"lk.telemetry.capture_content": True, "lk.telemetry.safe_body": True},
        )
        values.append("changed after admission" * 10000)
        body["new"] = "changed"
        assert processor.force_flush()
        (record,) = exporter.get_finished_logs()
        assert record.log_record.body == {"values": ["captured"]}
        assert record.log_record.attributes["lk.telemetry.safe_body"] is True
    finally:
        provider.shutdown()


def test_early_log_arguments_are_captured_before_application_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
    values = ["captured"]
    record = logging.LogRecord("snapshot", logging.INFO, __file__, 1, "values: %s", (values,), None)
    record.__dict__["nested"] = {"values": values}
    handler = _destination._BufferingHandler()
    try:
        handler.emit(record)
        values.append("changed after admission")
        (captured,) = list(handler.buffer)
        assert captured.getMessage() == "values: ['captured']"
        assert captured.__dict__["nested"] == {"values": ["captured"]}
    finally:
        handler.close()


def test_arbitrary_precision_integer_cannot_bypass_record_byte_admission() -> None:
    assert _core._core_value_bytes(1 << (_core.SDK_RECORD_BYTES * 8)) > _core.SDK_RECORD_BYTES


def test_destination_refuses_oversized_event_sets_instead_of_silently_truncating() -> None:
    inner = InMemorySpanExporter()
    exporter = _redaction.RedactingSpanExporter(inner)
    span = ReadableSpan(
        "many-events",
        resource=Resource({}),
        status=Status(StatusCode.UNSET),
        events=[
            Event("work", {"sequence": index}) for index in range(_core.TELEMETRY_MAX_EVENTS + 1)
        ],
    )
    before = _export.diagnostics()["dropped"]
    try:
        assert exporter.export([span]) is SpanExportResult.FAILURE
        assert inner.get_finished_spans() == ()
        assert _export.diagnostics()["dropped"] == before + 1
    finally:
        exporter.shutdown()


def test_destination_redacts_forged_safe_bodies_and_native_report_content_independently() -> None:
    attrs = {"lk.telemetry.capture_content": True, "lk.telemetry.safe_body": True}
    report_attrs = {
        **attrs,
        "chat.item": {"content": "chat secret"},
        "session.options": {"prompt": "options secret"},
        "session.tags": ["tag secret"],
        "tag": {"name": "private", "metadata": {"notes": "metadata secret"}},
        "evaluation": {
            "verdict": "pass",
            "reasoning": "reason secret",
            "instructions": "prompt secret",
        },
        "outcome": {"outcome": "pass", "reason": "outcome secret"},
        "operation": {"reason": "retry", "instructions": "continue"},
    }
    records = [
        ReadableLogRecord(
            log_record=LogRecord(body="body secret", attributes=attrs), resource=Resource({})
        ),
        ReadableLogRecord(
            log_record=LogRecord(body="evaluation", attributes=report_attrs), resource=Resource({})
        ),
    ]
    denied, allowed = InMemoryLogRecordExporter(), InMemoryLogRecordExporter()
    private = _redaction.RedactingLogExporter(denied)
    consenting = _redaction.RedactingLogExporter(allowed, allow_pii=True)
    try:
        assert private.export(records) is LogRecordExportResult.SUCCESS
        assert consenting.export(records) is LogRecordExportResult.SUCCESS
        actual = [
            (record.log_record.body, dict(record.log_record.attributes or {}))
            for record in denied.get_finished_logs()
        ]
        assert actual == [
            ("log body redacted", {"lk.telemetry.capture_content": True}),
            (
                "evaluation",
                {
                    "lk.telemetry.capture_content": True,
                    "evaluation": {"verdict": "pass"},
                    "outcome": {"outcome": "pass"},
                    "operation": {"reason": "retry", "instructions": "continue"},
                },
            ),
        ]
        assert allowed.get_finished_logs()[0].log_record.body == "body secret"
        assert (
            allowed.get_finished_logs()[1].log_record.attributes["evaluation"]["reasoning"]
            == "reason secret"
        )
        assert records[1].log_record.attributes["chat.item"] == {"content": "chat secret"}
    finally:
        private.shutdown()
        consenting.shutdown()
