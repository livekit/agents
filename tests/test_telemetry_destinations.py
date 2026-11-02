"""Provider ownership and independent destination redaction using the host SDK."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.telemetry import _context, _destination, _providers, traces

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def test_setup_adopts_provider_installed_after_import_without_reordering(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"application": "kept"}), shutdown_on_exit=False)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    dynamic = _context._DynamicTracer("late-provider")
    old = dynamic._tracer_provider
    monkeypatch.setattr(_context, "tracer", dynamic)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    cloud = _providers._CloudTelemetry()
    monkeypatch.setattr(cloud, "_session", SimpleNamespace())
    monkeypatch.setattr(_providers, "OTLPSpanExporter", lambda **_: InMemorySpanExporter())
    try:
        cloud._ensure_trace_pipeline(Resource({}), "https://unused")
        assert old is not provider
        assert dynamic._tracer_provider is provider
        assert provider.resource.attributes == {"application": "kept"}
        assert provider._active_span_processor._span_processors[0] is processor
        with trace.get_tracer("standard-api", tracer_provider=provider).start_as_current_span(
            "app"
        ):
            pass
        assert [s.name for s in exporter.get_finished_spans()] == ["app"]
    finally:
        cloud.shutdown_at_exit(timeout=0.1)
        provider.shutdown()


def test_outside_job_spans_never_receive_last_job_metadata(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    processor = _providers._MetadataSpanProcessor()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    cloud = _providers._CloudTelemetry()
    cloud._session = SimpleNamespace()
    monkeypatch.setattr(cloud, "_ensure_trace_pipeline", lambda *_: None)
    monkeypatch.setattr(cloud, "_ensure_logger_provider", lambda: None)
    monkeypatch.setattr(cloud, "_ensure_meter_provider", lambda *_: None)
    monkeypatch.setattr(_providers, "_job_stamp_attributes", lambda: None)
    try:
        cloud.configure(room_id="last-room", job_id="last-job", observability_url="https://unused")
        with provider.get_tracer("worker").start_as_current_span("heartbeat"):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert "job_id" not in attrs and "room_id" not in attrs
    finally:
        cloud.shutdown_at_exit()
        provider.shutdown()


def test_destination_redaction_never_mutates_other_exporters(monkeypatch):
    monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
    monkeypatch.setattr(traces.telemetry_utils, "redaction_enabled", lambda *_: False)
    redacted = InMemorySpanExporter()
    captured = InMemorySpanExporter()
    raw = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_providers._MetadataSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(traces.RedactingSpanExporter(redacted)))
    provider.add_span_processor(
        SimpleSpanProcessor(traces.RedactingSpanExporter(captured, allow_pii=True))
    )
    provider.add_span_processor(SimpleSpanProcessor(raw))
    try:
        with provider.get_tracer("tool").start_as_current_span("lookup") as span:
            span.set_attribute("lk.pii.tool_input", "private customer text")
            span.set_attribute("gen_ai.tool.name", "lookup")
            span.add_event("gen_ai.user.message", {"content": "private customer text"})
        filtered, allowed, original = (e.get_finished_spans()[0] for e in (redacted, captured, raw))
        assert "lk.pii.tool_input" not in filtered.attributes
        assert filtered.attributes["gen_ai.tool.name"] == "lookup"
        assert allowed.attributes["lk.pii.tool_input"] == "private customer text"
        assert original.attributes["lk.pii.tool_input"] == "private customer text"
        assert allowed is not original and filtered is not original
        assert len(original.events) == 1 and len(filtered.events) == 0
    finally:
        provider.shutdown()


def test_capture_policy_is_frozen_at_span_start(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_providers._MetadataSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_providers, "_job_stamp_attributes", lambda: None)
    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    try:
        with provider.get_tracer("node").start_as_current_span("llm") as span:
            span.set_attribute("gen_ai.input.messages", "private")
        monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
        destination = InMemorySpanExporter()
        traces.RedactingSpanExporter(destination, allow_pii=True).export(
            exporter.get_finished_spans()
        )
        assert "gen_ai.input.messages" not in destination.get_finished_spans()[0].attributes
    finally:
        provider.shutdown()


def test_cloud_disabled_keeps_application_destination_active():
    application, cloud = InMemorySpanExporter(), InMemorySpanExporter()
    states = {
        "off": _destination._JobTelemetry(
            attributes={"job_id": "off"}, traces_enabled=False, logs_enabled=False
        )
    }
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(application))
    provider.add_span_processor(SimpleSpanProcessor(_destination._GatedSpanExporter(cloud, states)))
    try:
        with provider.get_tracer("tool").start_as_current_span("lookup") as span:
            span.set_attribute("job_id", "off")
        assert [span.name for span in application.get_finished_spans()] == ["lookup"]
        assert cloud.get_finished_spans() == ()
    finally:
        provider.shutdown()


def test_log_destination_redaction_uses_an_independent_sdk_record(monkeypatch):
    from opentelemetry._logs import LogRecord
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    destination = InMemoryLogRecordExporter()
    original = ReadableLogRecord(
        log_record=LogRecord(
            body="private content", attributes={"job_id": "on", "lk.pii.input": "private"}
        ),
        resource=Resource({}),
    )
    states = {
        "on": _destination._JobTelemetry(
            attributes={"job_id": "on"}, traces_enabled=False, logs_enabled=True
        )
    }
    _destination._GatedLogExporter(destination, states).export([original])
    exported = destination.get_finished_logs()[0]
    assert "lk.pii.input" not in exported.log_record.attributes
    assert original.log_record.attributes["lk.pii.input"] == "private"
    assert original.log_record.body == "private content"
    assert exported.log_record.body != "private content"


def test_nested_log_content_is_filtered_before_destination_export(monkeypatch):
    from opentelemetry._logs import LogRecord
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    exporter = InMemoryLogRecordExporter()
    original = ReadableLogRecord(
        log_record=LogRecord(
            body="session report",
            attributes={
                "configuration": {
                    "stt": {"lk.pii.keyterms": ["private customer"], "language": "en"},
                    "models": [{"lk.pii.prompt": "private prompt", "model": "small"}],
                },
            },
        ),
        resource=Resource({}),
    )
    traces.RedactingLogExporter(exporter).export([original])
    assert exporter.get_finished_logs()[0].log_record.attributes == {
        "configuration": {
            "stt": {"language": "en"},
            "models": [{"model": "small"}],
        }
    }
    assert original.log_record.attributes["configuration"]["stt"]["lk.pii.keyterms"] == [
        "private customer"
    ]


def test_explicit_noop_provider_keeps_tracing_disabled(monkeypatch):
    provider = trace.NoOpTracerProvider()
    dynamic = _context._DynamicTracer("disabled-provider")
    dynamic.set_provider(provider)
    monkeypatch.setattr(_context, "tracer", dynamic)
    cloud = _providers._CloudTelemetry()
    cloud._ensure_trace_pipeline(Resource({}), "https://unused")
    with dynamic.start_as_current_span("tool") as span:
        assert not span.is_recording()
        assert trace.get_current_span() is span
    assert cloud._owned_trace_provider is None
    assert dynamic._tracer_provider is provider


@pytest.mark.asyncio
async def test_a_stalled_exporter_bounds_the_cleanup_wait_and_reports_it():
    import asyncio
    import threading
    import time

    from opentelemetry.sdk.trace.export import SpanExportResult

    from livekit.agents.telemetry._export import BoundedSpanProcessor, diagnostics, run_bounded

    class StalledExporter(InMemorySpanExporter):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def export(self, spans):
            self.started.set()
            self.release.wait()
            return SpanExportResult.SUCCESS

    exporter = StalledExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    processor = BoundedSpanProcessor(exporter)
    provider.add_span_processor(processor)
    before = diagnostics()
    ticks = 0
    stopped = False

    async def heartbeat():
        nonlocal ticks
        while not stopped:
            ticks += 1
            await asyncio.sleep(0)

    beat = asyncio.create_task(heartbeat())
    try:
        for index in range(256):
            with provider.get_tracer("node").start_as_current_span(f"request-{index}"):
                pass
            await asyncio.sleep(0)
        assert await asyncio.to_thread(exporter.started.wait, 1)
        deadline = 0.02
        started = time.monotonic()
        await asyncio.to_thread(
            run_bounded, "flush", [("saturation-test", processor.force_flush)], deadline
        )
        assert time.monotonic() - started < deadline + 0.2
        assert diagnostics()["incomplete_cleanup"] == before["incomplete_cleanup"] + 1
        assert ticks > 1
    finally:
        stopped = True
        await beat
        exporter.release.set()
        await asyncio.to_thread(provider.shutdown)


def test_redaction_covers_links_resources_and_instrumentation_scope(monkeypatch):
    from opentelemetry.trace import Link, SpanContext

    monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
    destination, untouched = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource({"lk.pii.tenant": "private tenant", "service.name": "application"}),
        shutdown_on_exit=False,
    )
    provider.add_span_processor(SimpleSpanProcessor(traces.RedactingSpanExporter(destination)))
    provider.add_span_processor(SimpleSpanProcessor(untouched))
    try:
        tracer = provider.get_tracer(
            "application", attributes={"lk.pii.scope": "private scope", "component": "tool"}
        )
        with tracer.start_as_current_span(
            "tool",
            links=[
                Link(
                    SpanContext(trace_id=1, span_id=2, is_remote=False),
                    attributes={"lk.pii.prompt": "private prompt", "relation": "caused_by"},
                )
            ],
        ):
            pass
        filtered, original = (e.get_finished_spans()[0] for e in (destination, untouched))
        assert dict(filtered.resource.attributes) == {"service.name": "application"}
        assert dict(filtered.links[0].attributes) == {"relation": "caused_by"}
        assert dict(filtered.instrumentation_scope.attributes) == {"component": "tool"}
        assert original.resource.attributes["lk.pii.tenant"] == "private tenant"
        assert original.links[0].attributes["lk.pii.prompt"] == "private prompt"
        assert original.instrumentation_scope.attributes["lk.pii.scope"] == "private scope"
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_interleaved_jobs_keep_identity_captured_before_export():
    import asyncio

    from livekit.agents.job import _JobContextVar

    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_providers._MetadataSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    async def job(job_id):
        state = _destination._JobTelemetry(
            attributes={"job_id": job_id, "room_id": f"room-{job_id}"},
            traces_enabled=True,
            logs_enabled=True,
        )
        token = _JobContextVar.set(SimpleNamespace(_telemetry_state=state))
        try:
            with provider.get_tracer("standard-tool-api").start_as_current_span(job_id):
                await asyncio.sleep(0)
                with provider.get_tracer("standard-node-api").start_as_current_span(
                    f"node-{job_id}"
                ):
                    await asyncio.sleep(0)
        finally:
            _JobContextVar.reset(token)

    try:
        await asyncio.gather(job("one"), job("two"))
        with provider.get_tracer("worker").start_as_current_span("heartbeat"):
            pass
        spans = {span.name: span for span in exporter.get_finished_spans()}
        for job_id in ("one", "two"):
            parent, child = spans[job_id], spans[f"node-{job_id}"]
            assert child.context.trace_id == parent.context.trace_id
            assert child.parent.span_id == parent.context.span_id
            assert parent.attributes["job_id"] == child.attributes["job_id"] == job_id
            assert parent.attributes["room_id"] == child.attributes["room_id"] == f"room-{job_id}"
        assert "job_id" not in spans["heartbeat"].attributes
        assert "room_id" not in spans["heartbeat"].attributes
    finally:
        provider.shutdown()


def test_mutating_registration_metadata_cannot_reattribute_later_spans():
    from livekit.agents.job import _JobContextVar

    metadata = {"job_id": "created-job", "room_id": "created-room"}
    state = _destination._JobTelemetry(attributes=metadata, traces_enabled=True, logs_enabled=False)
    metadata.update(job_id="another-job", room_id="another-room")
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(_providers._MetadataSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    token = _JobContextVar.set(SimpleNamespace(_telemetry_state=state))
    try:
        with provider.get_tracer("tool").start_as_current_span("lookup"):
            pass
        attrs = exporter.get_finished_spans()[0].attributes
        assert attrs["job_id"] == "created-job"
        assert attrs["room_id"] == "created-room"
    finally:
        _JobContextVar.reset(token)
        provider.shutdown()


def test_broken_exporter_reports_once_without_recursive_logging(caplog):
    from livekit.agents.telemetry._export import BoundedSpanProcessor, diagnostics

    class BrokenExporter(InMemorySpanExporter):
        def export(self, spans):
            raise RuntimeError("broken SDK destination")

    before = diagnostics()
    provider = TracerProvider(shutdown_on_exit=False)
    processor = BoundedSpanProcessor(BrokenExporter())
    provider.add_span_processor(processor)
    try:
        with provider.get_tracer("tool").start_as_current_span("lookup"):
            pass
        assert processor.force_flush(1000)
        assert diagnostics()["failed"] == before["failed"] + 1
        assert not [record for record in caplog.records if "exporting" in record.getMessage()]
    finally:
        provider.shutdown()


def test_preinitialization_logs_keep_the_most_recent_bounded_window_for_replay():
    import logging

    from livekit.agents import _core
    from livekit.agents.telemetry._export import diagnostics

    before = diagnostics()
    handler = _destination._BufferingHandler()
    try:
        for index in range(256):
            handler.handle(
                logging.LogRecord("application", logging.INFO, __file__, index, "log", (), None)
            )
        assert diagnostics()["dropped"] == before["dropped"] + 256 - _core.SDK_PENDING_RECORDS
        replayed = [record.lineno for record in handler.buffer]
        assert replayed == list(range(256 - _core.SDK_PENDING_RECORDS, 256))
        assert list(handler.buffer) == []
    finally:
        handler.close()


def test_unstamped_content_cannot_be_enabled_after_span_creation(monkeypatch):
    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    raw, destination = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(raw))
    try:
        with provider.get_tracer("application-before-setup").start_as_current_span(
            "lookup"
        ) as span:
            span.set_attribute("lk.pii.input", "private customer input")
        monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
        traces.RedactingSpanExporter(destination, allow_pii=True).export(raw.get_finished_spans())
        assert "lk.pii.input" not in destination.get_finished_spans()[0].attributes
    finally:
        provider.shutdown()


def test_selected_provider_stamps_capture_for_standard_host_spans(monkeypatch):
    raw, destination = InMemorySpanExporter(), InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(raw))
    dynamic = _context._DynamicTracer("selected")
    monkeypatch.setattr(_context, "tracer", dynamic)
    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    try:
        traces.set_tracer_provider(provider)
        with provider.get_tracer("standard-host-api").start_as_current_span("node") as span:
            span.set_attribute("lk.pii.input", "private prompt")
            monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
        traces.RedactingSpanExporter(destination, allow_pii=True).export(raw.get_finished_spans())
        assert raw.get_finished_spans()[0].attributes["lk.telemetry.capture_content"] is False
        assert "lk.pii.input" not in destination.get_finished_spans()[0].attributes
    finally:
        provider.shutdown()


def test_one_log_destination_cannot_mutate_another_destinations_nested_snapshot():
    from opentelemetry._logs import LogRecord
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

    class MutatingExporter(InMemoryLogRecordExporter):
        def export(self, records):
            records[0].log_record.attributes["options"]["models"][0]["name"] = "changed"
            return super().export(records)

    original = ReadableLogRecord(
        log_record=LogRecord(
            body="report",
            attributes={
                "lk.telemetry.capture_content": True,
                "options": {"models": [{"name": "original"}]},
            },
        ),
        resource=Resource({}),
    )
    traces.RedactingLogExporter(MutatingExporter(), allow_pii=True).export([original])
    destination = InMemoryLogRecordExporter()
    traces.RedactingLogExporter(destination, allow_pii=True).export([original])
    assert original.log_record.attributes["options"]["models"][0]["name"] == "original"
    assert (
        destination.get_finished_logs()[0].log_record.attributes["options"]["models"][0]["name"]
        == "original"
    )


def test_native_metric_admission_preserves_names_and_bounds_process_cardinality():
    import subprocess
    import sys
    import textwrap

    from livekit.agents.telemetry._export import _core

    program = textwrap.dedent("""
        import importlib.util
        import sys
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import InMemoryMetricReader

        spec = importlib.util.spec_from_file_location('_core', sys.argv[1])
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        series = native._core_MetricSeries()
        reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
        histogram = provider.get_meter('application').create_histogram('gen_ai.execute_tool.duration')
        configured = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': 'lookup',
            'gen_ai.agent.name': 'back-office',
            'model_name': 'configured-model',
            'model_provider': 'configured-provider',
        }
        assert series.admit('gen_ai.execute_tool.duration', list(configured.items()))
        histogram.record(0.125, attributes=configured)
        for key in ('room_id', 'job_id', 'customer', 'exception.message'):
            assert not series.admit('gen_ai.execute_tool.duration', [(key, 'private')])
        accepted = 1
        for index in range(1100):
            attributes = {'gen_ai.tool.name': f'tool-{index}'}
            if series.admit('gen_ai.execute_tool.duration', list(attributes.items())):
                accepted += 1
                histogram.record(0.125, attributes=attributes)
        assert accepted == 1024
        # Once saturated, existing configured series still record.
        assert series.admit('gen_ai.execute_tool.duration', list(configured.items()))
        histogram.record(0.125, attributes=configured)
        data = reader.get_metrics_data()
        points = [
            point for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics for point in metric.data.data_points
        ]
        assert len(points) == 1024
        assert any(dict(point.attributes) == configured and point.count == 2 for point in points)
        provider.shutdown()
    """)
    subprocess.run([sys.executable, "-c", program, _core.__file__], check=True, timeout=10)


@pytest.mark.parametrize("conflict", ["endpoint", "meter"])
def test_configuration_conflicts_are_reported_outside_the_manager_lock(monkeypatch, conflict):
    import logging
    import warnings

    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    cloud = _providers._CloudTelemetry()
    cloud._session = SimpleNamespace()
    monkeypatch.setattr(cloud, "_ensure_trace_pipeline", lambda *_: None)
    monkeypatch.setattr(cloud, "_ensure_logger_provider", lambda: None)
    provider = MeterProvider(metric_readers=[InMemoryMetricReader()], shutdown_on_exit=False)
    if conflict == "endpoint":
        cloud._observability_url = "https://original"
        monkeypatch.setattr(cloud, "_ensure_meter_provider", lambda *_: None)
    else:
        monkeypatch.setattr(_providers.metrics_api, "get_meter_provider", lambda: provider)
    observed = []

    class WarningHandler(logging.Handler):
        def emit(self, record):
            observed.append(cloud._lock.locked())

    handler = WarningHandler()
    traces.logger.addHandler(handler)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            monkeypatch.setattr(
                warnings,
                "showwarning",
                lambda *_args, **_kwargs: observed.append(cloud._lock.locked()),
            )
            cloud.configure(
                job_id="warning-job",
                room_id="warning-room",
                observability_url="https://new",
                enable_logs=False,
            )
        assert observed and not any(observed)
    finally:
        traces.logger.removeHandler(handler)
        cloud.release("warning-job")
        cloud.shutdown_at_exit()
        provider.shutdown()


def test_early_logs_keep_capture_consent_from_emission_until_replay(monkeypatch):
    import logging

    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    from livekit.agents.job import _JobContextVar

    monkeypatch.setattr(_context, "capture_content_enabled", lambda: False)
    buffer = _destination._BufferingHandler()
    record = logging.LogRecord(
        "application", logging.INFO, __file__, 1, "private transcript", (), None
    )
    buffer.handle(record)
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider(shutdown_on_exit=False)
    provider.add_log_record_processor(_providers._MetadataLogProcessor())
    provider.add_log_record_processor(
        SimpleLogRecordProcessor(traces.RedactingLogExporter(exporter, allow_pii=True))
    )
    handler = _destination._TraceLevelLoggingHandler(logger_provider=provider)
    token = _JobContextVar.set(
        SimpleNamespace(
            _redaction_enabled=False,
            _telemetry_state=_destination._JobTelemetry(
                attributes={"job_id": "replay-job", "room_id": "replay-room"},
                traces_enabled=True,
                logs_enabled=True,
            ),
        )
    )
    try:
        monkeypatch.setattr(_context, "capture_content_enabled", lambda: True)
        buffered = list(buffer.buffer)
        assert all(item.getMessage() != "private transcript" for item in buffered)
        for retained in buffered:
            handler.handle(retained)
        exported = exporter.get_finished_logs()[0]
        assert exported.log_record.body != "private transcript"
        assert exported.log_record.attributes["lk.telemetry.capture_content"] is False
        assert "job_id" not in exported.log_record.attributes
        assert "room_id" not in exported.log_record.attributes
    finally:
        _JobContextVar.reset(token)
        buffer.close()
        provider.shutdown()


def test_sdk_failure_logs_do_not_reenter_the_framework_destination():
    import logging

    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
    from opentelemetry.sdk.trace.export import SpanExportResult

    from livekit.agents.telemetry._export import BoundedSpanProcessor, diagnostics

    class LoggingFailureExporter(InMemorySpanExporter):
        def export(self, spans):
            logging.getLogger("opentelemetry.exporter.test").warning("SDK export failed")
            return SpanExportResult.FAILURE

    cloud = InMemoryLogRecordExporter()
    log_provider = LoggerProvider(shutdown_on_exit=False)
    log_provider.add_log_record_processor(SimpleLogRecordProcessor(cloud))
    cloud_handler = _destination._TraceLevelLoggingHandler(logger_provider=log_provider)
    application = []

    class ApplicationHandler(logging.Handler):
        def emit(self, record):
            application.append(record.getMessage())

    application_handler = ApplicationHandler()
    logging.getLogger().addHandler(cloud_handler)
    logging.getLogger().addHandler(application_handler)
    provider = TracerProvider(shutdown_on_exit=False)
    processor = BoundedSpanProcessor(LoggingFailureExporter())
    provider.add_span_processor(processor)
    before = diagnostics()["failed"]
    try:
        with provider.get_tracer("tool").start_as_current_span("lookup"):
            pass
        assert processor.force_flush(1000)
        assert diagnostics()["failed"] == before + 1
        assert application == ["SDK export failed"]
        assert cloud.get_finished_logs() == ()
    finally:
        logging.getLogger().removeHandler(cloud_handler)
        logging.getLogger().removeHandler(application_handler)
        provider.shutdown()
        log_provider.shutdown()


def test_local_framework_meter_preserves_global_and_adopts_a_later_application_provider(
    monkeypatch,
):
    import io

    from opentelemetry.metrics._internal import _ProxyMeterProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, InMemoryMetricReader

    from livekit.agents.telemetry import metrics as telemetry_metrics

    original_meter = telemetry_metrics._meter_provider
    proxy = _ProxyMeterProvider()
    application_reader = InMemoryMetricReader()
    application_provider = MeterProvider(
        metric_readers=[application_reader], shutdown_on_exit=False
    )
    state = {"global": proxy}
    monkeypatch.setattr(_providers.metrics_api, "get_meter_provider", lambda: state["global"])
    cloud = _providers._CloudTelemetry()
    cloud._session = SimpleNamespace()
    monkeypatch.setattr(cloud, "_ensure_trace_pipeline", lambda *_: None)
    monkeypatch.setattr(cloud, "_ensure_logger_provider", lambda: None)
    monkeypatch.setattr(
        _providers, "OTLPMetricExporter", lambda **_: ConsoleMetricExporter(out=io.StringIO())
    )
    try:
        cloud.configure(
            job_id="local-meter-job",
            room_id="local-room",
            observability_url="https://unused",
            enable_logs=False,
        )
        local = cloud._owned_meter_provider
        assert local is not None and telemetry_metrics._meter_provider is local
        assert state["global"] is proxy
        telemetry_metrics.record_span_metric(
            "gen_ai.execute_tool.duration", 0.125, {"gen_ai.tool.name": "local-tool"}
        )
        state["global"] = application_provider
        with pytest.warns(RuntimeWarning, match="reader registered at application MeterProvider"):
            cloud.configure(
                job_id="application-meter-job",
                room_id="application-room",
                observability_url="https://unused",
                enable_logs=False,
            )
        assert state["global"] is application_provider
        assert telemetry_metrics._meter_provider is application_provider
        telemetry_metrics.record_span_metric(
            "gen_ai.execute_tool.duration", 0.125, {"gen_ai.tool.name": "application-tool"}
        )
        data = application_reader.get_metrics_data()
        points = [
            point
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
            for point in metric.data.data_points
        ]
        assert len(points) == 1 and dict(points[0].attributes) == {
            "gen_ai.tool.name": "application-tool"
        }
        cloud.release("local-meter-job")
        cloud.release("application-meter-job")
        cloud.shutdown_at_exit()
        application_provider.get_meter("application").create_counter("still.alive").add(1)
        assert application_reader.get_metrics_data() is not None
    finally:
        cloud.shutdown_at_exit()
        application_provider.shutdown()
        telemetry_metrics._adopt_meter_provider(original_meter)


def test_oversized_metadata_is_refused_before_freezing_or_traversing_values(monkeypatch):
    from livekit.agents.telemetry._export import diagnostics

    class OversizedList(list):
        def __len__(self):
            return 1_000_000

        def __iter__(self):
            raise AssertionError("oversized metadata was traversed before admission")

    cloud = _providers._CloudTelemetry()
    cloud._session = SimpleNamespace()
    monkeypatch.setattr(cloud, "_ensure_trace_pipeline", lambda *_: None)
    monkeypatch.setattr(cloud, "_ensure_logger_provider", lambda: None)
    monkeypatch.setattr(cloud, "_ensure_meter_provider", lambda *_: None)
    before = diagnostics()
    try:
        state = cloud.configure(
            job_id="oversize-job",
            room_id="oversize-room",
            observability_url="https://unused",
            enable_logs=False,
            metadata={"customer.data": OversizedList()},
        )
        assert dict(state.attributes) == {"job_id": "oversize-job", "room_id": "oversize-room"}
        assert "oversize-job" not in cloud._export_jobs
        assert diagnostics()["dropped"] == before["dropped"] + 1
    finally:
        cloud.discard("oversize-job")
        cloud.shutdown_at_exit()


@pytest.mark.asyncio
async def test_report_without_recording_options_still_emits_its_outcome(monkeypatch):
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

    from livekit.agents import _core, llm
    from livekit.agents.job import SessionReport

    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    provider = LoggerProvider(shutdown_on_exit=False)
    exporter = InMemoryLogRecordExporter()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    cloud = _providers._CloudTelemetry()
    cloud._logger_provider = provider
    monkeypatch.setattr(traces, "_cloud", cloud)
    monkeypatch.setattr(_destination, "upload_gate", traces._core.UploadGate())
    report = SessionReport(
        job_id="outcome-job",
        room_id="outcome-room",
        room="outcome-room-name",
        options=None,
        events=[],
        chat_history=llm.ChatContext.empty(),
    )
    tagger = _core.Tagger()
    tagger.success("private reason")
    try:
        await traces._upload_session_report(
            agent_name="back-office",
            observability_url="http://127.0.0.1:1",
            report=report,
            tagger=tagger,
            http_session=SimpleNamespace(),
        )
        records = exporter.get_finished_logs()
        assert len(records) == 1
        assert records[0].log_record.body == "outcome"
        assert records[0].log_record.attributes["outcome"] == {"outcome": "success"}
        assert records[0].log_record.attributes["job_id"] == "outcome-job"
    finally:
        provider.shutdown()


def test_automatic_provider_adoption_reports_legacy_pii_conflict_outside_lock(monkeypatch):
    import warnings

    provider = TracerProvider(shutdown_on_exit=False)
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(_context, "tracer", _context._DynamicTracer("automatic-adoption"))
    monkeypatch.setenv("LIVEKIT_TELEMETRY_ALLOW_PII", "0")
    monkeypatch.setattr(_providers, "OTLPSpanExporter", lambda **_: InMemorySpanExporter())
    cloud = _providers._CloudTelemetry()
    cloud._session = SimpleNamespace()
    monkeypatch.setattr(cloud, "_ensure_logger_provider", lambda: None)
    monkeypatch.setattr(cloud, "_ensure_meter_provider", lambda *_: None)
    observed = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            monkeypatch.setattr(
                warnings,
                "showwarning",
                lambda *_args, **_kwargs: observed.append(cloud._lock.locked()),
            )
            cloud.prepare(
                job_id="env-policy-job",
                room_id="env-policy-room",
                agent_name="",
                observability_url="https://unused",
            )
            cloud.configure(
                job_id="env-policy-job",
                room_id="env-policy-room",
                observability_url="https://unused",
                enable_logs=False,
            )
        assert observed == [False]
        assert _context.tracer._tracer_provider is provider
    finally:
        cloud.release("env-policy-job")
        cloud.discard("env-policy-job")
        cloud.shutdown_at_exit()
        provider.shutdown()


def test_an_unsuccessful_sdk_flush_is_observed_without_failing_the_session():
    from livekit.agents.telemetry._export import diagnostics, run_bounded

    before = diagnostics()["failed"]
    run_bounded("flush", [("unsuccessful", lambda: False)], 1)
    assert diagnostics()["failed"] == before + 1
