"""Regression tests for the unresolved review findings on the report-only pipeline.

A ``record=False`` job still uploads its session report (evaluations, tags,
outcome) through the log pipeline. These tests pin the isolation that upload
needs:

* session-report records are recognised by an internal origin marker, not by a
  public logger name, so an ordinary Python logger called ``evaluations`` stays
  off for a job whose ``logs`` option is off;
* a job that resolved metrics off records nothing through the process-wide
  meter another job installed, while the recording job still does;
* a project that mandates redaction does not export the free-form parts of the
  evaluation, tag, and outcome payloads (judge reasoning, tag metadata, outcome
  reason); the verdict, tag name, and outcome still upload.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from livekit.agents.evals import EvaluationResult, JudgmentResult
from livekit.agents.job import JobContext, _JobContextVar
from livekit.agents.metrics import LLMMetrics
from livekit.agents.observability import Tagger
from livekit.agents.telemetry import otel_metrics, pii, traces
from livekit.agents.telemetry.traces import (
    _GatedLogExporter,
    _JobTelemetry,
    _setup_cloud_tracer,
    _upload_session_report,
)
from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED
from livekit.agents.voice.agent_session import _RECORDING_ALL_OFF

from .test_recording import _stub_cloud_tracer_deps

pytestmark = pytest.mark.unit

# The marker is implementation-internal; written out here so the test pins the
# contract rather than importing the constant under test.
_ORIGIN_MARKER = "lk.internal.session_report"
_SECRET = "the caller's card number is 4111 1111 1111 1111"


def _job_ctx(
    job_id: str,
    *,
    state: _JobTelemetry | None = None,
    redaction: bool = False,
) -> JobContext:
    ctx = object.__new__(JobContext)
    ctx._info = SimpleNamespace(
        url="https://test.livekit.cloud",
        fake_job=False,
        job=SimpleNamespace(
            id=job_id,
            agent_name="test-agent",
            enable_recording=False,
            enable_redaction=redaction,
            room=SimpleNamespace(sid="room-1", name="room-1"),
            attributes={},
        ),
    )
    ctx._recording_initialized = False
    ctx._redaction_enabled = redaction
    ctx._telemetry_state = state
    ctx._tagger = Tagger()
    return ctx


def _log_provider(*processors: Any) -> LoggerProvider:
    provider = LoggerProvider(resource=Resource.create({}))
    for processor in processors:
        provider.add_log_record_processor(processor)
    return provider


def _llm_metrics() -> LLMMetrics:
    return LLMMetrics(
        label="test.LLM",
        request_id="req-1",
        timestamp=0.0,
        duration=1.0,
        ttft=0.1,
        cancelled=False,
        completion_tokens=10,
        prompt_tokens=100,
        prompt_cached_tokens=20,
        total_tokens=110,
        tokens_per_second=10.0,
    )


def test_python_log_named_evaluations_does_not_upload_for_a_logs_off_job() -> None:
    """The gate must not treat a public logger name as proof of session-report origin."""
    exporter = InMemoryLogRecordExporter()
    export_jobs = {
        "job-off": _JobTelemetry(attributes={}, traces_enabled=False, logs_enabled=False)
    }
    provider = _log_provider(SimpleLogRecordProcessor(_GatedLogExporter(exporter, export_jobs)))

    provider.get_logger("evaluations").emit(
        body="ordinary diagnostic", attributes={"job_id": "job-off"}
    )

    assert not exporter.get_finished_logs()


def test_origin_marked_session_report_record_uploads_and_drops_the_marker() -> None:
    """Only records _upload_session_report emitted pass; the marker never ships."""
    exporter = InMemoryLogRecordExporter()
    export_jobs = {
        "job-off": _JobTelemetry(attributes={}, traces_enabled=False, logs_enabled=False)
    }
    provider = _log_provider(SimpleLogRecordProcessor(_GatedLogExporter(exporter, export_jobs)))

    provider.get_logger("evaluations").emit(
        body="evaluation",
        attributes={"job_id": "job-off", _ORIGIN_MARKER: True},
    )

    records = exporter.get_finished_logs()
    assert [r.log_record.body for r in records] == ["evaluation"]
    assert _ORIGIN_MARKER not in (records[0].log_record.attributes or {})


def test_all_off_job_does_not_record_through_a_previous_jobs_meter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused worker keeps the meter, but a job that resolved metrics off stays off."""
    with _stub_cloud_tracer_deps():
        state_recording = _setup_cloud_tracer(
            room_id="room-1",
            job_id="job-recording",
            observability_url="https://test.livekit.cloud",
            enable_traces=True,
            enable_logs=True,
            enable_metrics=True,
        )
        state_all_off = _setup_cloud_tracer(
            room_id="room-2",
            job_id="job-all-off",
            observability_url="https://test.livekit.cloud",
            enable_traces=False,
            enable_logs=False,
            enable_session_report=True,
            enable_metrics=False,
        )

        usage_counter = MagicMock()
        monkeypatch.setattr(otel_metrics, "_llm_input_tokens", usage_counter)

        token = _JobContextVar.set(_job_ctx("job-all-off", state=state_all_off))
        try:
            otel_metrics.collect_usage(_llm_metrics())
        finally:
            _JobContextVar.reset(token)
        usage_counter.add.assert_not_called()

        token = _JobContextVar.set(_job_ctx("job-recording", state=state_recording))
        try:
            otel_metrics.collect_usage(_llm_metrics())
        finally:
            _JobContextVar.reset(token)
        usage_counter.add.assert_called()


def _report() -> SimpleNamespace:
    return SimpleNamespace(
        room_id="room-1",
        job_id="job-eval",
        room="room-1",
        options=SimpleNamespace(recording_options=dict(_RECORDING_ALL_OFF)),
        started_at=1000.0,
        timestamp=1010.0,
        sdk_version="1.0.0",
        model_usage=[],
        chat_history=SimpleNamespace(items=[]),
        audio_recording_path=None,
        audio_recording_started_at=None,
    )


def _sensitive_tagger() -> Tagger:
    tagger = Tagger()
    tagger._evaluation(
        EvaluationResult(
            judgments={"judge": JudgmentResult(verdict="pass", reasoning=_SECRET)},
        )
    )
    tagger.add("appointment:booked", metadata={"slot": _SECRET})
    tagger.success(reason=_SECRET)
    return tagger


def _contains_secret(value: Any) -> bool:
    if isinstance(value, str):
        return _SECRET in value
    if isinstance(value, Mapping):
        return any(_contains_secret(k) or _contains_secret(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret(item) for item in value)
    return False


async def _upload_with_redaction(*, redaction: bool) -> list[Any]:
    exporter = InMemoryLogRecordExporter()
    provider = _log_provider(
        pii._PIIFilteringLogProcessor(),
        SimpleLogRecordProcessor(exporter),
    )

    previous_cloud = traces._cloud
    traces._cloud = traces._CloudTelemetry()
    traces._cloud._logger_provider = provider
    traces._upload_gate.reset()
    token = _JobContextVar.set(_job_ctx("job-eval", redaction=redaction))
    try:
        await _upload_session_report(
            agent_name="test-agent",
            observability_url="https://test.livekit.cloud",
            report=_report(),
            tagger=_sensitive_tagger(),
            http_session=MagicMock(),
            # _on_session_end passes the resolved flag this way
            metadata={ATTRIBUTE_REDACTION_ENABLED: True} if redaction else None,
        )
    finally:
        _JobContextVar.reset(token)
        traces._cloud = previous_cloud

    return exporter.get_finished_logs()


async def test_redacted_session_report_env_withholds_evaluation_and_tag_content() -> None:
    records = await _upload_with_redaction(redaction=True)

    assert [r.log_record.body for r in records] == ["evaluation", "tag", "outcome"]
    leaked = [r.log_record.body for r in records if _contains_secret(r.log_record.attributes)]
    assert leaked == [], f"PII reached the exporter in records: {leaked!r}"

    by_body = {r.log_record.body: r.log_record.attributes or {} for r in records}
    # the non-content fields still describe the result
    assert by_body["evaluation"]["evaluation"]["verdict"] == "pass"
    assert "reasoning" not in by_body["evaluation"]["evaluation"]
    assert "instructions" not in by_body["evaluation"]["evaluation"]
    assert by_body["tag"]["tag"]["name"] == "appointment:booked"
    assert "metadata" not in by_body["tag"]["tag"]
    assert by_body["outcome"]["outcome"]["outcome"] == "success"
    assert "reason" not in by_body["outcome"]["outcome"]


async def test_unredacted_session_report_still_carries_evaluation_and_tag_content() -> None:
    records = await _upload_with_redaction(redaction=False)

    assert [r.log_record.body for r in records] == ["evaluation", "tag", "outcome"]
    assert any(_contains_secret(r.log_record.attributes) for r in records), (
        "redaction-only withholding must not hide content from a project that allows it"
    )
