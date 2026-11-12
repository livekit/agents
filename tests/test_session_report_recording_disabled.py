"""A session started with recording fully disabled (``record=False``) must still
export its evaluations, tag metadata, and outcome to LiveKit Cloud.

``JobContext._on_session_end`` has an explicit ``has_evals`` branch for exactly this
case, but ``init_recording`` used to return before any log pipeline existed for
all-off options: the records were emitted into a no-op provider in a fresh process,
or dropped by the log gate in a reused one. This test drives the real
``JobContext.init_recording`` / ``_on_session_end`` pair with a capturing log
exporter.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.metrics._internal import _ProxyMeterProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor

from livekit.agents.evals import EvaluationResult, JudgmentResult
from livekit.agents.job import JobContext, _JobContextVar
from livekit.agents.observability import Tagger
from livekit.agents.telemetry import traces
from livekit.agents.voice.agent_session import _RECORDING_ALL_OFF

pytestmark = pytest.mark.unit

_TRACES_MOD = "livekit.agents.telemetry.traces"


def _stub_access_token(mock_access_token: MagicMock) -> None:
    token = MagicMock()
    token.with_observability_grants.return_value = token
    token.with_ttl.return_value = token
    token.to_jwt.return_value = "test-jwt"
    mock_access_token.return_value = token


@contextlib.contextmanager
def _capturing_telemetry() -> Iterator[InMemoryLogRecordExporter]:
    """Fresh cloud telemetry with a synchronous, in-memory log exporter.

    Mirrors ``_stub_cloud_tracer_deps`` in ``tests/test_recording.py``: no network,
    no process-global OTel providers, no atexit handlers; log records export as
    soon as they are emitted, keeping the assertion deterministic.
    """
    exporter = InMemoryLogRecordExporter()
    logger_state: dict[str, Any] = {"provider": MagicMock()}  # not an SDK LoggerProvider
    meter_state: dict[str, Any] = {"provider": _ProxyMeterProvider()}

    previous_cloud = traces._cloud
    previous_tracer_provider = traces.tracer._tracer_provider
    traces._cloud = traces._CloudTelemetry()
    try:
        with (
            patch(f"{_TRACES_MOD}.api.AccessToken") as mock_access_token,
            patch(f"{_TRACES_MOD}.OTLPSpanExporter"),
            patch(f"{_TRACES_MOD}.BatchSpanProcessor"),
            patch(f"{_TRACES_MOD}.OTLPLogExporter", return_value=exporter),
            patch(f"{_TRACES_MOD}.BatchLogRecordProcessor", SimpleLogRecordProcessor),
            patch(f"{_TRACES_MOD}.OTLPMetricExporter"),
            patch(f"{_TRACES_MOD}.PeriodicExportingMetricReader"),
            patch(f"{_TRACES_MOD}.SdkMeterProvider"),
            patch(
                f"{_TRACES_MOD}.get_logger_provider",
                side_effect=lambda: logger_state["provider"],
            ),
            patch(
                f"{_TRACES_MOD}.set_logger_provider",
                side_effect=lambda p: logger_state.__setitem__("provider", p),
            ),
            patch(
                f"{_TRACES_MOD}.metrics_api.get_meter_provider",
                side_effect=lambda: meter_state["provider"],
            ),
            patch(
                f"{_TRACES_MOD}.metrics_api.set_meter_provider",
                side_effect=lambda p: meter_state.__setitem__("provider", p),
            ),
            patch(f"{_TRACES_MOD}.atexit"),
        ):
            _stub_access_token(mock_access_token)
            traces._upload_gate.reset()
            yield exporter
    finally:
        if (handler := traces._cloud.log_handler) is not None:
            logging.getLogger().removeHandler(handler)
        traces._cloud = previous_cloud
        traces.tracer.set_provider(previous_tracer_provider)


def _make_job_ctx() -> JobContext:
    ctx = object.__new__(JobContext)
    ctx._info = SimpleNamespace(
        url="https://test.livekit.cloud",
        fake_job=False,
        job=SimpleNamespace(
            id="job-eval",
            agent_name="test-agent",
            enable_recording=False,
            enable_redaction=False,
            room=SimpleNamespace(sid="room-eval", name="room-eval"),
            attributes={},
        ),
    )
    ctx._early_log_handler = None
    ctx._recording_initialized = False
    ctx._redaction_enabled = False
    ctx._telemetry_state = None
    ctx._tagger = Tagger()
    ctx.simulation_context = lambda: None  # type: ignore[method-assign]
    return ctx


def _session_report() -> SimpleNamespace:
    return SimpleNamespace(
        room_id="room-eval",
        job_id="job-eval",
        room="room-eval",
        options=SimpleNamespace(recording_options=dict(_RECORDING_ALL_OFF)),
        started_at=1000.0,
        timestamp=1010.0,
        sdk_version="1.0.0",
        model_usage=[],
        chat_history=SimpleNamespace(items=[]),
        audio_recording_path=None,
        audio_recording_started_at=None,
    )


def test_disabled_recording_still_exports_evaluations_tags_and_outcome() -> None:
    with _capturing_telemetry() as exporter:
        ctx = _make_job_ctx()
        ctx.tagger.success(reason="booking completed")
        ctx.tagger.add("appointment:booked", metadata={"slot": "9am"})
        ctx.tagger._evaluation(
            EvaluationResult(
                judgments={
                    "task_completion": JudgmentResult(verdict="pass", reasoning="done"),
                }
            )
        )
        ctx._primary_agent_session = SimpleNamespace(history=[], _recorder_io=None)
        report = _session_report()
        ctx.make_session_report = lambda session: report  # type: ignore[method-assign]

        console = MagicMock()
        console.enabled = False

        ctx._prepare_telemetry()
        ctx.init_recording(dict(_RECORDING_ALL_OFF))

        token = _JobContextVar.set(ctx)
        try:
            with (
                patch("livekit.agents.cli.AgentsConsole.get_instance", return_value=console),
                patch("livekit.agents.telemetry.otel_metrics.flush_turn_metrics"),
                patch("livekit.agents.job.http_context") as mock_http_context,
            ):
                mock_http_context.http_session.return_value = MagicMock()
                asyncio.run(ctx._on_session_end())
        finally:
            _JobContextVar.reset(token)

        # record=False still means Python log export is off, not even through the
        # pipeline created for the session report records.
        logging.getLogger("test.session-report").warning("ordinary python log")

    finished = exporter.get_finished_logs()
    evaluations = [
        r
        for r in finished
        if r.instrumentation_scope is not None and r.instrumentation_scope.name == "evaluations"
    ]
    bodies = sorted(str(r.log_record.body) for r in evaluations)
    assert bodies == ["evaluation", "outcome", "tag"], (
        "evaluations, tag metadata, and the outcome must be exported for a "
        f"record=False session; exported records: {[str(r.log_record.body) for r in finished]!r}"
    )
    assert all(str(r.log_record.body) != "ordinary python log" for r in finished), (
        "record=False must not start exporting ordinary Python logs"
    )
