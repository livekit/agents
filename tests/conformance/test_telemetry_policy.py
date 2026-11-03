"""J-5: metrics carry no per-job labels. Judge explanations follow GenAI message capture in the
core (`lk-agents-inference` evals tests)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from livekit.agents import job, metrics
from livekit.agents.telemetry import metrics as otel_metrics

pytestmark = pytest.mark.unit


@pytest.mark.no_concurrent
def test_metric_series_do_not_grow_with_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    meter = provider.get_meter("metric-policy-test")
    monkeypatch.setattr(otel_metrics, "_meter", meter)
    attributes: dict[str, str] = {}
    context = SimpleNamespace(_telemetry_state=SimpleNamespace(attributes=attributes))
    monkeypatch.setattr(job, "get_job_context", lambda **_: context)
    otel_metrics._histogram.cache_clear()
    otel_metrics._counter.cache_clear()
    count = 64
    try:
        for index in range(count):
            attributes.update(room_id=f"RM_{index}", job_id=f"AJ_{index}", customer=f"user-{index}")
            otel_metrics.collect_usage(
                metrics.STTMetrics(
                    label="test",
                    request_id=str(index),
                    timestamp=0.0,
                    duration=0.25,
                    audio_duration=0.25,
                    streamed=True,
                    acquire_time=0.125,
                    connection_reused=bool(index % 2),
                    metadata=metrics.Metadata(model_name="nova-3", model_provider="deepgram"),
                )
            )
            otel_metrics.record_span_metric(
                "gen_ai.execute_tool.duration",
                0.125,
                {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup"},
            )
        data = reader.get_metrics_data()
        assert data is not None
        exported = {
            metric.name: [dict(point.attributes or {}) for point in metric.data.data_points]
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        model = {"model_provider": "deepgram", "model_name": "nova-3"}
        assert exported == {
            "lk.agents.usage.stt_audio_duration": [model],
            "lk.agents.connection.acquire_time": [
                {**model, "connection_reused": "false"},
                {**model, "connection_reused": "true"},
            ],
            "gen_ai.execute_tool.duration": [
                {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "lookup"}
            ],
        }
    finally:
        otel_metrics._histogram.cache_clear()
        otel_metrics._counter.cache_clear()
        provider.shutdown()
