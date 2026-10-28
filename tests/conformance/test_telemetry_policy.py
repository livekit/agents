"""J-5: content is captured only after explicit consent; metrics carry no per-job labels."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from livekit.agents import job, metrics
from livekit.agents.telemetry import metrics as otel_metrics

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("setting", "enabled"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        ("garbage", False),
        ("true", True),
        ("  TrUe  ", True),
        ("1", True),
    ],
)
def test_content_capture_requires_explicit_opt_in(setting: str | None, enabled: bool) -> None:
    env = os.environ.copy()
    variable = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
    env.pop(variable, None)
    if setting is not None:
        env[variable] = setting
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import asyncio
                import json
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
                from livekit.agents import evals
                from livekit.agents.llm import ChatContext
                from livekit.agents.telemetry import trace_types

                class Private(evals.Judge):
                    def __init__(self):
                        super().__init__(name="private")

                    async def evaluate(self, **kwargs):
                        return evals.JudgmentResult(verdict="pass", reasoning="private content")

                exporter = InMemorySpanExporter()
                provider = TracerProvider(shutdown_on_exit=False)
                provider.add_span_processor(SimpleSpanProcessor(exporter))
                evals.tracer = provider.get_tracer("capture-policy-test")
                asyncio.run(evals._evaluate([Private()], ChatContext.empty(), None, None))
                (result,) = exporter.get_finished_spans()[0].events
                attrs = result.attributes
                assert attrs[trace_types.ATTR_GEN_AI_EVALUATION_SCORE_LABEL] == "pass"
                print(json.dumps(trace_types.ATTR_GEN_AI_EVALUATION_EXPLANATION in attrs))
                provider.shutdown()
            """),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    assert result.stdout.strip() == str(enabled).lower()


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
