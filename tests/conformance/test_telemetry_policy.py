"""J-5: content is captured only after explicit consent."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    HistogramDataPoint,
    InMemoryMetricReader,
    NumberDataPoint,
)

from livekit.agents import job, llm, metrics
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
                from unittest.mock import AsyncMock
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import SimpleSpanProcessor
                from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
                from livekit.agents import evals
                from livekit.agents.telemetry import trace_types

                exporter = InMemorySpanExporter()
                provider = TracerProvider(shutdown_on_exit=False)
                provider.add_span_processor(SimpleSpanProcessor(exporter))
                evals.tracer = provider.get_tracer("capture-policy-test")
                evals._judge_with_llm = AsyncMock(
                    return_value=evals.JudgmentResult(verdict="pass", reasoning="private content")
                )
                asyncio.run(evals._evaluate_with_llm(None, "prompt", name="test"))
                attrs = exporter.get_finished_spans()[0].attributes
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
def test_metric_series_are_bounded_across_jobs_and_arbitrary_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    meter = provider.get_meter("metric-policy-test")
    monkeypatch.setattr(otel_metrics, "_meter", meter)
    monkeypatch.setattr(
        otel_metrics,
        "_event_loop_blocked",
        meter.create_histogram("lk.agents.event_loop.blocked_duration"),
    )
    attributes: dict[str, str] = {}
    context = SimpleNamespace(_telemetry_state=SimpleNamespace(attributes=attributes))
    monkeypatch.setattr(job, "get_job_context", lambda **_: context)
    otel_metrics._histogram.cache_clear()
    otel_metrics._counter.cache_clear()
    count = 64
    try:
        for index in range(count):
            attributes.update(room_id=f"RM_{index}", job_id=f"AJ_{index}", customer=f"user-{index}")
            metadata = metrics.Metadata(
                model_name=f"model-{index}", model_provider=f"provider-{index}"
            )
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
                    metadata=metadata,
                )
            )
            history = llm.ChatContext.empty()
            history.add_message(role="assistant", content="private content")
            history.messages()[0].metrics["llm_node_ttft"] = 0.125
            history.messages()[0].metrics["llm_metadata"] = {
                "model_name": f"model-{index}",
                "model_provider": f"provider-{index}",
            }
            otel_metrics.flush_turn_metrics(history)
            otel_metrics.record_span_metric(
                "gen_ai.execute_tool.duration",
                0.125,
                {"gen_ai.tool.name": f"tool-{index}", "exception.message": f"error-{index}"},
            )
            otel_metrics.record_event_loop_blocked(
                0.125, severity="error" if index % 2 else "warning"
            )
        otel_metrics.record_event_loop_blocked(0.125, severity="private exception text")
        data = reader.get_metrics_data()
        assert data is not None
        exported = {
            metric.name: metric.data.data_points
            for resource in data.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        expected_labels = {
            "lk.agents.usage.stt_audio_duration": [{}],
            "lk.agents.connection.acquire_time": [
                {"connection_reused": "false"},
                {"connection_reused": "true"},
            ],
            "lk.agents.turn.llm_ttft": [{}],
            "gen_ai.execute_tool.duration": [{}],
            "lk.agents.event_loop.blocked_duration": [
                {"severity": "warning"},
                {"severity": "error"},
                {},
            ],
        }
        assert exported.keys() == expected_labels.keys()
        for name, labels in expected_labels.items():
            points = exported[name]
            assert [dict(point.attributes or {}) for point in points] == labels
            total = 0.0
            if name.endswith("audio_duration"):
                for point in points:
                    assert isinstance(point, NumberDataPoint)
                    total += point.value
                assert total == count * 0.25
            else:
                for point in points:
                    assert isinstance(point, HistogramDataPoint)
                    total += point.count
                expected_count = count + (name == "lk.agents.event_loop.blocked_duration")
                assert total == expected_count
    finally:
        otel_metrics._histogram.cache_clear()
        otel_metrics._counter.cache_clear()
        provider.shutdown()
