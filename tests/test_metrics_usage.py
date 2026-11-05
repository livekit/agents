"""Unit tests for model usage aggregation."""

from __future__ import annotations

import pytest

from livekit.agents.metrics import (
    LLMMetrics,
    LLMModelUsage,
    ModelUsageCollector,
    RealtimeModelMetrics,
    STTMetrics,
    STTModelUsage,
)
from livekit.agents.metrics.base import Metadata

pytestmark = pytest.mark.unit


def _llm_metrics(**overrides: object) -> LLMMetrics:
    base: dict[str, object] = {
        "label": "test.LLM",
        "request_id": "req-1",
        "timestamp": 0.0,
        "duration": 1.0,
        "ttft": 0.1,
        "cancelled": False,
        "completion_tokens": 10,
        "prompt_tokens": 100,
        "prompt_cached_tokens": 20,
        "total_tokens": 110,
        "tokens_per_second": 10.0,
        "metadata": Metadata(model_provider="anthropic", model_name="claude-sonnet-4"),
    }
    base.update(overrides)
    return LLMMetrics(**base)


def test_llm_metrics_defaults_cache_creation_to_zero() -> None:
    m = _llm_metrics()
    assert m.cache_creation_tokens == 0


def test_llm_metrics_carries_cache_creation_tokens() -> None:
    m = _llm_metrics(cache_creation_tokens=42)
    assert m.cache_creation_tokens == 42


def test_collector_aggregates_cache_creation_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(cache_creation_tokens=42))
    collector.collect(_llm_metrics(cache_creation_tokens=8))

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.input_cache_creation_tokens == 50


def test_llm_metrics_defaults_reasoning_to_zero() -> None:
    m = _llm_metrics()
    assert m.reasoning_tokens == 0


def test_llm_metrics_carries_reasoning_tokens() -> None:
    m = _llm_metrics(reasoning_tokens=64)
    assert m.reasoning_tokens == 64


def test_collector_aggregates_reasoning_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(_llm_metrics(completion_tokens=100, reasoning_tokens=64))
    collector.collect(_llm_metrics(completion_tokens=50, reasoning_tokens=8))

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.output_reasoning_tokens == 72
    # reasoning is a subset of the output tokens, never added on top of them
    assert llm_usage.output_tokens == 150


def test_collector_aggregates_realtime_reasoning_tokens() -> None:
    collector = ModelUsageCollector()
    collector.collect(
        RealtimeModelMetrics(
            request_id="req-1",
            timestamp=0.0,
            input_tokens=500,
            output_tokens=300,
            reasoning_tokens=40,
            total_tokens=800,
            input_token_details=RealtimeModelMetrics.InputTokenDetails(),
            output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                text_tokens=200, audio_tokens=100
            ),
            metadata=Metadata(model_provider="openai", model_name="gpt-4o-realtime"),
        )
    )

    usage = collector.flatten()
    assert len(usage) == 1
    llm_usage = usage[0]
    assert isinstance(llm_usage, LLMModelUsage)
    assert llm_usage.output_tokens == 300
    assert llm_usage.output_text_tokens == 200
    assert llm_usage.output_audio_tokens == 100
    assert llm_usage.output_reasoning_tokens == 40


def _stt_metrics(**overrides: object) -> STTMetrics:
    base: dict[str, object] = {
        "label": "test.STT",
        "request_id": "req-1",
        "timestamp": 0.0,
        "duration": 0.0,
        "audio_duration": 1.0,
        "streamed": True,
        "metadata": Metadata(model_provider="mistralai", model_name="voxtral-mini"),
    }
    base.update(overrides)
    return STTMetrics(**base)


def test_stt_metrics_defaults_total_tokens_to_input_plus_output() -> None:
    metrics = _stt_metrics(input_tokens=80, output_tokens=20)
    assert metrics.total_tokens == 100


def test_stt_metrics_keeps_an_explicit_total() -> None:
    metrics = _stt_metrics(input_tokens=80, output_tokens=20, total_tokens=99)
    assert metrics.total_tokens == 99


def test_collector_aggregates_streaming_stt_token_usage() -> None:
    collector = ModelUsageCollector()
    collector.collect(_stt_metrics(input_tokens=80, input_audio_tokens=60, output_tokens=20))
    collector.collect(_stt_metrics(input_tokens=40, input_audio_tokens=30, output_tokens=10))

    usage = collector.flatten()
    assert len(usage) == 1
    stt_usage = usage[0]
    assert isinstance(stt_usage, STTModelUsage)
    assert stt_usage.input_tokens == 120
    assert stt_usage.input_audio_tokens == 90
    assert stt_usage.output_tokens == 30
    assert stt_usage.audio_duration == 2.0
