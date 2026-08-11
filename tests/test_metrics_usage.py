"""Unit tests for LLM usage aggregation, incl. prompt-cache creation tokens."""

from __future__ import annotations

import pytest

from livekit.agents.metrics import LLMMetrics, LLMModelUsage, ModelUsageCollector
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
