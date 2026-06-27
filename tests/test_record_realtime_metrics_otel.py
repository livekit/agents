from __future__ import annotations

from unittest.mock import MagicMock

from livekit.agents.metrics.base import Metadata, RealtimeModelMetrics
from livekit.agents.telemetry import trace_types
from livekit.agents.telemetry.utils import record_realtime_metrics


def test_record_realtime_metrics_sets_langfuse_otel_usage_keys() -> None:
    span = MagicMock()
    span.is_recording.return_value = True

    ev = RealtimeModelMetrics(
        request_id="req-1",
        timestamp=0.0,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(
            text_tokens=120,
            audio_tokens=5,
            cached_tokens=32,
            cached_tokens_details=RealtimeModelMetrics.CachedTokenDetails(
                text_tokens=30,
                audio_tokens=2,
            ),
        ),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(
            text_tokens=40,
            audio_tokens=7,
        ),
        metadata=Metadata(model_name="gpt-realtime", model_provider="openai"),
    )

    record_realtime_metrics(span, ev)

    span.set_attributes.assert_called_once()
    attrs = span.set_attributes.call_args[0][0]

    assert attrs[trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 120
    assert attrs[trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 40
    assert attrs[trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 30
    assert attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS] == 5
    assert attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS] == 2
    assert attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS] == 7


def test_record_realtime_metrics_omits_zero_breakdown() -> None:
    span = MagicMock()
    span.is_recording.return_value = True

    ev = RealtimeModelMetrics(
        request_id="req-2",
        timestamp=0.0,
        input_token_details=RealtimeModelMetrics.InputTokenDetails(
            text_tokens=10,
            audio_tokens=0,
            cached_tokens_details=None,
        ),
        output_token_details=RealtimeModelMetrics.OutputTokenDetails(
            text_tokens=3,
            audio_tokens=0,
        ),
    )

    record_realtime_metrics(span, ev)

    attrs = span.set_attributes.call_args[0][0]
    assert trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs
    assert trace_types.ATTR_GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS not in attrs
    assert trace_types.ATTR_GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS not in attrs
    assert trace_types.ATTR_GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS not in attrs
