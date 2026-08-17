from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from opentelemetry import trace

from . import trace_types

if TYPE_CHECKING:
    from ..metrics import RealtimeModelMetrics


def record_exception(span: trace.Span, exception: Exception) -> None:
    span.record_exception(exception)
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))
    # set the exception in span attributes in case the exception event is not rendered
    span.set_attributes(
        {
            trace_types.ATTR_EXCEPTION_TYPE: exception.__class__.__name__,
            trace_types.ATTR_EXCEPTION_MESSAGE: str(exception),
            trace_types.ATTR_EXCEPTION_TRACE: traceback.format_exc(),
        }
    )


def record_realtime_metrics(span: trace.Span, ev: RealtimeModelMetrics) -> None:
    """Set OpenTelemetry GenAI usage attributes for Langfuse-compatible OTEL ingestion.

    Uses text token counts for top-level ``gen_ai.usage.input_tokens`` and
    ``gen_ai.usage.output_tokens`` (from ``input_token_details.text_tokens`` and
    ``output_token_details.text_tokens``). OpenAI Realtime reports those text counts
    inclusive of cached text; cache read for pricing breakdown is
    ``gen_ai.usage.cache_read.input_tokens`` from ``cached_tokens_details.text_tokens``.
    Audio tokens use ``gen_ai.usage.details.*`` keys.
    """
    model_name = ev.metadata.model_name if ev.metadata else None
    model_provider = ev.metadata.model_provider if ev.metadata else None

    cached = ev.input_token_details.cached_tokens_details
    cache_read_text = cached.text_tokens if cached else 0
    cache_read_audio = cached.audio_tokens if cached else 0

    attrs: dict[str, str | int] = {
        trace_types.ATTR_GEN_AI_OPERATION_NAME: "chat",
        trace_types.ATTR_GEN_AI_PROVIDER_NAME: model_provider or "unknown",
        trace_types.ATTR_GEN_AI_REQUEST_MODEL: model_name or "unknown",
        trace_types.ATTR_REALTIME_MODEL_METRICS: ev.model_dump_json(),
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: ev.input_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: ev.output_token_details.text_tokens,
    }
    if cache_read_text:
        attrs[trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = cache_read_text
    if ev.input_token_details.audio_tokens:
        attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_INPUT_AUDIO_TOKENS] = (
            ev.input_token_details.audio_tokens
        )
    if cache_read_audio:
        attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_CACHE_AUDIO_READ_TOKENS] = cache_read_audio
    if ev.output_token_details.audio_tokens:
        attrs[trace_types.ATTR_GEN_AI_USAGE_DETAILS_OUTPUT_AUDIO_TOKENS] = (
            ev.output_token_details.audio_tokens
        )
    if ev.ttft != -1:
        completion_start_time = ev.timestamp + ev.ttft
        # This attribute is used by LangFuse to calculate "time to first token metric"
        # in same way we calculate in livekit (ttft = first_token_timestamp - ev.timestamp)
        # So providing it explicitly here so we can graph and search by ttft.
        # Must be provided as UTC isoformat string for LangFuse
        completion_start_time_utc = datetime.fromtimestamp(
            completion_start_time, tz=timezone.utc
        ).isoformat()
        attrs[trace_types.ATTR_LANGFUSE_COMPLETION_START_TIME] = completion_start_time_utc
    if span.is_recording():
        span.set_attributes(attrs)
    else:
        from .traces import tracer

        # create a dedicated child span for orphaned metrics
        with trace.use_span(span):
            with tracer.start_span("realtime_metrics") as child:
                child.set_attributes(attrs)
