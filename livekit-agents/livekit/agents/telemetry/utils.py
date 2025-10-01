from __future__ import annotations

import traceback
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
    model_name = ev.metadata.model_name if ev.metadata else None
    attrs: dict[str, str | int] = {
        trace_types.ATTR_GEN_AI_REQUEST_MODEL: model_name or "unknown",
        trace_types.ATTR_REALTIME_MODEL_METRICS: ev.model_dump_json(),
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: ev.input_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: ev.output_tokens,
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS: ev.input_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS: ev.input_token_details.audio_tokens,
        trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS: ev.input_token_details.cached_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS: ev.output_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS: ev.output_token_details.audio_tokens,
    }

    if span.is_recording():
        span.set_attributes(attrs)
    else:
        from .traces import tracer

        # create a dedicated child span for orphaned metrics
        with trace.use_span(span):
            with tracer.start_span("realtime_metrics") as child:
                child.set_attributes(attrs)
