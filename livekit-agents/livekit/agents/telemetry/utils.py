from __future__ import annotations

import os
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from ..types import ATTRIBUTE_REDACTION_ENABLED, NOT_GIVEN, NotGivenOr
from . import trace_types

if TYPE_CHECKING:
    from ..metrics import RealtimeModelMetrics


REDACTED_EXCEPTION_MESSAGE = "exception details redacted"

_ALLOW_PII_ENV_VAR = "LIVEKIT_TELEMETRY_ALLOW_PII"
_FALSY = ("0", "false", "no", "off")


def allow_pii_from_env() -> bool | None:
    """The ``LIVEKIT_TELEMETRY_ALLOW_PII`` setting, or ``None`` when unset.

    For integrators who let the framework adopt the ambient OpenTelemetry provider (a
    NodeSDK-style setup) and so have nowhere to pass ``allow_pii``. Set it to ``0`` to
    withhold conversational content from third-party exporters.
    """
    raw = os.environ.get(_ALLOW_PII_ENV_VAR)
    if raw is None:
        return None
    return raw.strip().lower() not in _FALSY


def redaction_enabled(span_attributes: Mapping[str, Any] | None = None) -> bool:
    """Whether the project has mandated PII redaction.

    Set in the LiveKit Cloud dashboard (or per session with
    ``record={"redaction": True}``) and never weakened from here — when it is on, PII is
    stripped for every destination, LiveKit Cloud included. ``span_attributes`` lets a
    span ended off the job's thread resolve from the flag stamped on it at span start.

    Stripping PII for *third-party* exporters is not this flag: that is the default, and
    is lifted per provider with ``set_tracer_provider(..., allow_pii=True)``.
    """
    if span_attributes and span_attributes.get(ATTRIBUTE_REDACTION_ENABLED):
        return True

    from ..job import get_job_context

    job_ctx = get_job_context(required=False)
    return job_ctx is not None and job_ctx._redaction_enabled


def record_exception(
    span: trace.Span, exception: Exception, *, redacted: NotGivenOr[bool] = NOT_GIVEN
) -> None:
    if redacted is NOT_GIVEN:
        redacted = redaction_enabled()

    # `error.type` is the GenAI/HTTP conventions' low-cardinality error identifier;
    # unlike the message it never carries user data, so it is set either way
    from .gen_ai import set_error_type

    set_error_type(span, exception)

    if redacted:
        attrs = {
            trace_types.ATTR_EXCEPTION_TYPE: exception.__class__.__name__,
            trace_types.ATTR_EXCEPTION_MESSAGE: REDACTED_EXCEPTION_MESSAGE,
        }
        span.add_event("exception", attrs)
        span.set_status(trace.Status(trace.StatusCode.ERROR, REDACTED_EXCEPTION_MESSAGE))
        span.set_attributes(attrs)
        return

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
    model_provider = ev.metadata.model_provider if ev.metadata else None

    attrs: dict[str, str | int | float | bool] = {
        # a realtime turn is a multimodal generation; `gen_ai.output.type` is set on the
        # inference span, which knows whether this session outputs audio
        trace_types.ATTR_GEN_AI_OPERATION_NAME: (trace_types.GenAIOperationName.GENERATE_CONTENT),
        trace_types.ATTR_GEN_AI_PROVIDER_NAME: (
            trace_types.gen_ai_provider_name(model_provider) or "unknown"
        ),
        trace_types.ATTR_GEN_AI_REQUEST_MODEL: model_name or "unknown",
        trace_types.ATTR_GEN_AI_RESPONSE_MODEL: model_name or "unknown",
        trace_types.ATTR_REALTIME_MODEL_METRICS: ev.model_dump_json(),
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: ev.input_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TOKENS: ev.output_tokens,
        trace_types.ATTR_GEN_AI_USAGE_TEXT_INPUT_TOKENS: ev.input_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_AUDIO_INPUT_TOKENS: ev.input_token_details.audio_tokens,
        trace_types.ATTR_GEN_AI_USAGE_IMAGE_INPUT_TOKENS: ev.input_token_details.image_tokens,
        trace_types.ATTR_GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: ev.output_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: ev.output_token_details.audio_tokens,
        trace_types.ATTR_GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: (
            ev.input_token_details.cached_tokens
        ),
        # unofficial spellings LangFuse reads, kept alongside the official ones
        trace_types.ATTR_GEN_AI_USAGE_INPUT_TEXT_TOKENS: ev.input_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_INPUT_AUDIO_TOKENS: ev.input_token_details.audio_tokens,
        trace_types.ATTR_GEN_AI_USAGE_INPUT_CACHED_TOKENS: ev.input_token_details.cached_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_TEXT_TOKENS: ev.output_token_details.text_tokens,
        trace_types.ATTR_GEN_AI_USAGE_OUTPUT_AUDIO_TOKENS: ev.output_token_details.audio_tokens,
    }
    if ev.request_id:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_ID] = ev.request_id
    if cached := ev.input_token_details.cached_tokens_details:
        attrs[trace_types.ATTR_GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS] = cached.text_tokens
        attrs[trace_types.ATTR_GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS] = cached.audio_tokens
        attrs[trace_types.ATTR_GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS] = cached.image_tokens
    if ev.ttft >= 0:
        attrs[trace_types.ATTR_GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = ev.ttft
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
        with tracer.use_span(span):
            with tracer.start_span("realtime_metrics") as child:
                child.set_attributes(attrs)
