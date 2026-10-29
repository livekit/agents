from __future__ import annotations

import os
import traceback
from collections.abc import Mapping

from opentelemetry import trace

from livekit import rtc

from .. import _core
from ..types import ATTRIBUTE_REDACTION_ENABLED, NOT_GIVEN, NotGivenOr
from . import trace_types

REDACTED_EXCEPTION_MESSAGE = _core.REDACTED_EXCEPTION

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


def redaction_enabled(span_attributes: Mapping[str, object] | None = None) -> bool:
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


def participant_attributes(participant: rtc.Participant) -> dict[str, str]:
    """Span attributes identifying a room participant (identity is tagged ``lk.pii``)."""
    return {
        trace_types.ATTR_PARTICIPANT_ID: participant.sid,
        trace_types.ATTR_PARTICIPANT_IDENTITY: participant.identity,
        trace_types.ATTR_PARTICIPANT_KIND: rtc.ParticipantKind.Name(participant.kind),
    }


def record_exception(
    span: trace.Span,
    exception: BaseException,
    *,
    redacted: NotGivenOr[bool] = NOT_GIVEN,
    record_event: bool = True,
    set_status: bool = True,
) -> None:
    """Record an exception with redaction and independent event/status controls."""
    if not (record_event or set_status) or not span.is_recording():
        return
    if redacted is NOT_GIVEN:
        redacted = redaction_enabled()

    # `error.type` is the GenAI/HTTP conventions' low-cardinality error identifier;
    # unlike the message it never carries user data, so it is set either way
    from .traces import set_error_type

    set_error_type(span, exception)

    if set_status:
        message = REDACTED_EXCEPTION_MESSAGE if redacted else str(exception)
        span.set_status(trace.Status(trace.StatusCode.ERROR, message))
    if not record_event:
        return

    if redacted:
        attrs = {
            trace_types.ATTR_EXCEPTION_TYPE: exception.__class__.__name__,
            trace_types.ATTR_EXCEPTION_MESSAGE: REDACTED_EXCEPTION_MESSAGE,
        }
        span.add_event("exception", attrs)
        span.set_attributes(attrs)
        return

    span.record_exception(exception)
    # set the exception in span attributes in case the exception event is not rendered
    span.set_attributes(
        {
            trace_types.ATTR_EXCEPTION_TYPE: exception.__class__.__name__,
            trace_types.ATTR_EXCEPTION_MESSAGE: str(exception),
            trace_types.ATTR_EXCEPTION_TRACE: traceback.format_exc(),
        }
    )
