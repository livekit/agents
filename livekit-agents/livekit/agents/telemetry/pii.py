from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry.sdk._logs import LogRecordProcessor, ReadWriteLogRecord
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.trace import Status, StatusCode

from . import trace_types
from .utils import REDACTED_EXCEPTION_MESSAGE

# Field-level filtering, not entity-level redaction: a matching attribute is dropped
# whole, never scanned and masked. "Redaction" in this codebase means the project setting
# (LiveKit Cloud dashboard, or record={"redaction": True}); this module is what the client
# does about it.
#
# LiveKit marks attributes carrying conversational content, tool payloads or other user
# data with a dot-delimited `pii` segment (`lk.pii.<name>`), and the GenAI content
# attributes carry the same payload under names the semantic convention fixes, where the
# marker cannot be applied. Both are filtered here, before any exporter that is not
# LiveKit Cloud's — and before every exporter, Cloud included, once the project has
# enabled redaction, so the client never depends on a collector to strip a new key.

# exception details are recorded by `record_exception`, which resolves redaction from the
# ambient job context; that can disagree with the span's own stamp, so they are filtered
# here as well
_REDACTED_EXCEPTION_ATTRIBUTES = (
    trace_types.ATTR_EXCEPTION_MESSAGE,
    trace_types.ATTR_EXCEPTION_TRACE,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Event


GEN_AI_PII_ATTRIBUTES: frozenset[str] = frozenset(
    {
        # flagged "likely to contain sensitive information including user/PII data"
        # by the GenAI semantic conventions
        trace_types.ATTR_GEN_AI_INPUT_MESSAGES,
        trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES,
        trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS,
        trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS,
        trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT,
        trace_types.ATTR_GEN_AI_TOOL_DESCRIPTION,
        trace_types.ATTR_GEN_AI_TOOL_DEFINITIONS,
        # free-form text the caller supplied or the model produced
        trace_types.ATTR_GEN_AI_RETRIEVAL_QUERY_TEXT,
        trace_types.ATTR_GEN_AI_RETRIEVAL_DOCUMENTS,
        trace_types.ATTR_GEN_AI_MEMORY_QUERY_TEXT,
        trace_types.ATTR_GEN_AI_MEMORY_RECORDS,
        trace_types.ATTR_GEN_AI_EVALUATION_EXPLANATION,
    }
)
"""GenAI attributes carrying content. Fixed names, so they cannot be ``lk.pii.``-marked."""

PII_EVENT_NAMES: frozenset[str] = frozenset(
    {
        # the GenAI content events carry the message body in a generic `content`
        # attribute, so the whole event is dropped rather than filtered
        trace_types.EVENT_GEN_AI_SYSTEM_MESSAGE,
        trace_types.EVENT_GEN_AI_USER_MESSAGE,
        trace_types.EVENT_GEN_AI_ASSISTANT_MESSAGE,
        trace_types.EVENT_GEN_AI_TOOL_MESSAGE,
        trace_types.EVENT_GEN_AI_CHOICE,
        trace_types.EVENT_GEN_AI_CLIENT_INFERENCE_OPERATION_DETAILS,
    }
)


def is_pii_attribute(key: str) -> bool:
    if "pii" in key.split("."):
        return True
    if key in GEN_AI_PII_ATTRIBUTES:
        return True
    # gen_ai.prompt.variable.<key> holds the values interpolated into a prompt template
    return key.startswith(trace_types.ATTR_GEN_AI_PROMPT_VARIABLE)


def filter_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}

    redacted: dict[str, Any] = {}
    for key, value in attributes.items():
        if is_pii_attribute(key):
            continue
        if key == trace_types.ATTR_EXCEPTION_TRACE:
            continue
        if key == trace_types.ATTR_EXCEPTION_MESSAGE:
            # `error.type` still names the class; only the free-form message goes
            redacted[key] = REDACTED_EXCEPTION_MESSAGE
            continue
        redacted[key] = value
    return redacted


def _contains_pii(attributes: Mapping[str, Any] | None) -> bool:
    return any(is_pii_attribute(k) or k in _REDACTED_EXCEPTION_ATTRIBUTES for k in attributes or ())


_RAW_ATTRIBUTES = "_lk_raw_attributes"
_RAW_EVENTS = "_lk_raw_events"
_RAW_STATUS = "_lk_raw_status"


class _PIIFilteringSpanProcessor(SpanProcessor):
    """Drops PII attributes so they never reach an exporter that is not LiveKit Cloud's.

    Must be registered **before** any exporting processor: ``on_end`` is dispatched in
    registration order over one shared :class:`ReadableSpan` snapshot, so rewriting the
    snapshot here is what the downstream processors go on to export.

    LiveKit Cloud is exempt — its PII handling is the project's setting in the dashboard,
    not ours to second-guess — so the original attributes are stashed on the snapshot for
    :func:`restore_pii` to put back on that one export path. When the project does mandate
    redaction, nothing is stashed and Cloud is stripped along with everyone else.

    ``allow_pii`` lifts the stripping for a provider whose exporters the integrator has
    explicitly granted PII (``set_tracer_provider(..., allow_pii=True)``).
    """

    def __init__(self, *, allow_pii: bool = False) -> None:
        self._allow_pii = allow_pii

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        from .utils import redaction_enabled

        attributes = span.attributes
        # the flag stamped at span start keeps this correct for a span ended off the
        # job's thread, and when concurrent jobs share the providers (THREAD executor)
        project_redaction = redaction_enabled(attributes)
        if self._allow_pii and not project_redaction:
            return

        events = span.events
        filtered_events = _filter_events(events)
        if not _contains_pii(attributes) and filtered_events is None:
            return

        if not project_redaction:
            # LiveKit Cloud still receives what the project allows
            setattr(span, _RAW_ATTRIBUTES, dict(attributes or {}))
            setattr(span, _RAW_EVENTS, tuple(events))
            setattr(span, _RAW_STATUS, span.status)

        # rebind rather than mutate: the snapshot's BoundedAttributes is shared with the
        # live span, and is immutable once the span has ended
        span._attributes = filter_attributes(attributes)
        if filtered_events is not None:
            span._events = filtered_events
        # record_exception also puts the message in the span status
        if span.status.status_code is StatusCode.ERROR and span.status.description:
            span._status = Status(StatusCode.ERROR, REDACTED_EXCEPTION_MESSAGE)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def restore_pii(span: ReadableSpan) -> ReadableSpan:
    """The span as it was before PII was stripped for third-party exporters.

    Only LiveKit Cloud's own exporter calls this; every other destination sees the
    redacted snapshot. Returns ``span`` unchanged when nothing was stashed.
    """
    attributes = getattr(span, _RAW_ATTRIBUTES, None)
    if attributes is None:
        return span

    from .utils import redaction_enabled

    # nothing is stashed once the project mandates redaction, so this is belt-and-braces:
    # it keeps a change to that upstream check from turning into a leak here
    if redaction_enabled(span.attributes):
        return span

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=getattr(span, _RAW_EVENTS, span.events),
        links=span.links,
        kind=span.kind,
        status=getattr(span, _RAW_STATUS, span.status),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class _PIIFilteringLogProcessor(LogRecordProcessor):
    """Log counterpart of :class:`_PIIFilteringSpanProcessor`.

    ``_TraceLevelLoggingHandler`` redacts the records the framework's own handler creates;
    this covers every other emitter on the logger provider, including the exporters an
    integrator attached before handing us the provider.
    """

    def on_emit(self, log_data: ReadWriteLogRecord) -> None:
        # on_emit runs synchronously on the emitting thread, so the ambient job context is
        # the right source here — unlike spans, a log record has no earlier stamp to read
        from .utils import redaction_enabled

        if not redaction_enabled():
            return

        record = log_data.log_record
        if _contains_pii(record.attributes):
            record.attributes = filter_attributes(record.attributes)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _filter_events(events: Sequence[Event]) -> list[Event] | None:
    """``None`` when nothing changed."""
    from opentelemetry.sdk.trace import Event as SdkEvent

    changed = False
    kept: list[Event] = []
    for event in events:
        if event.name in PII_EVENT_NAMES:
            changed = True
            continue
        if _contains_pii(event.attributes):
            changed = True
            kept.append(
                SdkEvent(
                    name=event.name,
                    attributes=filter_attributes(event.attributes),
                    timestamp=event.timestamp,
                )
            )
            continue
        kept.append(event)

    return kept if changed else None
