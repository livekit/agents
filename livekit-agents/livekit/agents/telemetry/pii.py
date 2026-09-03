from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from ..types import ATTRIBUTE_REDACTION_ENABLED
from . import trace_types

# LiveKit marks attributes carrying conversational content, tool payloads or other user
# data with a dot-delimited `pii` segment (`lk.pii.<name>`), which PII-enabled projects
# have stripped at the LiveKit Cloud collector. That only protects records reaching
# LiveKit Cloud, and the GenAI content attributes cannot carry the marker at all — the
# semantic convention fixes their names. So the stripping happens here as well, before
# any exporter observes the record.

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


def redact_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    return {k: v for k, v in attributes.items() if not is_pii_attribute(k)}


def contains_pii(attributes: Mapping[str, Any] | None) -> bool:
    return any(is_pii_attribute(k) for k in attributes or ())


def _span_redaction_enabled(attributes: Mapping[str, Any] | None) -> bool:
    """Resolved from the span's own attributes rather than the ambient job context: the
    flag is stamped at span start, so it stays correct when concurrent jobs share the
    providers (THREAD executor) and when a span ends off the job's thread."""
    if attributes and attributes.get(ATTRIBUTE_REDACTION_ENABLED):
        return True

    # spans created before the job registered its recording options, or outside a job
    # context entirely, fall back to the ambient context
    from .utils import _redaction_enabled

    return _redaction_enabled()


class PIIRedactingSpanProcessor(SpanProcessor):
    """Must be registered **before** any exporting processor: ``on_end`` is dispatched in
    registration order over one shared :class:`ReadableSpan` snapshot, so rewriting the
    snapshot here is what the downstream processors go on to export."""

    def on_start(self, span: Span, parent_context: otel_context.Context | None = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        attributes = span.attributes
        if not _span_redaction_enabled(attributes):
            return

        if contains_pii(attributes):
            # rebind rather than mutate: the snapshot's BoundedAttributes is shared with
            # the live span, and is immutable once the span has ended
            span._attributes = redact_attributes(attributes)

        events = span.events
        if events:
            redacted = _redact_events(events)
            if redacted is not None:
                span._events = redacted

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _redact_events(events: Sequence[Event]) -> list[Event] | None:
    """``None`` when nothing changed."""
    from opentelemetry.sdk.trace import Event as SdkEvent

    changed = False
    kept: list[Event] = []
    for event in events:
        if event.name in PII_EVENT_NAMES:
            changed = True
            continue
        if contains_pii(event.attributes):
            changed = True
            kept.append(
                SdkEvent(
                    name=event.name,
                    attributes=redact_attributes(event.attributes),
                    timestamp=event.timestamp,
                )
            )
            continue
        kept.append(event)

    return kept if changed else None
