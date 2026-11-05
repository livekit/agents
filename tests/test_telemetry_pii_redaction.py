from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from livekit.agents.telemetry import pii, trace_types, utils as telemetry_utils
from livekit.agents.telemetry.traces import _install_pii_redaction
from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED

pytestmark = pytest.mark.unit

# Pins the SDK-side guarantee: PII never reaches an exporter that is not LiveKit Cloud's,
# whose own handling is the project's setting in the dashboard. `allow_pii` lifts that per
# provider; the project flag overrides it and strips for every destination.

_PII_ATTRS = {
    trace_types.ATTR_CHAT_CTX: '{"items": []}',
    trace_types.ATTR_USER_TRANSCRIPT: "my card number is 4111",
    trace_types.ATTR_GEN_AI_INPUT_MESSAGES: '[{"role": "user"}]',
    trace_types.ATTR_GEN_AI_OUTPUT_MESSAGES: '[{"role": "assistant"}]',
    trace_types.ATTR_GEN_AI_SYSTEM_INSTRUCTIONS: '[{"type": "text"}]',
    trace_types.ATTR_GEN_AI_TOOL_CALL_ARGUMENTS: '{"location": "Paris"}',
    trace_types.ATTR_GEN_AI_TOOL_CALL_RESULT: '{"temp": 14}',
    # free-form, and recorded whenever the project allows it
    trace_types.ATTR_EXCEPTION_TRACE: 'Traceback: "my pin is 1234"',
}
_SAFE_ATTRS = {
    trace_types.ATTR_GEN_AI_OPERATION_NAME: "chat",
    trace_types.ATTR_GEN_AI_USAGE_INPUT_TOKENS: 100,
    trace_types.ATTR_SPEECH_ID: "speech_1",
}


def _emit(
    *, redaction: bool = False, exporter_first: bool = False, allow_pii: bool | None = None
) -> InMemorySpanExporter:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    if exporter_first:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        _install_pii_redaction(provider, allow_pii=allow_pii)
    else:
        _install_pii_redaction(provider, allow_pii=allow_pii)
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    with provider.get_tracer(__name__).start_as_current_span("llm_request") as span:
        if redaction:
            span.set_attribute(ATTRIBUTE_REDACTION_ENABLED, True)
        span.set_attributes({**_PII_ATTRS, **_SAFE_ATTRS})
        span.add_event(trace_types.EVENT_GEN_AI_USER_MESSAGE, {"content": "my pin is 1234"})
        span.add_event("llm_started", {trace_types.ATTR_INSTRUCTIONS: "be brief", "n": 1})
    return exporter


def test_pii_is_withheld_from_third_party_exporters_on_request() -> None:
    span = _emit(allow_pii=False).get_finished_spans()[0]

    leaked = sorted(set(_PII_ATTRS) & set(span.attributes or {}))
    assert not leaked, f"leaked: {leaked}"
    for key, value in _SAFE_ATTRS.items():
        assert (span.attributes or {})[key] == value

    events = {e.name: dict(e.attributes or {}) for e in span.events}
    # the GenAI content events carry the body in a generic `content` attribute that
    # cannot be marked, so the whole event goes
    assert trace_types.EVENT_GEN_AI_USER_MESSAGE not in events
    # a non-content event keeps its safe attributes and loses its PII ones
    assert events["llm_started"] == {"n": 1}


def test_exporters_receive_pii_by_default() -> None:
    # the GenAI conventions are only useful to a backend that can render the conversation
    span = _emit().get_finished_spans()[0]

    for key, value in _PII_ATTRS.items():
        assert (span.attributes or {})[key] == value
    assert trace_types.EVENT_GEN_AI_USER_MESSAGE in {e.name for e in span.events}


def test_livekit_cloud_still_receives_pii() -> None:
    # what Cloud may keep is decided at its collector, from the project's setting
    exporter = _emit(allow_pii=False)
    stripped = exporter.get_finished_spans()[0]
    restored = pii.restore_pii(stripped)

    for key, value in _PII_ATTRS.items():
        assert (restored.attributes or {})[key] == value
    assert trace_types.EVENT_GEN_AI_USER_MESSAGE in {e.name for e in restored.events}


def test_the_project_flag_withholds_pii_from_every_destination() -> None:
    # redaction mandated in the dashboard is not weakened by a local grant, and nothing is
    # stashed for LiveKit Cloud to restore
    stripped = _emit(allow_pii=True, redaction=True).get_finished_spans()[0]

    assert not set(_PII_ATTRS) & set(stripped.attributes or {})
    assert pii.restore_pii(stripped) is stripped


def test_redaction_runs_ahead_of_an_exporter_attached_first() -> None:
    # an integrator attaches their Datadog/Langfuse exporter, then hands us the provider
    span = _emit(allow_pii=False, exporter_first=True).get_finished_spans()[0]

    assert not set(_PII_ATTRS) & set(span.attributes or {})


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("lk.pii.chat_ctx", True),
        ("gen_ai.input.messages", True),
        ("gen_ai.prompt.variable.customer_name", True),
        ("gen_ai.usage.input_tokens", False),
        # a `pii` substring is not a `pii` segment
        ("lk.piidata.x", False),
    ],
)
def test_pii_classification(key: str, expected: bool) -> None:
    assert pii.is_pii_attribute(key) is expected


def test_logs_are_filtered_for_every_destination_when_redaction_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client filters the new keys itself rather than relying on a collector to know
    them, so an enabled project setting strips logs on every exporter, LiveKit Cloud's
    included."""
    monkeypatch.setattr("livekit.agents.telemetry.utils.redaction_enabled", lambda *_: True)

    record = SimpleNamespace(
        attributes={
            trace_types.ATTR_CHAT_CTX: '{"items": []}',
            trace_types.ATTR_GEN_AI_INPUT_MESSAGES: '[{"role": "user"}]',
            trace_types.ATTR_EXCEPTION_MESSAGE: "secret transcript",
            trace_types.ATTR_EXCEPTION_TRACE: 'Traceback: "my pin is 1234"',
            "function": "get_weather",
        }
    )
    pii._PIIFilteringLogProcessor().on_emit(SimpleNamespace(log_record=record))  # type: ignore[arg-type]

    assert record.attributes == {
        "function": "get_weather",
        trace_types.ATTR_EXCEPTION_MESSAGE: telemetry_utils.REDACTED_EXCEPTION_MESSAGE,
    }


def test_exception_details_are_withheld_from_third_party_exporters() -> None:
    """record_exception resolves the project's setting, so with redaction off it writes the
    real message onto the span, its `exception` event and the span status. None of that may
    reach an exporter that is not LiveKit Cloud's."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    _install_pii_redaction(provider, allow_pii=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    with provider.get_tracer(__name__).start_as_current_span("llm_request") as span:
        try:
            raise RuntimeError("my pin is 1234")
        except RuntimeError as exc:
            telemetry_utils.record_exception(span, exc, redacted=False)

    exported = exporter.get_finished_spans()[0]
    serialized = json.dumps(
        [
            dict(exported.attributes or {}),
            [dict(e.attributes or {}) for e in exported.events],
            exported.status.description,
        ]
    )
    assert "my pin is 1234" not in serialized
    # the class still identifies the failure
    assert (exported.attributes or {})[trace_types.ATTR_ERROR_TYPE] == "RuntimeError"

    # ... while LiveKit Cloud still receives all three, per the project's setting
    restored = pii.restore_pii(exported)
    assert restored.status.description == "my pin is 1234"
    assert (restored.attributes or {})[trace_types.ATTR_EXCEPTION_MESSAGE] == "my pin is 1234"
    assert any("my pin is 1234" in str(dict(e.attributes or {})) for e in restored.events)
