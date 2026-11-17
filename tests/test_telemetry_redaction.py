"""PII and credentials stay out of traces, logs and exceptions."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common._log_encoder import encode_logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, SimpleLogRecordProcessor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from yarl import URL

from livekit.agents import Agent, AgentSession, APIStatusError, function_tool, llm, utils
from livekit.agents.telemetry import pii, trace_types, utils as telemetry_utils
from livekit.agents.telemetry.traces import (
    _DynamicTracer,
    _install_pii_redaction,
    _TraceLevelLoggingHandler,
)
from livekit.agents.types import ATTRIBUTE_REDACTION_ENABLED, NOT_GIVEN, NotGivenOr

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = pytest.mark.unit


# -- span and log stripping --

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
    """Redaction strips logs in the client, so every exporter, Cloud's included, gets them clean."""
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
    """With redaction off the real message is recorded, but only LiveKit Cloud receives it."""
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


# -- every trace_types key is classified --

# a key's value carries a `pii` segment, is in pii.GEN_AI_PII_ATTRIBUTES, or is listed safe below

# Mirrors the collector's matcher: a whole dot-delimited "pii" segment,
# case-insensitive ("lk.chatpii" doesn't match, "lk.PII.x" does).
PII_SEGMENT_RE = re.compile(r"(^|\.)pii(\.|$)", re.IGNORECASE)

SENSITIVE_FIELDS = frozenset(
    {
        "arguments",
        "chat_ctx",
        "interim_transcript",
        "progress_message",
        "raw_arguments",
        "repaired",
        "transcript",
    }
)

# Keys that carry no conversational content, tool payloads, or other user data.
# A new constant must go either here or (if it can carry such data) adopt a
# ``pii`` dot-segment in its value — never both.
SAFE_KEYS = frozenset(
    {
        # correlation ids / session metadata
        "lk.speech_id",
        "lk.agent_label",
        "lk.retry_count",
        "lk.provider_request_ids",
        "lk.participant_id",
        "lk.participant_kind",
        "lk.job_id",
        "lk.agent_name",
        "lk.cloud_agent_id",
        "lk.deployment_id",
        # input delta markers: a flag, a count and span ids
        "lk.input.delta",
        "lk.input.base_span_id",
        "lk.input.dropped_from_base",
        "lk.generation_id",
        "lk.generation_count",
        "lk.parent_generation_id",
        "lk.interrupted",
        # llm node (tool *names* / schemas, not payloads)
        "lk.function_tools",
        "lk.provider_tools",
        "lk.tool_sets",
        "lk.response.ttft",
        # function tool metadata
        "lk.function_tool.id",
        "lk.function_tool.name",
        "lk.function_tool.is_error",
        # tts node
        "lk.tts.streaming",
        "lk.tts.label",
        "lk.response.ttfb",
        # eou detection (model identifiers / numeric / enum)
        "lk.eou.model",
        "lk.eou.provider",
        "lk.eou.probability",
        "lk.eou.unlikely_threshold",
        "lk.eou.endpointing_delay",
        "lk.eou.language",
        "lk.transcript_confidence",
        "lk.stt.events",
        "lk.transcription_delay",
        "lk.end_of_turn_delay",
        "lk.eou.source",
        "lk.eou.detection_delay",
        "lk.eou.from_cache",
        # metrics blobs (numeric)
        "lk.llm_metrics",
        "lk.tts_metrics",
        "lk.realtime_model_metrics",
        "lk.e2e_latency",
        # OTEL GenAI semconv (message content rides on event attributes
        # `content`/`tool_calls`, which the collector strips by name)
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.input_text_tokens",
        "gen_ai.usage.input_audio_tokens",
        "gen_ai.usage.input_cached_tokens",
        "gen_ai.usage.cache_read.input_tokens",
        "gen_ai.usage.output_text_tokens",
        "gen_ai.usage.output_audio_tokens",
        "gen_ai.usage.reasoning_tokens",
        "gen_ai.system.message",
        "gen_ai.user.message",
        "gen_ai.assistant.message",
        "gen_ai.tool.message",
        "gen_ai.choice",
        # OTEL exception semconv
        "exception.stacktrace",
        "exception.type",
        "exception.message",
        # vendor
        "langfuse.observation.completion_start_time",
        # amd (category/timings; transcript is tagged)
        "lk.amd.category",
        "lk.amd.reason",
        "lk.amd.speech_duration",
        "lk.amd.delay",
        # adaptive interruption (numeric)
        "lk.is_interruption",
        "lk.interruption.probability",
        "lk.interruption.total_duration",
        "lk.interruption.prediction_duration",
        "lk.interruption.detection_delay",
        # -- OTel GenAI semantic conventions ---------------------------------
        # identifiers, enums, counts and sampling settings; nothing free-form.
        # The content-bearing gen_ai attributes live in pii.GEN_AI_PII_ATTRIBUTES.
        "error.type",
        "gen_ai.agent.name",
        "gen_ai.conversation.id",
        "gen_ai.output.type",
        "gen_ai.request.stream",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.response.time_to_first_chunk",
        "gen_ai.token.type",
        "gen_ai.tool.call.id",
        "gen_ai.tool.name",
        "gen_ai.tool.type",
        "gen_ai.usage.audio.cache_read.input_tokens",
        "gen_ai.usage.audio.input_tokens",
        "gen_ai.usage.audio.output_tokens",
        "gen_ai.usage.cache_write.input_tokens",
        "gen_ai.usage.image.cache_read.input_tokens",
        "gen_ai.usage.image.input_tokens",
        "gen_ai.usage.reasoning.output_tokens",
        "gen_ai.usage.text.cache_read.input_tokens",
        "gen_ai.usage.text.input_tokens",
        "gen_ai.usage.text.output_tokens",
        "gen_ai.workflow.name",
        # a judge's name and its verdict enum (the reasoning is content, registered as PII)
        "gen_ai.evaluation.name",
        "gen_ai.evaluation.score.label",
        # GenAI event and metric names (not attribute keys)
        "gen_ai.client.inference.operation.details",
        "gen_ai.client.operation.duration",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.token.usage",
        "gen_ai.execute_tool.duration",
        "gen_ai.invoke_agent.duration",
        # interruptions, handoff, fallback, text input (enums, labels, sizes)
        "lk.interruption.source",
        "lk.playout.position",
        "lk.previous_agent_label",
        "lk.fallback.label",
        "lk.fallback.index",
        "lk.fallback.recovery",
        # rpc (semconv names, ids, sizes, codes; identities and payload are tagged)
        "rpc.method",
        "lk.rpc.request_id",
        "lk.rpc.caller_identity",
        "lk.rpc.destination_identity",
        "lk.rpc.payload_size",
        "lk.rpc.response_size",
        "lk.rpc.response_timeout",
        "lk.rpc.error_code",
        "lk.rpc.handler_registered",
        # join keys, dispatch timeline, room/track ids, lifecycle enums (no content)
        "lk.room_sid",
        "lk.dispatch_id",
        "lk.job.worker_id",
        "lk.job.agent_id",
        "lk.sip.",
        "lk.job.accept_latency",
        "lk.job.assignment_latency",
        "lk.job.launch_latency",
        "lk.job.entrypoint_latency",
        "lk.job.dispatch_latency",
        "lk.keyterms.count",
        "lk.keyterms.added",
        "lk.keyterms.removed",
        "lk.room.auto_subscribe",
        "lk.room.e2ee",
        "lk.room.remote_participant_count",
        "lk.room_io.participant_filter",
        "lk.track_sid",
        "lk.track_source",
        "lk.first_frame_delay",
        "lk.pre_connect_audio.duration",
        "lk.connection_state",
        "lk.disconnect_reason",
        "lk.old_state",
        "lk.new_state",
        "lk.close_reason",
        "lk.close.drain",
        "lk.shutdown.reason",
        "lk.shutdown.user_initiated",
        "lk.callback.name",
        # eou_wait / speech scheduling (timings and enums)
        "lk.eou.outcome",
        "lk.eou.wait_duration",
        "lk.eou.rearm_count",
        "lk.eou.not_committed_count",
        "lk.eou.resume_count",
        "lk.on_user_turn_completed_delay",
        "lk.speech.queue_wait",
        # event loop blocking (timings, task name, source-location stack; no values)
        "lk.blocking.duration",
        "lk.blocking.threshold",
        "lk.blocking.severity",
        "lk.blocking.task",
        "lk.blocking.stack",
        "lk.blocking.import",
        "lk.blocking.gc_time",
        "lk.blocking.cpu_time",
        "lk.blocking.suppressed",
        "lk.blocking.count",
        "lk.blocking.total_duration",
        "lk.blocking.max_duration",
    }
)


def _declared_keys() -> dict[str, str]:
    return {
        name: value
        for name, value in vars(trace_types).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def test_every_key_is_classified() -> None:
    unclassified = {
        name: value
        for name, value in _declared_keys().items()
        if value not in SAFE_KEYS
        and not PII_SEGMENT_RE.search(value)
        and not pii.is_pii_attribute(value)
    }
    assert not unclassified, (
        f"unclassified telemetry keys: {unclassified}. If the attribute can carry "
        "conversational content, tool payloads, or other user data, include a "
        "dot-delimited `pii` segment in its value (e.g. lk.pii.<name>) — or, for a "
        "gen_ai.* name fixed by the semantic convention, register it in "
        "telemetry.pii.GEN_AI_PII_ATTRIBUTES; otherwise add it to SAFE_KEYS in "
        "this test."
    )


def test_safe_list_and_the_in_process_stripper_agree() -> None:
    stripped = sorted(k for k in SAFE_KEYS if pii.is_pii_attribute(k))
    assert not stripped, f"keys listed as safe but stripped by telemetry.pii: {stripped}"

    # an integrator's own exporter never reaches the LiveKit Cloud collector, so every
    # pii-marked key must be stripped in-process too
    marked = [v for v in _declared_keys().values() if PII_SEGMENT_RE.search(v)]
    assert marked, "expected trace_types to declare pii-marked attributes"
    missed = sorted(k for k in marked if not pii.is_pii_attribute(k))
    assert not missed, f"pii-marked keys the in-process stripper misses: {missed}"

    # a rename must not silently drop an attribute out of the stripped set
    stale = sorted(pii.GEN_AI_PII_ATTRIBUTES - set(_declared_keys().values()))
    assert not stale, f"GEN_AI_PII_ATTRIBUTES entries not declared in trace_types.py: {stale}"


def test_safe_keys_do_not_carry_pii_segment() -> None:
    conflicting = sorted(k for k in SAFE_KEYS if PII_SEGMENT_RE.search(k))
    assert not conflicting, (
        f"keys listed as safe but carrying a `pii` segment: {conflicting}. "
        "A key is either safe-listed or pii-tagged, never both."
    )


def test_safe_keys_match_declared_keys() -> None:
    declared = set(_declared_keys().values())
    stale = sorted(SAFE_KEYS - declared)
    assert not stale, (
        f"SAFE_KEYS entries no longer declared in trace_types.py: {stale}. "
        "Remove them so the safe list stays an exact inventory."
    )


def test_sensitive_literal_telemetry_fields_carry_pii_segment() -> None:
    repo_root = Path(__file__).parent.parent
    source_roots = [repo_root / "livekit-agents", repo_root / "livekit-plugins"]
    untagged: list[str] = []
    candidate_key = re.compile(
        r"[\"'](?:[^\"'\n]*\.)?(?:" + "|".join(SENSITIVE_FIELDS) + r")[\"']\s*:"
    )

    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            text = path.read_text()
            # parsing is the slow part: only files with a sensitive dict key can fail
            if not candidate_key.search(text):
                continue
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue

                keyword_name: str | None = None
                if node.func.attr in {
                    "debug",
                    "info",
                    "warning",
                    "error",
                    "exception",
                    "critical",
                    "log",
                }:
                    keyword_name = "extra"
                elif (
                    node.func.attr == "add"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "tagger"
                ):
                    keyword_name = "metadata"

                if keyword_name is None:
                    continue

                for keyword in node.keywords:
                    if keyword.arg != keyword_name or not isinstance(keyword.value, ast.Dict):
                        continue
                    for key in keyword.value.keys:
                        if not (
                            isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                            and key.value.rsplit(".", 1)[-1] in SENSITIVE_FIELDS
                            and not PII_SEGMENT_RE.search(key.value)
                        ):
                            continue
                        relative_path = path.relative_to(repo_root)
                        untagged.append(f"{relative_path}:{node.lineno}: {key.value}")

    assert not untagged, f"content-bearing telemetry fields missing pii segment: {untagged}"


# -- exception details --


class _FakeSpan:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.attributes: dict[str, Any] = {}
        self.recorded_exceptions: list[Exception] = []
        self.status: trace.Status | None = None

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, attributes))

    def record_exception(self, exception: Exception) -> None:
        self.recorded_exceptions.append(exception)

    def set_status(self, status: trace.Status) -> None:
        self.status = status

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)


def _capture_exception(span: _FakeSpan, *, redacted: NotGivenOr[bool]) -> None:
    try:
        raise RuntimeError("secret transcript")
    except RuntimeError as exc:
        telemetry_utils.record_exception(span, exc, redacted=redacted)  # type: ignore[arg-type]


def test_record_exception_preserves_details_when_not_redacted() -> None:
    span = _FakeSpan()

    _capture_exception(span, redacted=False)

    assert len(span.recorded_exceptions) == 1
    assert span.attributes[trace_types.ATTR_EXCEPTION_TYPE] == "RuntimeError"
    # the GenAI/HTTP conventions' low-cardinality error identifier
    assert span.attributes[trace_types.ATTR_ERROR_TYPE] == "RuntimeError"
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == "secret transcript"
    assert "secret transcript" in span.attributes[trace_types.ATTR_EXCEPTION_TRACE]
    assert span.status is not None
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == "secret transcript"


def test_record_exception_omits_details_when_redacted() -> None:
    span = _FakeSpan()

    _capture_exception(span, redacted=True)

    assert span.recorded_exceptions == []
    # `error.type` names the exception class, never its message, so it survives redaction
    assert span.attributes == {
        trace_types.ATTR_ERROR_TYPE: "RuntimeError",
        trace_types.ATTR_EXCEPTION_TYPE: "RuntimeError",
        trace_types.ATTR_EXCEPTION_MESSAGE: telemetry_utils.REDACTED_EXCEPTION_MESSAGE,
    }
    assert trace_types.ATTR_EXCEPTION_TRACE not in span.attributes
    assert span.events == [
        (
            "exception",
            {
                trace_types.ATTR_EXCEPTION_TYPE: "RuntimeError",
                trace_types.ATTR_EXCEPTION_MESSAGE: telemetry_utils.REDACTED_EXCEPTION_MESSAGE,
            },
        )
    ]
    assert span.status is not None
    assert span.status.status_code == trace.StatusCode.ERROR
    assert span.status.description == telemetry_utils.REDACTED_EXCEPTION_MESSAGE


def test_record_exception_uses_resolved_redaction_state(monkeypatch: pytest.MonkeyPatch) -> None:
    span = _FakeSpan()

    def get_job_context(*, required: bool = True) -> SimpleNamespace:
        return SimpleNamespace(_redaction_enabled=True)

    monkeypatch.setattr("livekit.agents.job.get_job_context", get_job_context)
    _capture_exception(span, redacted=NOT_GIVEN)

    assert span.recorded_exceptions == []
    assert span.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
        telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    )
    assert trace_types.ATTR_EXCEPTION_TRACE not in span.attributes


@pytest.mark.parametrize("use_span", [False, True], ids=["start_as_current_span", "use_span"])
@pytest.mark.parametrize("redaction_enabled", [False, True])
@pytest.mark.parametrize("record_exception", [False, True])
@pytest.mark.parametrize("set_status_on_exception", [False, True])
@pytest.mark.parametrize("end_on_exit", [False, True])
@pytest.mark.parametrize("call_style", ["keyword", "positional", "mixed"])
def test_dynamic_tracer_records_exceptions_with_caller_options(
    monkeypatch: pytest.MonkeyPatch,
    use_span: bool,
    redaction_enabled: bool,
    record_exception: bool,
    set_status_on_exception: bool,
    end_on_exit: bool,
    call_style: str,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-exception-redaction")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)
    options = {
        "record_exception": record_exception,
        "set_status_on_exception": set_status_on_exception,
        "end_on_exit": end_on_exit,
    }
    if call_style == "keyword":
        manager = (
            dynamic_tracer.use_span(dynamic_tracer.start_span("test-span"), **options)
            if use_span
            else dynamic_tracer.start_as_current_span("test-span", **options)
        )
    else:
        args: list[Any]
        if use_span:
            context_manager = dynamic_tracer.use_span
            args = [dynamic_tracer.start_span("test-span"), end_on_exit, record_exception]
            options = {"set_status_on_exception": set_status_on_exception}
        else:
            context_manager = dynamic_tracer.start_as_current_span
            args = ["test-span", None, trace.SpanKind.INTERNAL, None, None, None, record_exception]
            options = {
                "set_status_on_exception": set_status_on_exception,
                "end_on_exit": end_on_exit,
            }
        if call_style == "positional":
            args.extend(options.values())
            options = {}
        manager = context_manager(*args, **options)
    previous_span = trace.get_current_span()
    failure = RuntimeError("secret transcript")
    try:
        with pytest.raises(RuntimeError) as raised:
            with manager as span:
                assert trace.get_current_span() is span
                raise failure
        assert raised.value is failure
        assert trace.get_current_span() is previous_span
        assert span.is_recording() is (not end_on_exit)
        if not end_on_exit:
            span.end()

        (finished,) = exporter.get_finished_spans()
        events = [event for event in finished.events if event.name == "exception"]
        assert len(events) == int(record_exception)
        message = (
            telemetry_utils.REDACTED_EXCEPTION_MESSAGE if redaction_enabled else "secret transcript"
        )
        if record_exception:
            assert events[0].attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == message
        assert finished.status.status_code is (
            trace.StatusCode.ERROR if set_status_on_exception else trace.StatusCode.UNSET
        )
        assert finished.status.description == (message if set_status_on_exception else None)
        attrs = finished.attributes or {}
        assert attrs.get(trace_types.ATTR_ERROR_TYPE) == (
            "RuntimeError" if record_exception or set_status_on_exception else None
        )
        if not record_exception:
            assert trace_types.ATTR_EXCEPTION_MESSAGE not in attrs
        if redaction_enabled:
            assert "secret transcript" not in finished.to_json()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("use_span", [False, True], ids=["start_as_current_span", "use_span"])
@pytest.mark.parametrize("redaction_enabled", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_dynamic_tracer_decorator_preserves_logging_and_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    use_span: bool,
    redaction_enabled: bool,
    cancelled: bool,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    dynamic_tracer = _DynamicTracer("test-exception-decorator")
    dynamic_tracer.set_provider(provider)
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)
    manager = (
        dynamic_tracer.use_span(dynamic_tracer.start_span("test-span"), end_on_exit=True)
        if use_span
        else dynamic_tracer.start_as_current_span("test-span")
    )
    logger = logging.getLogger("test.exception-decorator")
    failure = asyncio.CancelledError() if cancelled else RuntimeError("secret transcript")

    @utils.log_exceptions(logger=logger)
    @manager
    async def failing_task() -> None:
        await asyncio.sleep(0)
        raise failure

    try:
        with pytest.raises(type(failure)) as raised:
            await failing_task()
        assert raised.value is failure
        (span,) = exporter.get_finished_spans()
        events = [event for event in span.events if event.name == "exception"]
        logs = [record for record in caplog.records if record.name == logger.name]
        assert len(events) == len(logs) == (0 if cancelled else 1)
        assert span.status.status_code is (
            trace.StatusCode.UNSET if cancelled else trace.StatusCode.ERROR
        )
        if not cancelled:
            assert logs[0].getMessage() == "Error in failing_task"
            assert logs[0].exc_info[1] is failure
    finally:
        provider.shutdown()


@pytest.mark.parametrize("redaction_enabled", [False, True])
def test_logging_handler_omits_automatic_exception_details_when_redacted(
    monkeypatch: pytest.MonkeyPatch, redaction_enabled: bool
) -> None:
    try:
        raise RuntimeError("secret transcript")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="livekit.agents.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="operation failed",
        args=(),
        exc_info=exc_info,
    )
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: redaction_enabled)

    translated = _TraceLevelLoggingHandler()._translate(record)
    assert translated.attributes is not None
    assert translated.attributes[trace_types.ATTR_EXCEPTION_TYPE] == "RuntimeError"
    if redaction_enabled:
        assert translated.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
            telemetry_utils.REDACTED_EXCEPTION_MESSAGE
        )
        assert trace_types.ATTR_EXCEPTION_TRACE not in translated.attributes
    else:
        assert translated.attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == "secret transcript"
        assert "secret transcript" in translated.attributes[trace_types.ATTR_EXCEPTION_TRACE]


def test_redacted_exception_log_can_be_otlp_encoded(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    handler = _TraceLevelLoggingHandler(logger_provider=provider)

    try:
        raise RuntimeError("secret transcript")
    except RuntimeError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="livekit.agents.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="operation failed",
        args=(),
        exc_info=exc_info,
    )
    monkeypatch.setattr(telemetry_utils, "redaction_enabled", lambda *_: True)

    try:
        handler.emit(record)
        (exported_log,) = exporter.get_finished_logs()
        encoded_request = encode_logs([exported_log])
    finally:
        provider.shutdown()

    (encoded_log,) = encoded_request.resource_logs[0].scope_logs[0].log_records
    encoded_attributes = {
        attribute.key: attribute.value.string_value for attribute in encoded_log.attributes
    }
    assert encoded_attributes[trace_types.ATTR_EXCEPTION_MESSAGE] == (
        telemetry_utils.REDACTED_EXCEPTION_MESSAGE
    )
    assert trace_types.ATTR_EXCEPTION_TRACE not in encoded_attributes
    assert encoded_log.dropped_attributes_count == 0
    assert "secret transcript" not in str(encoded_log)


# -- tool errors and websocket handshakes --

SECRET = "jane.doe@example.com"


@function_tool
async def lookup_orders(count: int) -> str:
    """Look up the customer's latest orders.

    Args:
        count: how many orders to return
    """
    return "ok"


@pytest.mark.parametrize(
    "arguments",
    [
        json.dumps({"count": SECRET}),  # valid JSON, wrong type
        '{"count": "' + SECRET,  # truncated JSON (repaired, then fails validation)
    ],
    ids=["wrong_type", "truncated_json"],
)
async def test_toolerror_warning_keeps_argument_values_out_of_log_body(
    arguments: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Argument detail stays under lk.pii.error, since a log body cannot be redacted."""
    llm_ = FakeLLM(
        fake_responses=[
            FakeLLMResponse(
                input="show my orders",
                content="",
                ttft=0.0,
                duration=0.0,
                tool_calls=[
                    llm.FunctionToolCall(
                        name="lookup_orders",
                        arguments=arguments,
                        call_id="call_1",
                    )
                ],
            ),
        ]
    )
    session = AgentSession(llm=llm_)

    with caplog.at_level(logging.WARNING, logger="livekit.agents"):
        await session.start(agent=Agent(instructions="test", tools=[lookup_orders]))
        await session.run(user_input="show my orders")
        await session.aclose()

    tool_warnings = [
        r
        for r in caplog.records
        if r.name.startswith("livekit.agents")
        and "ToolError while executing tool" in r.getMessage()
    ]
    assert tool_warnings, "expected a ToolError warning from the invalid tool call"
    for record in tool_warnings:
        assert SECRET not in record.getMessage()
        assert getattr(record, "lk.pii.error", None), "detail should be under lk.pii.error"
        assert SECRET in str(getattr(record, "lk.pii.error"))


SECRET_API_KEY = "secret-api-key-do-not-log"


def _handshake_error(host: str, header: str, api_key: str) -> WSServerHandshakeError:
    url = URL(f"wss://{host}/listen")
    headers = CIMultiDict({"Host": host, header: api_key})
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


def _session_raising(err: Exception) -> MagicMock:
    async def _raise(*_args, **_kwargs):
        raise err

    session = MagicMock()
    session.ws_connect = MagicMock(side_effect=_raise)
    return session


def _assert_redacted(err: APIStatusError) -> None:
    assert err.status_code == 401
    assert SECRET_API_KEY not in repr(err)
    assert SECRET_API_KEY not in str(err)
    assert err.__cause__ is None


async def test_deepgram_stt_redacts_api_key_from_handshake_error() -> None:
    """A rejected upgrade must not carry the request headers, which hold the key."""
    from livekit.plugins.deepgram import STT

    stt = STT(
        api_key=SECRET_API_KEY,
        http_session=_session_raising(
            _handshake_error("api.deepgram.com", "Authorization", f"Token {SECRET_API_KEY}")
        ),
    )
    stream = stt.stream()

    with pytest.raises(APIStatusError) as exc_info:
        await stream._connect_ws()

    await stream.aclose()
    _assert_redacted(exc_info.value)


async def test_xai_tts_redacts_api_key_from_handshake_error() -> None:
    from livekit.plugins.xai import TTS

    tts = TTS(api_key=SECRET_API_KEY)
    tts._session = _session_raising(
        _handshake_error("api.x.ai", "Authorization", f"Bearer {SECRET_API_KEY}")
    )

    with pytest.raises(APIStatusError) as exc_info:
        await tts._connect_ws(timeout=5.0, opts=tts._opts)

    _assert_redacted(exc_info.value)
