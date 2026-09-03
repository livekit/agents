"""Guard test: every telemetry attribute key must be classified for PII redaction.

``telemetry.pii`` strips attributes whose key carries a dot-delimited ``pii``
segment (e.g. ``lk.pii.chat_ctx``) before any exporter that is not LiveKit
Cloud's, whose own handling is the project's setting in the dashboard.

The OpenTelemetry GenAI attributes cannot carry the ``pii`` marker: their names
are fixed by the semantic convention. Those are enumerated in
``telemetry.pii.GEN_AI_PII_ATTRIBUTES`` instead.

Every constant in ``telemetry/trace_types.py`` must be explicitly accounted for:
its value carries the ``pii`` segment, or it is registered in the GenAI PII set,
or it is listed here as safe (no conversational content, tool payloads, or other
user data).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from livekit.agents.telemetry import pii, trace_types

pytestmark = pytest.mark.unit

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
        "lk.start_time",
        "lk.end_time",
        "lk.retry_count",
        "lk.provider_request_ids",
        "lk.participant_id",
        "lk.participant_kind",
        "lk.job_id",
        "lk.agent_name",
        "lk.cloud_agent_id",
        "lk.deployment_id",
        "lk.session_options",
        "lk.generation_id",
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
        # eou detection (numeric / enum only)
        "lk.eou.probability",
        "lk.eou.unlikely_threshold",
        "lk.eou.endpointing_delay",
        "lk.eou.language",
        "lk.transcript_confidence",
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
        # GenAI event and metric names (not attribute keys)
        "gen_ai.client.inference.operation.details",
        "gen_ai.client.operation.duration",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.token.usage",
        "gen_ai.execute_tool.duration",
        "gen_ai.invoke_agent.duration",
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

    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text())
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
