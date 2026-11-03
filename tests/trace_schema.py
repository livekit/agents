# GENERATED FILE -- DO NOT EDIT.
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`): the span rules come from the
# declarations in rust/crates/lk-agents-abi/src/telemetry/spans/, the checker from
# rust/ai_slop/lk-agents-codegen/src/telemetry/trace_schema.py.in.

"""The shape every livekit-agents trace must have, and a checker that applies it.

The nesting of spans is an emergent property of many call sites, and a refactor can move a
span under the wrong parent while every existing test still passes, because each test only
names the one edge it cares about. This module writes the rules down once:

* :data:`SPAN_PARENTS`: for each span name, the parents it may have (``ROOT`` for none).
* :data:`MAY_OUTLIVE_PARENT`: the few child/parent edges where the child is allowed to end
  after its parent, each with the reason. Everything else must sit inside its parent.
* :func:`check_trace`: applies those rules plus the per-turn invariants (one ``agent_turn``
  per speech, its own generation events matching ``lk.generation_count``) and returns the
  violations.

It reads spans from an in-memory exporter (the fake-session tests) or from an OTLP JSON export
downloaded from LiveKit Cloud, so the same rules check a unit test and a real run::

    uv run python -m tests.trace_schema path/to/traces.json

Both tables are generated from the span declarations in ``lk-agents-abi``. Moving a span without
updating its declaration fails every test that calls :func:`assert_trace_well_formed`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = None
"""Allowed parent meaning "no parent at all"."""

ANY = "*"
"""Allowed parent meaning "whatever was current": spans that follow their caller."""

_TOLERANCE_NS = 2_000_000  # 2 ms of slack for clocks read on either side of a span boundary

SPAN_PARENTS: dict[str, frozenset[str | None]] = {
    k: frozenset(v)
    for k, v in {
        "a2a_request": {ROOT},
        "agent_run": {"a2a_request", ANY},
        "agent_session": {"job_entrypoint", ANY},
        "agent_speaking": {"agent_turn"},
        "agent_turn": {"agent_session"},
        "amd": {"agent_session"},
        "drain_agent_activity": {"session_close", "update_agent"},
        "eou_detection": {"eou_wait"},
        "eou_wait": {"user_turn"},
        "event_loop_blocked": {ANY},
        "function_tool": {"agent_run", "agent_turn"},
        "job_entrypoint": {ROOT},
        "job_shutdown": {"job_entrypoint"},
        "judge_evaluation": {ANY},
        "keyterm_detection": {"agent_session", "agent_turn"},
        "llm_fallback_adapter": {ANY},
        "llm_node": {"agent_turn"},
        "llm_request": {ANY},
        "llm_request_run": {"llm_request"},
        "on_enter": {"start_agent_activity"},
        "on_exit": {"drain_agent_activity"},
        "on_session_end": {"job_shutdown"},
        "on_user_turn_completed": {"user_turn"},
        "pause_agent_activity": {"update_agent"},
        "publish_audio_output": {"session_start"},
        "realtime_inference": {"agent_turn"},
        "realtime_metrics": {"realtime_inference"},
        "resume_agent_activity": {"update_agent"},
        "room_connect": {"job_entrypoint", "session_start"},
        "room_disconnect": {"job_shutdown"},
        "rpc_call": {ANY},
        "rpc_handler": {"agent_session"},
        "session_close": {"agent_session"},
        "session_end_upload": {"job_shutdown"},
        "session_start": {"agent_session"},
        "setup_toolsets": {"start_agent_activity"},
        "shutdown_callback": {"job_shutdown"},
        "start_agent_activity": {"session_start", "update_agent"},
        "tts_fallback_adapter": {ANY},
        "tts_node": {"agent_turn"},
        "tts_request": {ANY},
        "tts_request_run": {"tts_request"},
        "tts_stream_adapter": {ANY},
        "update_agent": {"agent_session"},
        "user_speaking": {"user_turn"},
        "user_turn": {"agent_session"},
        "wait_for_audio_track": {"agent_session", "session_start"},
        "wait_for_participant": {"job_entrypoint", "session_start"},
    }.items()
}

MAY_OUTLIVE_PARENT: dict[tuple[str, str], str] = {
    (
        "keyterm_detection",
        "agent_turn",
    ): "the pass runs alongside the reply and can outlast a short or interrupted turn",
    (
        "keyterm_detection",
        "agent_session",
    ): "the pass runs alongside the reply and can outlast a short or interrupted turn",
    (
        "llm_request",
        ANY,
    ): "a recovery probe (lk.fallback.recovery) runs in the background past the request",
    ("on_enter", "start_agent_activity"): "on_enter runs as a task the activity does not wait for",
    (
        "publish_audio_output",
        "session_start",
    ): "session.start() returns before the track is published",
    ("realtime_metrics", "realtime_inference"): "the metrics arrive after the inference ended",
    (
        "tts_request",
        ANY,
    ): "a recovery probe (lk.fallback.recovery) runs in the background past the request",
    (
        "tts_stream_adapter",
        ANY,
    ): "a recovery probe (lk.fallback.recovery) runs in the background past the request",
    ("wait_for_audio_track", "session_start"): "session.start() returns before the first frame",
    ("wait_for_audio_track", "agent_session"): "session.start() returns before the first frame",
    (
        "wait_for_participant",
        "session_start",
    ): "session.start() returns before a participant is linked",
    (
        "wait_for_participant",
        "job_entrypoint",
    ): "session.start() returns before a participant is linked",
}
"""Child/parent edges where the child may end after its parent, with the reason. Deliberate:
each is a known property of the code, and a viewer draws them poking out of the parent."""


@dataclass
class EventRecord:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanRecord:
    """The part of a span the rules look at, from either source."""

    name: str
    span_id: str
    parent_id: str | None
    trace_id: str
    start_ns: int
    end_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[EventRecord] = field(default_factory=list)


def from_readable_spans(spans: Iterable[Any]) -> list[SpanRecord]:
    """Records from OpenTelemetry SDK ``ReadableSpan`` objects (an in-memory exporter)."""
    out: list[SpanRecord] = []
    for s in spans:
        ctx = s.context
        out.append(
            SpanRecord(
                name=s.name,
                span_id=format(ctx.span_id, "016x"),
                parent_id=format(s.parent.span_id, "016x") if s.parent is not None else None,
                trace_id=format(ctx.trace_id, "032x"),
                start_ns=int(s.start_time or 0),
                end_ns=int(s.end_time or 0),
                attributes=dict(s.attributes or {}),
                events=[EventRecord(e.name, dict(e.attributes or {})) for e in s.events],
            )
        )
    return out


def _otlp_value(value: Mapping[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    return value


def _otlp_attributes(item: Mapping[str, Any]) -> dict[str, Any]:
    return {a["key"]: _otlp_value(a["value"]) for a in item.get("attributes", [])}


def from_otlp_json(document: Mapping[str, Any] | str | Path) -> list[SpanRecord]:
    """Records from an OTLP/JSON export (``resourceSpans`` → ``scopeSpans`` → ``spans``)."""
    if not isinstance(document, Mapping):
        document = json.loads(Path(document).read_text(encoding="utf-8", errors="replace"))
    out: list[SpanRecord] = []
    for rs in document.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                out.append(
                    SpanRecord(
                        name=s["name"],
                        span_id=s["spanId"],
                        parent_id=s.get("parentSpanId") or None,
                        trace_id=s["traceId"],
                        start_ns=int(s["startTimeUnixNano"]),
                        end_ns=int(s["endTimeUnixNano"]),
                        attributes=_otlp_attributes(s),
                        events=[
                            EventRecord(e["name"], _otlp_attributes(e)) for e in s.get("events", [])
                        ],
                    )
                )
    return out


def check_trace(
    spans: Sequence[SpanRecord],
    *,
    tolerance_ns: int = _TOLERANCE_NS,
    allow_missing_parents: bool = False,
) -> list[str]:
    """Every way the spans break the rules, as one line each; empty when the trace is sound.

    ``allow_missing_parents`` is for partial exports (a view keyed to one span drops the
    ancestors): a span whose parent is absent is then checked as if it were a root."""
    violations: list[str] = []
    by_id = {s.span_id: s for s in spans}

    if len({s.trace_id for s in spans}) > 1:
        violations.append(
            f"spans belong to {len({s.trace_id for s in spans})} traces, expected one"
        )

    for s in spans:
        if s.end_ns < s.start_ns:
            # the duration is unsigned on the wire: a backend that subtracts reads 2^64 minus this
            violations.append(
                f"{s.name}: ends {(s.start_ns - s.end_ns) / 1e6:.1f} ms before it starts"
            )
        allowed = SPAN_PARENTS.get(s.name)
        if allowed is None:
            violations.append(
                f"{s.name}: unknown span, declare it in lk-agents-abi telemetry::spans"
            )
            continue
        parent = by_id.get(s.parent_id) if s.parent_id else None
        if s.parent_id and parent is None:
            if not allow_missing_parents:
                violations.append(f"{s.name}: parent {s.parent_id} is not in the trace")
            continue  # a partial export: nothing to check the edge against
        parent_name = parent.name if parent is not None else ROOT
        if ANY not in allowed and parent_name not in allowed:
            shown = "no parent" if parent_name is ROOT else parent_name
            violations.append(
                f"{s.name}: parent is {shown}, allowed: "
                + ", ".join(sorted("ROOT" if p is ROOT else p for p in allowed))
            )
        if parent is not None:
            if s.start_ns + tolerance_ns < parent.start_ns:
                violations.append(
                    f"{s.name}: starts {(parent.start_ns - s.start_ns) / 1e6:.1f} ms before its "
                    f"parent {parent.name}"
                )
            overrun = s.end_ns - parent.end_ns
            if overrun > tolerance_ns and (
                (s.name, parent.name) not in MAY_OUTLIVE_PARENT
                and (s.name, ANY) not in MAY_OUTLIVE_PARENT
            ):
                violations.append(
                    f"{s.name}: ends {overrun / 1e6:.1f} ms after its parent {parent.name}"
                )

    # one agent_turn per speech handle, and its generations accounted for
    speech_turns: dict[str, list[SpanRecord]] = defaultdict(list)
    for s in spans:
        if s.name != "agent_turn":
            continue
        speech_id = str(s.attributes.get("lk.speech_id") or "")
        if not speech_id:
            violations.append("agent_turn: no lk.speech_id")
            continue
        speech_turns[speech_id].append(s)
        # a preemptive attempt discarded for this speech left its generations on the span too;
        # the count is the finishing speech's own steps
        own_generations = sum(
            1
            for e in s.events
            if e.name == "generation"
            and str(e.attributes.get("lk.generation_id", f"{speech_id}_")).startswith(
                f"{speech_id}_"
            )
        )
        count = s.attributes.get("lk.generation_count")
        try:
            count_ok = count is not None and int(count) == own_generations
        except (TypeError, ValueError):
            count_ok = False
        if not count_ok:
            violations.append(
                f"agent_turn {speech_id}: lk.generation_count={count!r} but "
                f"{own_generations} of its own generation events"
            )
    for speech_id, turns in speech_turns.items():
        if len(turns) > 1:
            violations.append(
                f"agent_turn: speech {speech_id} has {len(turns)} turns, expected one"
            )

    # eou_detection never runs outside a wait; a wait always ends with an outcome
    for s in spans:
        if s.name == "eou_wait" and "lk.eou.outcome" not in s.attributes:
            violations.append("eou_wait: no lk.eou.outcome")

    return violations


def assert_trace_well_formed(spans: Iterable[Any], **kwargs: Any) -> None:
    """For tests: ``spans`` are ``ReadableSpan`` objects from an in-memory exporter."""
    violations = check_trace(from_readable_spans(spans), **kwargs)
    assert not violations, "trace shape violations:\n  " + "\n  ".join(violations)


def _summary(spans: Sequence[SpanRecord]) -> str:
    counts = Counter(s.name for s in spans)
    return ", ".join(f"{n}×{c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1]))


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tests.trace_schema <traces.json>", file=sys.stderr)
        return 2
    records = from_otlp_json(argv[1])
    print(f"{len(records)} spans: {_summary(records)}")
    violations = check_trace(records, allow_missing_parents=True)
    if not violations:
        print("trace shape OK")
        return 0
    print(f"{len(violations)} violation(s):")
    for v in violations:
        print("  -", v)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
