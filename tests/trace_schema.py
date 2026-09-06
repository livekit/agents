"""The shape every livekit-agents trace must have, and a checker that applies it.

The nesting of spans is an emergent property of many call sites, and a refactor can move a
span under the wrong parent while every existing test still passes, because each test only
names the one edge it cares about. This module writes the rules down once:

* :data:`SPAN_PARENTS`: for each span name, the parents it may have (``ROOT`` for none).
* :data:`MAY_OUTLIVE_PARENT`: the few child/parent edges where the child is allowed to end
  after its parent, each with the reason. Everything else must sit inside its parent.
* :func:`check_trace`: applies those rules plus the per-turn invariants (one ``agent_turn``
  per speech, generation events matching ``lk.generation_count``) and returns the violations.

It reads spans from an in-memory exporter (the fake-session tests) or from an OTLP JSON export
downloaded from LiveKit Cloud, so the same rules check a unit test and a real run::

    uv run python -m tests.trace_schema path/to/traces.json

Adding a span means adding a row here. Moving one without updating the row fails every test
that calls :func:`assert_trace_well_formed`.
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
        # -- the job (livekit.agents.ipc.job_proc_lazy_main)
        "job_entrypoint": {ROOT},
        "job_shutdown": {"job_entrypoint"},
        "on_session_end": {"job_shutdown"},
        "session_end_upload": {"job_shutdown"},
        "room_disconnect": {"job_shutdown"},
        "shutdown_callback": {"job_shutdown"},
        # -- the session (voice.agent_session); ROOT outside a job (tests, integrators)
        "agent_session": {"job_entrypoint", ROOT},
        "session_start": {"agent_session"},
        "session_close": {"agent_session"},
        "update_agent": {"agent_session"},
        # -- startup work: under session_start while the session starts, under the job before
        "room_connect": {"session_start", "job_entrypoint"},
        "wait_for_participant": {"session_start", "job_entrypoint"},
        "wait_for_audio_track": {"session_start", "agent_session"},
        "wait_for_video_track": {"session_start", "agent_session"},
        "publish_audio_output": {"session_start"},
        # -- agent activity lifecycle
        "start_agent_activity": {"session_start", "update_agent"},
        "setup_toolsets": {"start_agent_activity"},
        "on_enter": {"start_agent_activity"},
        "pause_agent_activity": {"update_agent"},
        "resume_agent_activity": {"update_agent"},
        "drain_agent_activity": {"update_agent", "session_close", "agent_session"},
        "on_exit": {"drain_agent_activity"},
        # -- the user's turn
        "user_turn": {"agent_session"},
        "user_speaking": {"user_turn", "agent_session"},
        "eou_wait": {"user_turn"},
        "eou_detection": {"eou_wait"},
        "on_user_turn_completed": {"user_turn", "agent_session"},
        # -- the agent's turn: one per speech handle, every generation inside it
        "agent_turn": {"agent_session"},
        "llm_node": {"agent_turn"},
        "tts_node": {"agent_turn"},
        "function_tool": {"agent_turn"},
        "agent_speaking": {"agent_turn"},
        "realtime_inference": {"agent_turn"},
        "realtime_metrics": {"realtime_inference", "agent_turn"},
        # -- model requests: under the node that made them, or the feature that owns them
        "llm_request": {
            "llm_node",
            "llm_fallback_adapter",
            "keyterm_detection",
            "judge_evaluation",
            "amd",
        },
        "llm_fallback_adapter": {"llm_node", "keyterm_detection"},
        "llm_request_run": {"llm_request", "llm_fallback_adapter"},
        "tts_request": {"tts_node", "tts_fallback_adapter"},
        "tts_fallback_adapter": {"tts_node"},
        "tts_request_run": {"tts_request", "tts_fallback_adapter"},
        # -- session-scoped features
        "keyterm_detection": {"agent_session"},
        "amd": {"agent_session"},
        "judge_evaluation": {"agent_session", ROOT},
        # -- RPC: handlers are session events, calls follow their caller
        "rpc_handler": {"agent_session", "job_entrypoint"},
        "rpc_call": {ANY},
        # -- a stall lands under whatever was blocked (any span), or the session/job
        "event_loop_blocked": {ANY},
    }.items()
}

MAY_OUTLIVE_PARENT: dict[tuple[str, str], str] = {
    (
        "wait_for_participant",
        "session_start",
    ): "session.start() returns before a participant is linked",
    ("wait_for_audio_track", "session_start"): "session.start() returns before the first frame",
    (
        "publish_audio_output",
        "session_start",
    ): "session.start() returns before the track is published",
    ("on_enter", "start_agent_activity"): "on_enter runs as a task the activity does not wait for",
    (
        "event_loop_blocked",
        ANY,
    ): "the heartbeat notices a stall one tick after the blocked call returned",
}
"""Child/parent edges where the child may end after its parent, with the reason. Deliberate:
each is a known property of the code, and a viewer draws them poking out of the parent."""


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
    events: list[str] = field(default_factory=list)


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
                events=[e.name for e in s.events],
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
                        attributes={
                            a["key"]: _otlp_value(a["value"]) for a in s.get("attributes", [])
                        },
                        events=[e["name"] for e in s.get("events", [])],
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
        allowed = SPAN_PARENTS.get(s.name)
        if allowed is None:
            violations.append(f"{s.name}: unknown span, add it to tests.trace_schema.SPAN_PARENTS")
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
        if s.name == "agent_turn":
            speech_id = s.attributes.get("lk.speech_id")
            if speech_id:
                speech_turns[str(speech_id)].append(s)
            generations = s.events.count("generation")
            count = s.attributes.get("lk.generation_count")
            if count is not None and int(count) != generations:
                violations.append(
                    f"agent_turn {speech_id}: lk.generation_count={count} but "
                    f"{generations} generation events"
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
