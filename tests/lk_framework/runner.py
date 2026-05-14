"""Runner sketch: structure and signatures, not a full implementation.

Shows how a compiled JSON test spec is consumed end-to-end. The real
implementation reuses existing tests/fake_session.py machinery; this file
documents the algorithm and the integration points.

Pipeline:
    JSON spec  ──▶  build session  ──▶  walk scenario  ──▶  trace.jsonl
                                                                │
                                                                ▼
                                                          assertion engine
                                                                │
                                                                ▼
                                                              result
"""

from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Reuse what already exists.
from tests.fake_llm import FakeLLM, FakeLLMResponse
from tests.fake_stt import FakeSTT
from tests.fake_tts import FakeTTS
from tests.fake_vad import FakeVAD
from tests.fake_io import FakeAudioInput, FakeAudioOutput
from livekit.agents import AgentSession


# ── Trace types ──────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    t_ms: int                       # virtual time, monotonic
    ev: str                         # event kind, e.g. "eou.fired"
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def find(self, ev: str) -> list[TraceEvent]:
        return [e for e in self.events if e.ev == ev]

    def first(self, ev: str) -> TraceEvent | None:
        return next((e for e in self.events if e.ev == ev), None)

    def write_jsonl(self, path: Path) -> None:
        with path.open("w") as f:
            for e in self.events:
                f.write(json.dumps({"t": e.t_ms, "ev": e.ev, **e.fields}) + "\n")


# ── Recorder: subscribes to AgentSession events + audio output ───────────

class TraceRecorder:
    """Hooks into the session's event bus and audio output sink.

    Maps existing AgentSession events (voice/events.py) to the unified trace
    vocabulary, and adds pipeline-milestone events the framework needs
    (audio.first_sample, audio.played, llm.cancelled).
    """

    def __init__(self, session: AgentSession, clock: "Clock") -> None:
        self._session = session
        self._clock = clock
        self.trace = Trace()

        # Subscribe to every event the SDK already emits.
        session.on("user_state_changed",     self._on_user_state)
        session.on("agent_state_changed",    self._on_agent_state)
        session.on("user_input_transcribed", self._on_transcribed)
        session.on("conversation_item_added", self._on_item_added)
        session.on("function_tools_executed", self._on_tools)
        session.on("metrics_collected",      self._on_metrics)
        session.on("overlapping_speech",     self._on_overlap)
        session.on("agent_false_interruption", self._on_false_interruption)
        session.on("error",                  self._on_error)

    def _emit(self, ev: str, **fields: Any) -> None:
        self.trace.events.append(TraceEvent(t_ms=self._clock.now_ms(), ev=ev, fields=fields))

    # Each handler translates an SDK event to one or more trace events.
    def _on_user_state(self, e):     self._emit("user.state_changed", to=e.new_state)
    def _on_agent_state(self, e):    self._emit("agent.state_changed", to=e.new_state)
    def _on_transcribed(self, e):    self._emit("stt.final" if e.is_final else "stt.interim", text=e.transcript)
    def _on_item_added(self, e):     self._emit("message.committed", role=e.item.role, ...)
    def _on_tools(self, e):          self._emit("tools.executed", calls=[...])
    def _on_metrics(self, e):        self._emit_metric(e.metrics)   # EOU, LLM, TTS
    def _on_overlap(self, e):        self._emit("overlap.detected", is_interruption=e.is_interruption)
    def _on_false_interruption(self, e): self._emit("agent.false_interruption", resumed=e.resumed)
    def _on_error(self, e):          self._emit("error", source=type(e.source).__name__, message=str(e.error))


# ── Scenario walker: executes user inputs against an anchor resolver ─────

class AnchorResolver:
    """Resolves scenario anchors like 'eou + 0ms' or '+1500ms' to wall times.

    Some anchors are immediate (t=N, +Nms). Others wait on pipeline
    milestones (eou.fired, agent.speaking) — the walker pauses until that
    event appears in the trace, then schedules the input.
    """

    def __init__(self, trace: Trace, clock: "Clock") -> None:
        self.trace = trace
        self.clock = clock

    async def resolve(self, anchor: str) -> int:
        ...  # parse, await milestone if needed, return target t_ms


class ScenarioWalker:
    def __init__(self, spec: dict, session: AgentSession, clock: "Clock", recorder: TraceRecorder) -> None:
        self.spec = spec
        self.session = session
        self.clock = clock
        self.recorder = recorder

    async def run(self) -> None:
        scenario = self.spec["scenario"]
        steps = scenario["steps"] if scenario["type"] == "user_timeline" else [scenario]

        resolver = AnchorResolver(self.recorder.trace, self.clock)
        for step in steps:
            if step["type"] == "user_says":
                target_t_ms = await resolver.resolve(step.get("at", "t=0"))
                await self.clock.advance_to(target_t_ms)
                await self._push_speech(step["text"], step["duration_ms"])

    async def _push_speech(self, text: str, duration_ms: int) -> None:
        # Push silence frames for `duration_ms`, while FakeSTT is scripted to
        # emit `text` as the final transcript. FakeVAD synthesizes
        # START/END_OF_SPEECH from the audio window.
        audio_input: FakeAudioInput = self.session.input.audio
        audio_input.push(duration_ms / 1000.0)


# ── Assertion engine: pure functions over the trace ──────────────────────

class AssertionEngine:
    def __init__(self, spec: dict, trace: Trace) -> None:
        self.spec = spec
        self.trace = trace

    def run(self) -> "Result":
        result = Result()

        # 1) expect — trace queries with path syntax (messages[0], llm_calls[0], etc.)
        for path, expected in self.spec.get("expect", {}).items():
            actual = self._resolve_path(path)
            for field_name, matcher in expected.items():
                result.check(f"{path}.{field_name}", self._match(actual.get(field_name), matcher))

        # 2) budgets — named timing deltas computed from the trace
        for budget_name, matcher in self.spec.get("budgets", {}).items():
            actual_ms = self._compute_budget(budget_name)
            result.check(f"budget.{budget_name}", self._match(actual_ms, matcher))

        # 3) invariants — temporal-logic style queries
        for inv in self.spec.get("invariants", []):
            result.check(f"invariant.{inv['id']}", INVARIANTS[inv["id"]](self.trace))

        return result

    def _resolve_path(self, path: str) -> dict:
        # messages[0]      → trace.find("message.committed")[0]
        # messages[-1]     → ...[-1]
        # tool_calls.name  → filter trace for that tool by name
        ...

    def _compute_budget(self, name: str) -> int:
        # eou_delay           = eou.fired.t - vad.end_of_speech.t
        # llm_ttft            = llm.first_token.t - llm.requested.t
        # llm_cancel_latency  = llm.cancelled.t - llm.cancel_requested.t
        # e2e                 = audio.first_sample.t - user.speech_end.t
        ...

    def _match(self, actual: Any, matcher: dict) -> bool:
        kind = matcher["kind"]
        if kind == "eq":       return actual == matcher["value"]
        if kind == "lt":       return actual < matcher["value"]
        if kind == "gt":       return actual > matcher["value"]
        if kind == "within":   lo, hi = matcher["range"]; return lo <= actual <= hi
        if kind == "contains": return matcher["value"] in (actual or "")
        ...


# ── Invariants: trace → bool ─────────────────────────────────────────────

def _no_audio_while_user_speaking(trace: Trace) -> bool:
    """For every t, if user.state == speaking, no audio.* events fire."""
    user_windows = []  # [(start_t, end_t)] when user is speaking
    audio_events = trace.find("audio.first_sample") + trace.find("audio.played")
    ...
    return True


INVARIANTS: dict[str, callable] = {
    "no_audio_while_user_speaking":  _no_audio_while_user_speaking,
    "no_errors":                     lambda tr: len(tr.find("error")) == 0,
    "no_false_interruption":         lambda tr: len(tr.find("agent.false_interruption")) == 0,
    # ...
}


# ── Top-level driver ────────────────────────────────────────────────────

@dataclass
class Result:
    passed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name if ok else (name, detail))


async def run_spec(spec_path: Path) -> Result:
    spec = json.loads(spec_path.read_text())
    AgentCls = _resolve_agent(spec["agent"])
    clock = _make_clock(spec["clock"])
    session = _build_session(spec, AgentCls, clock)
    recorder = TraceRecorder(session, clock)

    walker = ScenarioWalker(spec, session, clock, recorder)
    await walker.run()
    await session.aclose()

    trace_path = spec_path.with_suffix(".trace.jsonl")
    recorder.trace.write_jsonl(trace_path)

    return AssertionEngine(spec, recorder.trace).run()


# ── Helpers (implementations omitted for the sketch) ─────────────────────

class Clock:                              # virtual or real
    def now_ms(self) -> int: ...
    async def advance_to(self, t_ms: int) -> None: ...

def _resolve_agent(spec: str): ...         # importlib.import_module(...).Class
def _make_clock(mode: str) -> Clock: ...   # 'virtual' | 'real'
def _build_session(spec, AgentCls, clock) -> AgentSession:
    # Wires FakeLLM/STT/TTS/VAD per spec.fakes, returns a configured AgentSession.
    # ~50 lines. Mirrors tests/fake_session.create_session().
    ...
