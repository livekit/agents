"""Summarize a benchmark run: the benchmark's scores next to the agent's own session logs.

    python -m lk_voice_bench.report benchmarks/voice/results/tau2/<run> [<run> ...]

For tau2 it reads ``results.json`` (and ``simulations/*.json``), for EVA ``records/*/metrics.json``;
for every run it reads the agent-side JSONL logs in ``agent_logs/`` written by
``BenchAgent.dump_log``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentStats:
    sessions: int = 0
    replies: int = 0
    interrupted_replies: int = 0
    e2e_latency: list[float] = field(default_factory=list)
    """User end of turn to agent audio, per reply that has it (seconds)."""
    llm_ttft: list[float] = field(default_factory=list)
    tts_ttfb: list[float] = field(default_factory=list)
    end_of_turn_delay: list[float] = field(default_factory=list)
    tool_calls: int = 0
    tool_errors: int = 0
    errors: Counter[str] = field(default_factory=Counter)


def read_agent_logs(run_dir: Path) -> AgentStats:
    stats = AgentStats()
    for path in sorted((run_dir / "agent_logs").glob("*.jsonl")):
        stats.sessions += 1
        for line in path.read_text().splitlines():
            ev = json.loads(line)
            kind = ev["kind"]
            if kind == "message":
                m = ev.get("metrics") or {}
                if ev["role"] == "assistant":
                    stats.replies += 1
                    stats.interrupted_replies += bool(ev.get("interrupted"))
                    if "e2e_latency" in m:
                        stats.e2e_latency.append(m["e2e_latency"])
                    if "llm_node_ttft" in m:
                        stats.llm_ttft.append(m["llm_node_ttft"])
                    if "tts_node_ttfb" in m:
                        stats.tts_ttfb.append(m["tts_node_ttfb"])
                elif ev["role"] == "user" and "end_of_turn_delay" in m:
                    stats.end_of_turn_delay.append(m["end_of_turn_delay"])
            elif kind == "tool_call":
                stats.tool_calls += 1
            elif kind == "tool_result" and ev.get("error"):
                stats.tool_errors += 1
            elif kind == "error":
                stats.errors[ev.get("source", "?")] += 1
    return stats


@dataclass
class Tau2Stats:
    simulations: int = 0
    by_task: dict[str, list[float]] = field(default_factory=dict)
    termination: Counter[str] = field(default_factory=Counter)
    infra_errors: int = 0
    hallucination_discarded: int = 0
    """Calls tau2 threw away and re-ran because its simulated caller hallucinated."""
    durations: list[float] = field(default_factory=list)


_INFRA = {"infrastructure_error", "agent_error", "user_error", "unexpected_error"}


def read_tau2(run_dir: Path) -> Tau2Stats | None:
    # a custom run keeps its results at the top; a released frame has one dir per cell (en-airline)
    cells = [d for d in [run_dir, *sorted(run_dir.iterdir())] if (d / "results.json").exists()]
    if not cells:
        return None
    stats = Tau2Stats()
    sims: list[dict[str, Any]] = []
    for cell in cells:
        data = json.loads((cell / "results.json").read_text())
        cell_sims = list(data.get("simulations") or [])
        if not cell_sims:
            cell_sims = [
                json.loads(p.read_text()) for p in sorted((cell / "simulations").glob("*.json"))
            ]
        sims += cell_sims
        stats.hallucination_discarded += len(
            list((cell / "hallucination_discarded").glob("*.json"))
        )
    for sim in sims:
        stats.simulations += 1
        reason = str(sim.get("termination_reason"))
        stats.termination[reason] += 1
        if reason in _INFRA:
            stats.infra_errors += 1
            continue
        reward = float(((sim.get("reward_info") or {}).get("reward")) or 0.0)
        stats.by_task.setdefault(str(sim["task_id"]), []).append(reward)
        stats.durations.append(float(sim.get("duration") or 0.0))
    return stats


def pass_hat_k(by_task: dict[str, list[float]], k: int) -> float | None:
    """tau-bench pass^k: chance that k i.i.d. trials of a task all succeed, averaged over tasks."""
    scores = []
    for rewards in by_task.values():
        n, c = len(rewards), sum(r >= 1.0 for r in rewards)
        if n < k:
            return None
        scores.append(math.comb(c, k) / math.comb(n, k))
    return sum(scores) / len(scores) if scores else None


# EVA's pass definitions (eva/metrics/aggregation.py). EVA-A also requires
# agent_speech_fidelity >= 0.95, which needs EVA's Gemini audio judge; it is left out when absent.
_EVA_A = {"task_completion": ("==", 1.0), "faithfulness": (">=", 0.5)}
_EVA_X = {
    "conversation_progression": (">=", 0.5),
    "turn_taking": (">=", 0.8),
    "conciseness": (">=", 0.5),
}
_EVA_MEANS = (
    "task_completion",
    "faithfulness",
    "agent_speech_fidelity",
    "conversation_progression",
    "turn_taking",
    "conciseness",
    "stt_wer",
    "transcription_accuracy_key_entities",
    "tool_call_validity",
)


def _passes(scores: dict[str, float], checks: dict[str, tuple[str, float]]) -> bool | None:
    ok = True
    for name, (op, threshold) in checks.items():
        v = scores.get(name)
        if v is None:
            return None
        ok &= math.isclose(v, threshold) if op == "==" else v >= threshold - 1e-9
    return ok


def read_eva(run_dir: Path) -> dict[str, Any] | None:
    records_dir = run_dir / "records"
    if not records_dir.is_dir():
        return None
    metric_files = [
        p
        for p in sorted(records_dir.glob("**/metrics.json"))
        if "_failed_attempt_" not in str(p.relative_to(records_dir))
    ]
    per_record: list[dict[str, float]] = []
    response_speed: list[float] = []
    for path in metric_files:
        metrics = json.loads(path.read_text()).get("metrics", {})
        scores: dict[str, float] = {}
        for name, m in metrics.items():
            if not isinstance(m, dict) or m.get("error") or m.get("skipped"):
                continue
            # EVA normalizes stt_wer to 1 - WER; keep the WER itself
            v = m.get("score") if name == "stt_wer" else m.get("normalized_score")
            if v is None:
                v = m.get("score")
            if v is not None:
                scores[name] = float(v)
        if "response_speed" in metrics and metrics["response_speed"].get("score") is not None:
            response_speed.append(float(metrics["response_speed"]["score"]))
        per_record.append(scores)

    def rate(checks: dict[str, tuple[str, float]]) -> float | None:
        verdicts = [v for v in (_passes(r, checks) for r in per_record) if v is not None]
        return sum(verdicts) / len(verdicts) if verdicts else None

    has_fidelity = any("agent_speech_fidelity" in r for r in per_record)
    a_checks = dict(_EVA_A)
    if has_fidelity:
        a_checks["agent_speech_fidelity"] = (">=", 0.95)
    out: dict[str, Any] = {
        "records": len(per_record),
        "EVA-A_pass" if has_fidelity else "EVA-A_pass (no speech fidelity)": rate(a_checks),
        "EVA-X_pass": rate(_EVA_X),
    }
    for name in _EVA_MEANS:
        vals = [r[name] for r in per_record if name in r]
        if vals:
            out[f"mean_{name}"] = sum(vals) / len(vals)
    out["response_speed_p50"] = _pct(response_speed, 0.5)
    return out


def _pct(xs: list[float], q: float) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    return f"{xs[min(len(xs) - 1, int(q * len(xs)))]:.2f}"


def summarize(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run_dir.name}
    if (eva := read_eva(run_dir)) is not None:
        out.update(eva)
    tau2 = read_tau2(run_dir)
    if tau2 is not None:
        trials = max((len(r) for r in tau2.by_task.values()), default=0)
        out.update(
            {
                "simulations": tau2.simulations,
                "tasks": len(tau2.by_task),
                "infra_errors": tau2.infra_errors,
                "hallucination_discarded": tau2.hallucination_discarded,
                "pass^1": pass_hat_k(tau2.by_task, 1),
                **{f"pass^{k}": pass_hat_k(tau2.by_task, k) for k in range(2, trials + 1)},
                "termination": dict(tau2.termination),
                "median_call_s": statistics.median(tau2.durations) if tau2.durations else None,
            }
        )
    agent = read_agent_logs(run_dir)
    out.update(
        {
            "agent_sessions": agent.sessions,
            "replies": agent.replies,
            "interrupted_reply_rate": (
                agent.interrupted_replies / agent.replies if agent.replies else None
            ),
            "e2e_latency_p50": _pct(agent.e2e_latency, 0.5),
            "e2e_latency_p90": _pct(agent.e2e_latency, 0.9),
            "end_of_turn_delay_p50": _pct(agent.end_of_turn_delay, 0.5),
            "llm_ttft_p50": _pct(agent.llm_ttft, 0.5),
            "tts_ttfb_p50": _pct(agent.tts_ttfb, 0.5),
            "tool_calls_per_session": (
                agent.tool_calls / agent.sessions if agent.sessions else None
            ),
            "tool_errors": agent.tool_errors,
            "session_errors": dict(agent.errors),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [summarize(r) for r in args.runs]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    width = max(len(k) for k in keys)
    for k in keys:
        cells = []
        for row in rows:
            v = row.get(k)
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"{k:<{width}}  " + "  ".join(f"{c:<24}" for c in cells))


if __name__ == "__main__":
    main()
