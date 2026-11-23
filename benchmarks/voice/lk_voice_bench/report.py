"""Summarize a benchmark run: the benchmark's scores next to the agent's own session logs.

    python -m lk_voice_bench.report benchmarks/voice/results/tau2/<run> [<run> ...]

For tau2 it reads ``results.json`` (and ``simulations/*.json``); for every run it reads the
agent-side JSONL logs in ``agent_logs/`` written by ``BenchAgent.dump_log``.
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
    durations: list[float] = field(default_factory=list)
    domains: Counter[str] = field(default_factory=Counter)


_INFRA = {"infrastructure_error", "agent_error", "user_error", "unexpected_error"}


def read_tau2(run_dir: Path) -> Tau2Stats | None:
    results = run_dir / "results.json"
    if not results.exists():
        return None
    data = json.loads(results.read_text())
    sims: list[dict[str, Any]] = list(data.get("simulations") or [])
    if not sims:
        sims = [json.loads(p.read_text()) for p in sorted((run_dir / "simulations").glob("*.json"))]
    domain = (data.get("info") or {}).get("environment_info", {}).get("domain_name")
    stats = Tau2Stats()
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
        stats.domains[domain or "?"] += 1
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


def _pct(xs: list[float], q: float) -> str:
    if not xs:
        return "-"
    xs = sorted(xs)
    return f"{xs[min(len(xs) - 1, int(q * len(xs)))]:.2f}"


def summarize(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run_dir.name}
    tau2 = read_tau2(run_dir)
    if tau2 is not None:
        trials = max((len(r) for r in tau2.by_task.values()), default=0)
        out.update(
            {
                "simulations": tau2.simulations,
                "tasks": len(tau2.by_task),
                "infra_errors": tau2.infra_errors,
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
