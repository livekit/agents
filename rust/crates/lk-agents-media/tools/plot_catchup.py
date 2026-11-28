"""Plot recorded Catchup timing; keep the generated PNG outside Git.

Run with matplotlib installed, e.g. uv run --with matplotlib tools/plot_catchup.py.
The untreated FIFO is replayed from the recorded arrivals on the same render clock.
No NetEq per-frame timing is inferred from its aggregate counters.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RATE = 24_000
BLUE = "#2563eb"
ORANGE = "#e87816"
GRAY = "#64748b"


def read_rows(path: Path) -> list[dict[str, int]]:
    with path.open() as stream:
        return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="comparison output directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input / "report.json").read_text())
    case = report["scenarios"]["catchup"]
    rows = read_rows(args.input / "catchup" / "timeline.csv")
    arrivals = read_rows(args.input / "catchup" / "arrivals.csv")
    times_us = np.array([row["time_us"] for row in rows])
    step = int(times_us[1] - times_us[0])
    if times_us[0] != 0 or not np.all(np.diff(times_us) == step) or step * RATE % 1_000_000:
        raise ValueError("expected a regular render trace with integral sample counts")
    block = step * RATE // 1_000_000
    consumed = 0
    received = 0
    index = 0
    untreated = []
    for now in times_us:
        while index < len(arrivals) and arrivals[index]["arrival_us"] <= now:
            arrival = arrivals[index]
            if arrival["source_start"] != received:
                raise ValueError("arrival schedule must preserve complete ordered source")
            received += arrival["samples"]
            index += 1
        consumed += min(block, received - consumed)
        untreated.append(received - consumed)
    if rows[-1]["source_consumed"] != received or consumed != received:
        raise ValueError("trace must drain all source audio")
    total = np.array([row["accelerated"] for row in rows])
    fast = np.array([row["fast_accelerated"] for row in rows])
    if np.any(np.diff(total) < 0) or np.any(np.diff(fast) < 0) or np.any(fast > total):
        raise ValueError("invalid cumulative acceleration counters")

    # Stop at the final delivery, before the EOF queue-drain/observation tail. Both paths
    # retain their real startup delay. A common 100ms mean reveals the queue trend while
    # keeping the per-chunk sawtooth from obscuring the recovered delay.
    end = arrivals[-1]["arrival_us"] / 1_000_000
    visible = times_us <= arrivals[-1]["arrival_us"]
    time = times_us[visible] / 1_000_000
    group_size = max(1, 100_000 // step)

    def mean(values: np.ndarray) -> np.ndarray:
        starts = np.arange(0, len(values), group_size)
        counts = np.minimum(group_size, len(values) - starts)
        return np.add.reduceat(values, starts) / counts

    rust_queue = np.array([row["queued_source"] + row["pending_output"] for row in rows])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharex=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.13, top=0.86, hspace=0.4, wspace=0.24)
    fig.suptitle(
        "Playout · backlog recovery and learned startup",
        fontsize=21,
        fontweight="bold",
        x=0.075,
        ha="left",
        y=0.97,
    )
    fig.text(
        0.075,
        0.915,
        "Catchup uses identical arrivals. The answer replay compares retaining timing with resetting it.",
        fontsize=11,
        color=GRAY,
    )

    ax = axes[0, 0]
    native = read_rows(args.input / "catchup" / "neteq_timeline.csv")
    if [row["time_us"] for row in native] != list(times_us):
        raise ValueError("native and Rust traces must use the same render clock")
    native_queue = np.array([row["queued_ms"] for row in native])
    ax.plot(mean(time), mean(native_queue[visible]), color="#0f766e", label="WebRTC", linewidth=2)
    ax.plot(
        mean(time),
        mean(np.array(untreated)[visible] / 24),
        color=GRAY,
        label="Untreated FIFO",
        linewidth=2,
    )
    ax.plot(mean(time), mean(rust_queue[visible] / 24), color=BLUE, label="Rust", linewidth=2.3)
    ax.set(title="Queued audio · 100 ms mean", ylabel="Queued + pending audio (ms)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")

    ax = axes[0, 1]
    normal_ms = (total[visible] - fast[visible]) / 24
    fast_ms = fast[visible] / 24
    ax.stackplot(
        time, normal_ms, fast_ms, colors=[BLUE, ORANGE], alpha=0.8, labels=["Normal", "Fast"]
    )
    ax.set(title="Cumulative delay removed", ylabel="Audio removed by time scaling (ms)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
    removed = total[-1] / 24
    fast_removed = fast[-1] / 24
    ax.annotate(
        f"{removed:.0f} ms total",
        xy=(time[-1], (total[visible] / 24)[-1]),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        color=BLUE,
        fontweight="bold",
    )

    ax = axes[1, 0]
    delta = np.diff(total, prepend=0)[visible] / 24
    fast_delta = np.diff(fast, prepend=0)[visible]
    for mask, color, label in [(delta > 0, BLUE, "Normal"), (fast_delta > 0, ORANGE, "Fast")]:
        if label == "Normal":
            mask &= fast_delta == 0
        ax.vlines(time[mask], 0, delta[mask], color=color, alpha=0.7, linewidth=1.5, label=label)
    ax.set(
        title="When acceleration actually runs",
        ylabel="Removed per operation (ms)",
        xlabel="Elapsed playback time (s)",
    )
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="none")
    first = case["acceleration"]
    for key, label, color, y in [
        ("first_ms", "Normal", BLUE, 0.91),
        ("first_fast_ms", "Fast", ORANGE, 0.77),
    ]:
        value = first[key]
        if value is not None:
            when = value / 1000
            ax.text(
                0.48,
                y,
                f"{label} starts @ {when:.2f}s",
                transform=ax.transAxes,
                color=color,
                fontsize=10,
            )
            ax.axvline(when, color=color, alpha=0.35, linestyle=":")

    ax = axes[1, 1]
    with (args.input / "turns.csv").open() as stream:
        turns = list(csv.DictReader(stream))
    for variant, offset, color, label in [
        ("retained", -0.18, BLUE, "Retain timing: 20 → 80 ms"),
        ("reset", 0.18, GRAY, "Reset: 20 ms each answer"),
    ]:
        selected = [row for row in turns if row["variant"] == variant]
        positions = np.array([int(row["turn"]) for row in selected])
        ax.bar(
            positions + offset,
            [int(row["underruns"]) for row in selected],
            width=0.34,
            color=color,
            label=label,
        )
    ax.set(
        title="Later answers · 60 ms late second chunk",
        ylabel="Underruns per answer",
        xlabel="Answer",
    )
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1])
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        edgecolor="none",
        fontsize=9,
    )

    onset = case["fault_start_ms"] / 1000
    for ax in list(axes.flat)[:3]:
        ax.set_xlim(0, end)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        ax.axvline(onset, color=GRAY, linestyle="--", linewidth=1, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#cbd5e1")
    axes[1, 1].set_xlim(0.4, 10.6)
    axes[1, 1].set_xticks(range(1, 11))
    fig.text(
        0.075,
        0.065,
        f"{removed:.0f} ms removed ({fast_removed:.0f} ms fast)  ·  {case['rust_metrics']['underruns']} Rust underruns  ·  dashed line: schedule starts @ {onset:g}s",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.033,
        "First 3 panels: Catchup traces, including startup recovery. Last: repeated-answer replay with a 20ms cold prior. No timing is realigned.",
        fontsize=8.5,
        color=GRAY,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=180,
        facecolor="white",
        metadata={
            "Title": "Catchup timing",
            "Description": f"Source SHA-256: {report['source_sha256']}; recorded Rust trace and replayed untreated FIFO.",
        },
    )
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
