"""Compare actual pre-PR SDK builds with the prepared-publication builds."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from analyze import analyze

ARMS = [
    ("before", "example-order"),
    ("before", "automatic"),
    ("after", "prepared"),
    ("after", "prepared-custom"),
]
METRICS = [
    "client_connected_ms",
    "agent_ready_ui_ms",
    "first_message_ack_ms",
    "first_rendered_audio_ms",
    "session_returned_ms",
    "subscription_ready_ms",
    "activity_start_delay_ms",
]


def compare(source: Path, output: Path) -> None:
    """Export paired differences as well as independent group medians."""
    analyze([source], output)
    with (output / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sample"] = int(row["sample"])
        for metric in METRICS:
            row[metric] = float(row[metric])
    index = {(r["case"], r["variant"], r["profile"], r["sample"]): r for r in rows}
    cases = sorted({r["case"] for r in rows})
    summaries, contrasts = [], []
    randomizer = random.Random(3610)
    preparation_pair = {(r["variant"], r["profile"]) for r in rows} == {
        ("before", "prepared"),
        ("after", "prepared"),
    }
    arms = [("before", "prepared"), ("after", "prepared")] if preparation_pair else ARMS
    comparisons = (
        [("Native preparation versus synchronous preparation", arms[0], arms[1])]
        if preparation_pair
        else [
            ("Application sequencing + PRs (custom output)", ARMS[0], ARMS[3]),
            ("SDK PRs only (automatic Session)", ARMS[1], ARMS[2]),
            ("Original sequence to default prepared Session", ARMS[0], ARMS[2]),
        ]
    )
    for case in cases:
        for variant, profile in arms:
            group = [
                r
                for r in rows
                if (r["case"], r["variant"], r["profile"]) == (case, variant, profile)
            ]
            summaries.append(
                {
                    "case": case,
                    "variant": variant,
                    "profile": profile,
                    "n": len(group),
                    "medians_ms": {
                        metric: round(statistics.median(r[metric] for r in group), 1)
                        for metric in METRICS
                    },
                }
            )
        for title, before, after in comparisons:
            ids = sorted(
                r["sample"]
                for r in rows
                if (r["case"], r["variant"], r["profile"]) == (case, *before)
            )
            assert all((case, *after, sample) in index for sample in ids)
            metrics = {}
            for metric in METRICS:
                old = [index[(case, *before, sample)][metric] for sample in ids]
                new = [index[(case, *after, sample)][metric] for sample in ids]
                differences = [a - b for a, b in zip(old, new, strict=True)]
                boot = sorted(
                    statistics.median(randomizer.choices(differences, k=len(ids)))
                    for _ in range(10000)
                )
                metrics[metric] = {
                    "before_median_ms": round(statistics.median(old), 1),
                    "after_median_ms": round(statistics.median(new), 1),
                    "difference_of_medians_ms": round(
                        statistics.median(old) - statistics.median(new), 1
                    ),
                    "paired_median_saving_ms": round(statistics.median(differences), 1),
                    "paired_savings_ms": [round(value, 1) for value in differences],
                    "paired_median_bootstrap_95pct_ms": [round(boot[249], 1), round(boot[9749], 1)],
                }
            contrasts.append(
                {"case": case, "comparison": title, "pairs": len(ids), "metrics": metrics}
            )
    (output / "before-after.json").write_text(
        json.dumps({"groups": summaries, "contrasts": contrasts}, indent=2)
    )
    for item in contrasts:
        print(item["case"], item["comparison"])
        for metric in ["agent_ready_ui_ms", "first_message_ack_ms", "first_rendered_audio_ms"]:
            values = item["metrics"][metric]
            print(
                f"  {metric}: {values['before_median_ms']} -> {values['after_median_ms']} ms; paired median saving {values['paired_median_saving_ms']} ms, bootstrap interval {values['paired_median_bootstrap_95pct_ms']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    compare(options.source, options.output)
