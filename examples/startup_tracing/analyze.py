"""Summarize measured startup critical paths and export browser/agent trace timelines."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def analyze(inputs: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    examples = []
    for root in inputs:
        for client in json.loads((root / "clients.json").read_text()):
            case = client["scenario"]
            if case["sample"] < 0:
                continue
            events = [
                json.loads(line)
                for line in (root / "traces" / f"{case['run_id']}.jsonl").read_text().splitlines()
            ]
            marks = defaultdict(list)
            spans = []
            for event in events:
                if event["name"] == "span":
                    spans.append(event)
                else:
                    marks[event["name"]].append(event["at_ms"])
            t0 = client["marks"]["click"]

            def first(name, marks=marks, run_id=case["run_id"]):
                if not marks[name]:
                    raise ValueError(f"Missing {name}: {run_id}")
                return min(marks[name])

            pcm = first("first_pcm_at_output")
            subscribed = first("output_subscription_end")
            publication_duration = (
                first("track_publish_end") - first("track_publish_begin")
                if marks["track_publish_begin"]
                else None
            )
            post_join = subscribed - (
                first("track_publish_begin")
                if publication_duration is not None
                else first("room_connect_end")
            )
            exposed_wait = max(0, subscribed - pcm)
            row = {
                "run_id": case["run_id"],
                "case": case["case"],
                "profile": case["profile"],
                "variant": case.get("variant", "after"),
                "sample": case["sample"],
                "connecting_paint_ms": client["marks"]["connecting_ui_painted"] - t0,
                "client_connected_ms": client["marks"]["room_connected"] - t0,
                "agent_ready_ui_ms": (
                    client["marks"]["agent_interaction_ready_painted"] - t0
                    if "agent_interaction_ready_painted" in client["marks"]
                    else None
                ),
                "first_message_ack_ms": (
                    client["marks"]["first_message_ack"] - t0
                    if "first_message_ack" in client["marks"]
                    else None
                ),
                "entrypoint_ms": first("entrypoint") - t0,
                "agent_room_connected_ms": first("room_connect_end") - t0,
                "activity_ready_ms": first("activity_start_end") - t0,
                "activity_start_delay_ms": first("activity_start_begin")
                - first("session_start_begin"),
                "session_returned_ms": first("session_start_end") - t0,
                "first_pcm_ms": pcm - t0,
                "publication_ms": publication_duration,
                "subscription_ready_ms": subscribed - t0,
                "first_rendered_audio_ms": client["marks"]["first_rendered_audio"] - t0,
                "pcm_waiting_for_media_ms": exposed_wait,
                "media_slack_before_pcm_ms": max(0, pcm - subscribed),
                "instant_registration_saving_bound_ms": (
                    min(publication_duration, exposed_wait)
                    if publication_duration is not None
                    else None
                ),
                "post_join_media_setup_ms": post_join,
                "instant_post_join_media_saving_bound_ms": min(post_join, exposed_wait),
                "browser_ice_rtt_ms": client.get("iceRttMs"),
            }
            rows.append(row)
            lanes = [
                ("Client joins", t0, client["marks"]["room_connected"]),
                (
                    "Dispatch → entrypoint",
                    min(marks.get("received_at", [first("entrypoint")])),
                    first("entrypoint"),
                ),
                ("Agent connects", first("room_connect_begin"), first("room_connect_end")),
                (
                    "Activity initialization",
                    first("activity_start_begin"),
                    first("activity_start_end"),
                ),
                ("Produce first PCM", first("on_enter"), pcm),
                *(
                    [("Publication", first("track_publish_begin"), first("track_publish_end"))]
                    if publication_duration is not None
                    else []
                ),
                ("Subscription wait", first("output_subscription_begin"), subscribed),
                ("First PCM waits for media", pcm, max(pcm, subscribed)),
                (
                    "Ready PCM → browser render",
                    max(pcm, subscribed),
                    client["marks"]["first_rendered_audio"],
                ),
            ]
            for span in spans:
                if span["span_name"] == "event_loop_blocked":
                    lanes.append(("Observed event-loop block", span["start_ms"], span["end_ms"]))
            trace = [
                {
                    "name": name,
                    "ph": "X",
                    "pid": 1,
                    "tid": index,
                    "ts": (start - t0) * 1000,
                    "dur": (end - start) * 1000,
                }
                for index, (name, start, end) in enumerate(lanes)
            ]
            trace += [
                {
                    "name": event["name"],
                    "ph": "i",
                    "s": "g",
                    "pid": 1,
                    "tid": 0,
                    "ts": (event["at_ms"] - t0) * 1000,
                }
                for event in events
                if event["name"] != "span"
            ]
            (output / f"{case['run_id']}.trace.json").write_text(json.dumps({"traceEvents": trace}))
            examples.append((case, row, lanes, t0))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["case"], row["profile"])].append(row)
    summaries = []
    for (case, profile), group in sorted(grouped.items()):
        summary = {"case": case, "profile": profile, "n": len(group)}
        for metric in rows[0]:
            values = sorted(row[metric] for row in group if isinstance(row[metric], (int, float)))
            if values:
                summary[metric] = {
                    "median": round(statistics.median(values), 1),
                    "p95": round(values[math.ceil(0.95 * len(values)) - 1], 1),
                }
        summaries.append(summary)
        print(
            case,
            profile,
            f"n={len(group)}",
            "audio",
            summary["first_rendered_audio_ms"],
            "PCM/media wait",
            summary["pcm_waiting_for_media_ms"],
            "instant registration bound",
            summary.get("instant_registration_saving_bound_ms"),
        )
    comparisons = []
    lookup = {(row["case"], row["profile"], row["sample"]): row for row in rows}
    for case in sorted({row["case"] for row in rows}):
        for variant in ["parallel-output", "automatic", "prepared"]:
            ids = sorted(
                {
                    row["sample"]
                    for row in rows
                    if row["case"] == case
                    and (case, "example-order", row["sample"]) in lookup
                    and (case, variant, row["sample"]) in lookup
                }
            )
            if not ids:
                continue
            differences = [
                lookup[(case, "example-order", index)]["first_rendered_audio_ms"]
                - lookup[(case, variant, index)]["first_rendered_audio_ms"]
                for index in ids
            ]
            rng = random.Random(3610)
            bootstrap = sorted(
                statistics.median(rng.choices(differences, k=len(ids))) for _ in range(10000)
            )
            comparisons.append(
                {
                    "case": case,
                    "baseline": "example-order",
                    "variant": variant,
                    "pairs": len(ids),
                    "median_paired_saving_ms": round(statistics.median(differences), 1),
                    "paired_bootstrap_95_interval_ms": [
                        round(bootstrap[249], 1),
                        round(bootstrap[9749], 1),
                    ],
                }
            )
    (output / "results.json").write_text(
        json.dumps({"summary": summaries, "samples": rows, "comparisons": comparisons}, indent=2)
    )

    with (output / "samples.csv").open("w") as target:
        writer = csv.DictWriter(target, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    body = [
        '<!doctype html><meta charset="utf-8"><title>Agent startup critical paths</title>',
        "<style>body{font:15px system-ui;background:#101827;color:#f1f5f9;margin:32px}section{margin:30px 0;padding:20px;background:#182334;border-radius:12px}svg{width:100%;max-width:1200px}text{fill:#ddd;font-size:12px}h2{font-size:19px}.note{max-width:1100px;color:#b8c5d8}</style>",
        "<h1>Agent startup: measured browser and agent timelines</h1>",
        '<p class="note">Real browser-triggered Cloud agent dispatch and real audio transport. Controlled cases use synthetic audio and explicit activity/PCM delays; they are not production model measurements. Bars use measured timestamps on this host. “Instant registration” is a counterfactual upper bound, not a measured optimization.</p>',
    ]
    for summary in summaries:
        group = [
            e
            for e in examples
            if (e[0]["case"], e[0]["profile"]) == (summary["case"], summary["profile"])
        ]
        chosen = min(
            group,
            key=lambda e: abs(
                e[1]["first_rendered_audio_ms"] - summary["first_rendered_audio_ms"]["median"]
            ),
        )
        case, row, lanes, t0 = chosen
        end = max(item["first_rendered_audio_ms"] for item in rows) + 100
        body.append(
            f"<section><h2>{html.escape(case['case'])} / {html.escape(case['profile'])} — median first audio {summary['first_rendered_audio_ms']['median']} ms</h2>"
        )
        body.append(
            f"<p>First PCM: {row['first_pcm_ms']:.0f} ms · media ready: {row['subscription_ready_ms']:.0f} ms · wait: {row['pcm_waiting_for_media_ms']:.0f} ms · media slack: {row['media_slack_before_pcm_ms']:.0f} ms</p>"
        )
        body.append(f'<svg viewBox="0 0 1200 {len(lanes) * 40 + 45}" role="img">')
        for i, (name, start, stop) in enumerate(lanes):
            x = 235 + 900 * (start - t0) / end
            width = max(2, 900 * (stop - start) / end)
            color = "#f6ad55" if "wait" in name.lower() else "#60a5fa"
            body.append(
                f'<text x="0" y="{30 + i * 40}">{html.escape(name)}</text><rect x="{x:.1f}" y="{15 + i * 40}" width="{width:.1f}" height="22" rx="4" fill="{color}"><title>{html.escape(name)}: {start - t0:.1f}–{stop - t0:.1f} ms</title></rect>'
            )
        for tick in range(0, int(end), 500):
            axis_x = 235 + 900 * tick / end
            body.append(
                f'<line x1="{axis_x:.1f}" x2="{axis_x:.1f}" y1="10" y2="{len(lanes) * 40}" stroke="#94a3b8" opacity=".15"/><text x="{axis_x:.1f}" y="{len(lanes) * 40 + 25}">{tick} ms</text>'
            )
        body.append("</svg></section>")
    (output / "timeline.html").write_text("\n".join(body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.inputs, args.output)
