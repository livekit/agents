"""Derive the fakes' calibration from the recordings beside this file (`record.py` wrote them).

Every number the fakes and the person draw from is an empirical sample here, in milliseconds,
taken from what the providers did: `lk-agents-test-support/src/corpus.rs` reads `calibration.json` and draws uniformly
from each list, so a fake's delays are distributed as the real provider's were.

    python3 rust/crates/lk-agents-test-support/corpus/derive.py           # rewrites calibration.json
    python3 rust/crates/lk-agents-test-support/corpus/derive.py --check   # fails if it is stale; writes nothing

Every family, model and quantity must have data: an empty recordings directory, a family nothing
recorded, or a quantity with no samples fails before calibration.json is touched, except a
quantity measured after an event the model was never seen to produce (`CONDITIONAL`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


def load(prefix: str) -> list[dict]:
    return [
        json.loads(p.read_text()) for p in sorted((HERE / "recordings").glob(f"{prefix}*.json"))
    ]


def nearest_before(spans: list[dict], t: float, key: str = "t0") -> dict | None:
    best = None
    for s in spans:
        if s[key] <= t and (best is None or s[key] > best[key]):
            best = s
    return best


# Quantities stated as `[hits, total]`: summed across recordings, not pooled.
PAIRS = {
    "usage_before_final",
    "start_with_first_transcript",
    "function_call_share",
    "duplicate_share",
}


def merge(store: dict, model: str, d: dict) -> None:
    """One recording's quantities into its model's, pooled across recordings."""
    cur = store.setdefault(model, {})
    for k, v in d.items():
        if k in PAIRS:
            cur[k] = [a + b for a, b in zip(cur.get(k, [0, 0]), v, strict=True)]
        elif isinstance(v, dict):
            counts = cur.setdefault(k, {})
            for kk, n in v.items():
                counts[kk] = counts.get(kk, 0) + n
        else:
            cur[k] = sorted(cur.get(k, []) + v)


def realtime() -> dict:
    out: dict[str, list] = {
        "speech_started_lag_semantic": [],
        "speech_started_lag_server": [],
        "speech_stopped_lag_semantic": [],
        "speech_stopped_lag_server": [],
        "created_after_stopped": [],
        "created_after_create": [],
        "item_after_created": [],
        "first_audio_after_item": [],
        "audio_gap": [],
        "audio_delta_ms": [],
        "done_after_last_audio": [],
        "transcription_after_committed": [],
        "transcription_deltas_per_word": [],
        "cancel_after_speech_started": [],
        "audio_ms_per_char": [],
        "committed_after_commit": [],
        "error_after_client": [],
        "done_after_cancel": [],
    }
    order: Counter[str] = Counter()
    tools = {"responses": 0, "with_call": 0, "rate_limits": 0}
    for rec in load("realtime-"):
        td = rec["turn_detection"]
        events = rec["events"]
        spans = rec.get("speech", [])
        client = rec.get("client", [])
        last_type = None
        created_at: dict[str, float] = {}
        item_at: dict[str, float] = {}
        last_audio: dict[str, float] = {}
        chars: dict[str, int] = {}
        audio: dict[str, float] = {}
        committed_at: dict[str, float] = {}
        deltas: Counter[str] = Counter()
        prev_audio_t = None
        started_t = None
        for e in events:
            ty = e["type"]
            t = e["t"]
            if ty != last_type:
                order[f"{last_type} -> {ty}"] += 1
                last_type = ty
            if ty == "input_audio_buffer.speech_started":
                started_t = t
                span = nearest_before(spans, t + 0.05)
                if span is not None and t - span["t0"] < 3:
                    key = "semantic" if td == "semantic_vad" else "server"
                    out[f"speech_started_lag_{key}"].append(ms(t - span["t0"]))
            elif ty == "input_audio_buffer.speech_stopped":
                span = nearest_before(spans, t, "t1")
                if span is not None and t - span["t1"] < 6:
                    key = "semantic" if td == "semantic_vad" else "server"
                    out[f"speech_stopped_lag_{key}"].append(ms(t - span["t1"]))
                stopped_t = t
            elif ty == "input_audio_buffer.committed":
                committed_at[e["item_id"]] = t
                commits = [c["t"] for c in client if c["type"] == "input_audio_buffer.commit"]
                before = [c for c in commits if c <= t]
                if before and td == "none":
                    out["committed_after_commit"].append(ms(t - before[-1]))
            elif ty == "response.created":
                rid = e["response"]["id"]
                created_at[rid] = t
                creates = [c["t"] for c in client if c["type"] == "response.create" and c["t"] <= t]
                if td != "none" and creates and t - creates[-1] < 2:
                    out["created_after_create"].append(ms(t - creates[-1]))
                elif td != "none":
                    out["created_after_stopped"].append(ms(t - stopped_t))
                elif creates:
                    out["created_after_create"].append(ms(t - creates[-1]))
            elif ty == "response.output_item.added" and e["item"]["type"] == "message":
                item_at[e["item"]["id"]] = t
                out["item_after_created"].append(ms(t - created_at[e["response_id"]]))
            elif ty == "response.output_audio.delta":
                iid = e["item_id"]
                if iid not in last_audio:
                    out["first_audio_after_item"].append(ms(t - item_at[iid]))
                elif prev_audio_t is not None:
                    out["audio_gap"].append(ms(t - prev_audio_t))
                prev_audio_t = t
                last_audio[iid] = t
                audio[iid] = audio.get(iid, 0) + e["ms"]
                out["audio_delta_ms"].append(round(e["ms"]))
            elif ty == "response.output_audio_transcript.delta":
                chars[e["item_id"]] = chars.get(e["item_id"], 0) + e.get("chars", 0)
            elif ty == "conversation.item.input_audio_transcription.delta":
                deltas[e["item_id"]] += 1
            elif ty == "conversation.item.input_audio_transcription.completed":
                iid = e["item_id"]
                if iid in committed_at:
                    out["transcription_after_committed"].append(ms(t - committed_at[iid]))
                if e.get("words"):
                    out["transcription_deltas_per_word"].append(round(deltas[iid] / e["words"], 2))
            elif ty == "response.done":
                resp = e["response"]
                tools["responses"] += 1
                if "function_call" in resp["outputs"]:
                    tools["with_call"] += 1
                if (
                    resp["status"] == "cancelled"
                    and resp["reason"] == "turn_detected"
                    and started_t
                ):
                    out["cancel_after_speech_started"].append(ms(t - started_t))
                if resp["status"] == "completed":
                    for at in last_audio.values():
                        if at <= t and t - at < 1:
                            out["done_after_last_audio"].append(ms(t - at))
                    last_audio = {k: v for k, v in last_audio.items() if v > t}
                prev_audio_t = None
            elif ty == "rate_limits.updated":
                tools["rate_limits"] += 1
            elif ty == "error":
                sent = [c["t"] for c in client if c["t"] <= t]
                if sent:
                    out["error_after_client"].append(ms(t - sent[-1]))
            if ty == "response.done" and e["response"]["reason"] == "client_cancelled":
                cancels = [c["t"] for c in client if c["type"] == "response.cancel" and c["t"] <= t]
                if cancels:
                    out["done_after_cancel"].append(ms(t - cancels[-1]))
        for iid, total in audio.items():
            if chars.get(iid):
                out["audio_ms_per_char"].append(round(total / chars[iid], 1))
    # The probes (`probe_realtime.py`) drive the edge cases on purpose: how long the server takes
    # to end a reply the client cancelled, and to refuse a frame (an error citing it).
    for path in sorted((HERE / "probes").glob("probe-*.json")):
        rec = json.loads(path.read_text())
        frames = rec["frames"]
        for e in rec["events"]:
            t = e["t"]
            if e["type"] == "response.done" and e["response"]["reason"] == "client_cancelled":
                cancels = [
                    f["t"]
                    for f in frames
                    if f["frame"]["type"] == "response.cancel" and f["t"] <= t
                ]
                if cancels:
                    out["done_after_cancel"].append(ms(t - cancels[-1]))
            elif e["type"] == "error" and e["error"].get("event_id"):
                cited = [
                    f["t"] for f in frames if f["frame"].get("event_id") == e["error"]["event_id"]
                ]
                if cited:
                    out["error_after_client"].append(ms(t - cited[-1]))
    for v in out.values():
        v.sort()
    out["function_call_share"] = [tools["with_call"], tools["responses"]]
    out["rate_limits_share"] = [tools["rate_limits"], tools["responses"]]
    out["transitions"] = dict(sorted(order.items()))
    return out


def stt() -> dict:
    models = {}
    for rec in load("stt-"):
        spans = rec["speech"]
        ev = rec["events"]
        d: dict[str, list] = {
            "first_after_onset": [],
            "interim_gap": [],
            "final_after_end": [],
            "finals_per_utterance": [],
            "empty_finals_per_minute": [],
            "duplicate_final_after": [],
            "final_after_preflight": [],
        }
        firsts: dict[int, float] = {}
        finals: Counter[int] = Counter()
        preflights: Counter[int] = Counter()
        last_preflight = None
        prev_interim = None
        prev_final = None
        empty = 0
        for e in ev:
            t = e["t"]
            span = nearest_before(spans, t)
            idx = spans.index(span) if span else -1
            if e["type"] == "interim_transcript":
                if idx not in firsts and span:
                    firsts[idx] = t
                    d["first_after_onset"].append(ms(t - span["t0"]))
                elif prev_interim is not None and t - prev_interim < 1.5:
                    d["interim_gap"].append(ms(t - prev_interim))
                prev_interim = t
            if e["type"] == "preflight_transcript":
                preflights[idx] += 1
                last_preflight = t
            if e["type"] == "final_transcript":
                prev_interim = None
                if last_preflight is not None:
                    d["final_after_preflight"].append(ms(t - last_preflight))
                    last_preflight = None
                if e.get("words", 0) == 0:
                    empty += 1
                    continue
                if prev_final and t - prev_final[0] < 0.5:
                    # A second final for words already finalised: AssemblyAI's formatted turn.
                    d["duplicate_final_after"].append(ms(t - prev_final[0]))
                    prev_final = None
                    continue
                prev_final = (t, e["words"])
                if span:
                    finals[idx] += 1
                    if idx not in firsts:
                        firsts[idx] = t
                        d["first_after_onset"].append(ms(t - span["t0"]))
                    if t >= span["t1"]:
                        d["final_after_end"].append(ms(t - span["t1"]))
        d["finals_per_utterance"] = sorted(finals.values())
        # Per utterance with words, how many preflights preceded its finals: zero for a model
        # that sends none.
        d["preflights_per_utterance"] = sorted(preflights.get(i, 0) for i in finals)
        d["duplicate_share"] = [len(d["duplicate_final_after"]), sum(finals.values())]
        duration_min = (ev[-1]["t"] - ev[0]["t"]) / 60 if ev else 1
        d["empty_finals_per_minute"] = [round(empty / duration_min, 2)]
        d["usage_before_final"] = [
            sum(
                1
                for a, b in pairwise(ev)
                if a["type"] == "recognition_usage" and b["type"] == "final_transcript"
            ),
            sum(1 for e in ev if e["type"] == "final_transcript"),
        ]
        d["start_with_first_transcript"] = [
            sum(
                1
                for a, b in pairwise(ev)
                if a["type"] == "start_of_speech" and b["t"] - a["t"] < 0.01
            ),
            sum(1 for e in ev if e["type"] == "start_of_speech"),
        ]
        for v in d.values():
            v.sort()
        merge(models, rec["model"], d)
    return models


def tts() -> dict:
    out = {}
    for rec in load("tts-"):
        d: dict[str, list] = {"first_audio": [], "ms_per_word": [], "chunk_gap": [], "chunks": []}
        for e in rec["events"]:
            ch = e.get("chunks") or []
            if not ch:
                continue
            d["first_audio"].append(ms(ch[0]["t"]))
            total = sum(c["ms"] for c in ch)
            d["ms_per_word"].append(round(total / max(1, e["words"])))
            d["chunks"].append(len(ch))
            for a, b in pairwise(ch):
                d["chunk_gap"].append(ms(b["t"] - a["t"]))
        for v in d.values():
            v.sort()
        merge(out, rec["model"], d)
    return out


def llm() -> dict:
    out = {}
    for rec in load("llm-"):
        d: dict[str, list] = {"first_chunk": [], "chunk_gap": [], "chars_per_chunk": []}
        shapes = Counter()
        for call in rec["events"]:
            ch = [c for c in call["chunks"] if "error" not in c]
            if not ch:
                continue
            d["first_chunk"].append(ms(ch[0]["t"]))
            for a, b in pairwise(ch):
                d["chunk_gap"].append(ms(b["t"] - a["t"]))
            for c in ch:
                if c["chars"]:
                    d["chars_per_chunk"].append(c["chars"])
                shape = (
                    "usage"
                    if c["usage"]
                    else "tool_call"
                    if c["tool_calls"]
                    else ("text" if c["chars"] else "empty")
                )
                shapes[shape] += 1
            shapes[f"last:{'usage' if ch[-1]['usage'] else 'other'}"] += 1
        for v in d.values():
            v.sort()
        d["shapes"] = dict(shapes)
        merge(out, rec["model"], d)
    return out


# Quantities measured only after an event a model may never produce, by the quantity counting
# that event: empty exactly when the event was never seen (Flux has no duplicate finals, the
# others no preflights).
CONDITIONAL = {
    "final_after_preflight": lambda q: not any(q.get("preflights_per_utterance", [])),
    "duplicate_final_after": lambda q: q.get("duplicate_share", [0])[0] == 0,
}


def empty(cal: dict) -> list[str]:
    """The families, models and quantities with no samples, by path."""
    out = []

    def walk(node: object, path: str, parent: dict) -> None:
        name = path.rsplit("/", 1)[-1]
        if isinstance(node, dict):
            if not node:
                out.append(path)
            for k, v in node.items():
                walk(v, f"{path}/{k}", node)
        elif isinstance(node, list) and not node:
            if not (name in CONDITIONAL and CONDITIONAL[name](parent)):
                out.append(path)

    for family, node in cal.items():
        walk(node, family, {})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--check", action="store_true", help="compare, write nothing")
    args = p.parse_args()
    cal = {"realtime": realtime(), "stt": stt(), "tts": tts(), "llm": llm()}
    if missing := empty(cal):
        sys.exit(f"derive: no samples for {', '.join(missing)}; calibration.json left as it is")
    text = json.dumps(cal, indent=1) + "\n"
    target = HERE / "calibration.json"
    if args.check:
        if target.read_text() != text:
            sys.exit("derive: calibration.json is stale; rerun derive.py and commit it")
        print("calibration.json matches the recordings")
        return
    target.write_text(text)
    print(
        json.dumps(
            {
                k: {kk: (vv if not isinstance(vv, list) else vv[:12]) for kk, vv in v.items()}
                if k == "realtime"
                else v
                for k, v in cal.items()
            },
            indent=1,
        )[:6000]
    )


if __name__ == "__main__":
    main()
