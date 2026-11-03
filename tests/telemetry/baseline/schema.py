"""Reduce decoded OTLP captures to a value-free schema snapshot.

Usage: schema.py <decoded dir> <out.json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def vtype(v: dict) -> str:
    ((k, inner),) = v.items() if v else (("empty", None),)
    if k == "array_value":
        types = sorted({vtype(x) for x in inner.get("values", [])})
        return f"array<{'|'.join(types) or 'empty'}>"
    if k == "kvlist_value":
        return "kvlist{" + ",".join(sorted(kv["key"] for kv in inner.get("values", []))) + "}"
    if k == "string_value":
        s = inner.strip()
        if s[:1] in "[{" and s[-1:] in "]}":
            try:
                json.loads(s)
                return "string(json)"
            except ValueError:
                pass
        return "string"
    return k.removesuffix("_value")


def attrs(items: list) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for kv in items or []:
        out[kv["key"]].add(vtype(kv.get("value", {})))
    return out


def merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst.setdefault(k, set()).update(v)


spans: dict[str, dict] = {}
by_id: dict[str, str] = {}
raw_spans = []
resource_keys: dict[str, dict] = defaultdict(dict)
traces = json.loads((D / "traces.json").read_text())
for rs in traces["resource_spans"]:
    merge(resource_keys["traces"], attrs(rs.get("resource", {}).get("attributes")))
    for ss in rs.get("scope_spans", []):
        for s in ss.get("spans", []):
            by_id[s["span_id"]] = s["name"]
            raw_spans.append(s)
for s in raw_spans:
    e = spans.setdefault(
        s["name"],
        {"kind": set(), "parents": set(), "attributes": {}, "events": {}, "status": set()},
    )
    e["kind"].add(s.get("kind", "SPAN_KIND_INTERNAL"))
    e["parents"].add(
        by_id.get(s.get("parent_span_id", ""), "<root or outside>")
        if s.get("parent_span_id")
        else "<root>"
    )
    merge(e["attributes"], attrs(s.get("attributes")))
    for ev in s.get("events", []):
        merge(e["events"].setdefault(ev["name"], {}), attrs(ev.get("attributes")))
    e["status"].add(s.get("status", {}).get("code", "STATUS_CODE_UNSET"))

logs: dict[str, dict] = {}
for rl in json.loads((D / "logs.json").read_text())["resource_logs"]:
    merge(resource_keys["logs"], attrs(rl.get("resource", {}).get("attributes")))
    for sl in rl.get("scope_logs", []):
        scope = sl.get("scope", {}).get("name", "")
        for r in sl.get("log_records", []):
            body = r.get("body", {}).get("string_value", "<non-string body>")
            key = scope if scope in ("chat_history", "evaluations") else "<other scopes>"
            name = body if key != "<other scopes>" else "<log line>"
            e = logs.setdefault(f"{key} / {name}", {"attributes": {}, "scope_attributes": {}})
            merge(e["attributes"], attrs(r.get("attributes")))
            merge(e["scope_attributes"], attrs(sl.get("scope", {}).get("attributes")))

metrics: dict[str, dict] = {}
for rm in json.loads((D / "metrics.json").read_text())["resource_metrics"]:
    merge(resource_keys["metrics"], attrs(rm.get("resource", {}).get("attributes")))
    for sm in rm.get("scope_metrics", []):
        for m in sm.get("metrics", []):
            kind = next(k for k in ("sum", "histogram", "gauge", "exponential_histogram") if k in m)
            e = metrics.setdefault(
                m["name"],
                {"kind": kind, "unit": m.get("unit", ""), "attributes": {}, "temporality": set()},
            )
            body = m[kind]
            e["temporality"].add(body.get("aggregation_temporality", ""))
            for dp in body.get("data_points", []):
                merge(e["attributes"], attrs(dp.get("attributes")))

recordings = {}
for rec in sorted(D.glob("recording_*")):
    header = json.loads((rec / "header.json").read_text())
    summary = json.loads(header.get("session_summary", "{}") or "{}")
    chat = json.loads((rec / "chat_history.bin").read_text())
    recordings = {
        "parts": sorted(p.stem for p in rec.glob("*.bin")),
        "header_fields": sorted(header),
        "session_summary_keys": sorted(summary),
        "session_summary_usage_entry_keys": sorted(
            {k for u in summary.get("usage") or [] for k in u}
        ),
        "chat_history_item_types": sorted({i.get("type") for i in chat.get("items", [])}),
        "chat_history_message_keys": sorted(
            {k for i in chat.get("items", []) if i.get("type") == "message" for k in i}
        ),
    }


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in sorted(x.items())}
    if isinstance(x, set):
        return sorted(x)
    return x


OUT.write_text(
    json.dumps(
        plain(
            {
                "spans": spans,
                "logs": logs,
                "metrics": metrics,
                "resources": resource_keys,
                "recording": recordings,
            }
        ),
        indent=1,
    )
)
print(len(spans), "span names,", len(logs), "log kinds,", len(metrics), "metrics")
