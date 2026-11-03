"""Compare two schema snapshots (schema.py's output) and print what changed, as Markdown.

Usage: diff.py <old schema.json> <new schema.json>

Each section lists what only the old one has (`-`), what only the new one has (`+`) and, for what
both have, the fields that differ: span kind, parents, status, attribute keys and their OTLP
types, events and their keys; log scopes, bodies and attribute keys; metric kind, unit,
temporality and label keys; resource keys; the recording upload's parts, header fields, session
summary keys and chat history items.
"""

import json
import sys
from pathlib import Path


def keyed(old: dict, new: dict, prefix: str = "") -> list[str]:
    """Keys only one side has, and keys whose type sets differ."""
    out = [f"- `{prefix}{k}` {old[k]}" for k in sorted(old.keys() - new.keys())]
    out += [f"+ `{prefix}{k}` {new[k]}" for k in sorted(new.keys() - old.keys())]
    out += [
        f"~ `{prefix}{k}` {old[k]} -> {new[k]}"
        for k in sorted(old.keys() & new.keys())
        if old[k] != new[k]
    ]
    return out


def listed(field: str, old: list, new: list) -> list[str]:
    gone, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    out = [f"- {field} `{x}`" for x in gone]
    out += [f"+ {field} `{x}`" for x in added]
    return out


def spans(old: dict, new: dict) -> list[str]:
    out = [f"- span `{n}`" for n in sorted(old.keys() - new.keys())]
    out += [f"+ span `{n}`" for n in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        a, b = old[name], new[name]
        lines = listed("kind", a["kind"], b["kind"])
        lines += listed("parent", a["parents"], b["parents"])
        lines += listed("status", a["status"], b["status"])
        lines += keyed(a["attributes"], b["attributes"])
        ea, eb = a["events"], b["events"]
        lines += [f"- event `{e}`" for e in sorted(ea.keys() - eb.keys())]
        lines += [f"+ event `{e}`" for e in sorted(eb.keys() - ea.keys())]
        for e in sorted(ea.keys() & eb.keys()):
            lines += keyed(ea[e], eb[e], f"{e}.")
        if lines:
            out.append(f"~ span `{name}`")
            out += [f"    {line}" for line in lines]
    return out


def logs(old: dict, new: dict) -> list[str]:
    out = [f"- log `{n}`" for n in sorted(old.keys() - new.keys())]
    out += [f"+ log `{n}`" for n in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        lines = keyed(old[name]["attributes"], new[name]["attributes"])
        lines += keyed(old[name]["scope_attributes"], new[name]["scope_attributes"], "scope.")
        if lines:
            out.append(f"~ log `{name}`")
            out += [f"    {line}" for line in lines]
    return out


def metrics(old: dict, new: dict) -> list[str]:
    out = [f"- metric `{n}` ({old[n]['kind']})" for n in sorted(old.keys() - new.keys())]
    out += [f"+ metric `{n}` ({new[n]['kind']})" for n in sorted(new.keys() - old.keys())]
    for name in sorted(old.keys() & new.keys()):
        a, b = old[name], new[name]
        lines = [
            f"~ {field} `{a[field]}` -> `{b[field]}`"
            for field in ("kind", "unit")
            if a[field] != b[field]
        ]
        lines += listed("temporality", a["temporality"], b["temporality"])
        lines += keyed(a["attributes"], b["attributes"])
        if lines:
            out.append(f"~ metric `{name}`")
            out += [f"    {line}" for line in lines]
    return out


def resources(old: dict, new: dict) -> list[str]:
    out = []
    for signal in sorted(old.keys() | new.keys()):
        lines = keyed(old.get(signal, {}), new.get(signal, {}))
        if lines:
            out.append(f"~ {signal}")
            out += [f"    {line}" for line in lines]
    return out


def recording(old: dict, new: dict) -> list[str]:
    out = []
    for field in sorted(old.keys() | new.keys()):
        out += listed(field, old.get(field, []), new.get(field, []))
    return out


def main() -> None:
    old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    print(f"# `{old_path.name}` -> `{new_path.name}`\n")
    print("`-` only in the old snapshot, `+` only in the new one, `~` in both but different.\n")
    for title, compare in (
        ("Spans", spans),
        ("Logs", logs),
        ("Metrics", metrics),
        ("Resources", resources),
        ("Recording upload", recording),
    ):
        key = title.split()[0].lower()
        lines = compare(old.get(key, {}), new.get(key, {}))
        print(f"## {title}\n")
        print("```diff" if lines else "No differences.")
        if lines:
            print("\n".join(lines))
            print("```")
        print()


if __name__ == "__main__":
    main()
