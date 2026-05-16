#!/usr/bin/env python3
"""Build a manifest JSON for all examples that ship a playground.yaml.

Walks the `examples/` directory, picks up every `playground.yaml`, reads the
adjacent README, parses `livekit.toml` for the deployed agent_id/subdomain, and
emits a single `examples-manifest.json` file.

Usage:
    python scripts/build_examples_manifest.py --root . --out manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ACCENT_PALETTE = ["cyan", "green", "amber", "blue", "violet", "rose", "pink", "teal"]


def parse_pixel_icon(icon_field: Any) -> dict | None:
    if not isinstance(icon_field, dict):
        return None
    size = int(icon_field.get("size", 16))
    pixels_raw = icon_field.get("pixels", "")
    rows = [r for r in str(pixels_raw).splitlines() if r.strip() != ""]
    if not rows:
        return None
    return {"size": size, "rows": rows}


def load_example(example_dir: Path) -> dict | None:
    yaml_path = example_dir / "playground.yaml"
    if not yaml_path.is_file():
        return None

    meta = yaml.safe_load(yaml_path.read_text()) or {}

    accent = meta.get("accent", "cyan")
    if accent not in ACCENT_PALETTE:
        print(f"warning: {example_dir.name} uses unknown accent '{accent}', defaulting to cyan", file=sys.stderr)
        accent = "cyan"
    accent_index = ACCENT_PALETTE.index(accent)

    readme_name = meta.get("readme", "README.md")
    readme_path = example_dir / readme_name
    readme = readme_path.read_text() if readme_path.is_file() else ""

    agent_id = ""
    subdomain = ""
    toml_path = example_dir / "livekit.toml"
    if toml_path.is_file():
        try:
            data = tomllib.loads(toml_path.read_text())
            agent_id = data.get("agent", {}).get("id", "") or ""
            subdomain = data.get("project", {}).get("subdomain", "") or ""
        except Exception as exc:
            print(f"warning: failed to parse {toml_path}: {exc}", file=sys.stderr)

    return {
        "slug": example_dir.name,
        "title": meta.get("title", example_dir.name),
        "description": meta.get("description", ""),
        "accent": accent,
        "accent_index": accent_index,
        "entry": meta.get("entry", ""),
        "readme_path": readme_name,
        "readme": readme,
        "tags": meta.get("tags", []) or [],
        "order": int(meta.get("order", 9999)),
        "deploy": bool(meta.get("deploy", False)),
        "agent_id": agent_id,
        "subdomain": subdomain,
        "icon": parse_pixel_icon(meta.get("icon")),
    }


def build_manifest(root: Path) -> dict:
    examples_dir = root / "examples"
    if not examples_dir.is_dir():
        raise SystemExit(f"examples directory not found at {examples_dir}")

    items: list[dict] = []
    for entry in sorted(examples_dir.iterdir()):
        if not entry.is_dir():
            continue
        item = load_example(entry)
        if item is not None:
            items.append(item)

    items.sort(key=lambda x: (x["order"], x["slug"]))

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "examples": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Path to the agents repo root")
    parser.add_argument("--out", default="-", help="Output path (- for stdout)")
    args = parser.parse_args()

    manifest = build_manifest(Path(args.root))
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text)
        print(f"wrote {len(manifest['examples'])} examples to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
