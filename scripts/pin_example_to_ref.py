#!/usr/bin/env python3
"""Render an example's requirements.txt from its pyproject.toml.

Each example is a uv workspace member, so its in-repo dependencies resolve
through ``[tool.uv.sources]`` and only inside the workspace. A deploy build
context is the example directory alone, so the image installs from a rendered
requirements.txt instead, with in-repo names repointed at a git ref.

    python scripts/render_example_requirements.py examples/hotel_receptionist --ref main

Needs Python 3.11+ for tomllib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO = "git+https://github.com/livekit/agents.git"

# The distribution name (and optional [extras]) at the start of a requirement,
# e.g. "livekit-agents[mcp]>=1.6".
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?P<extras>\[.*\])?")


def monorepo_path(name: str) -> str | None:
    """Path of ``name`` within this repo, or None if it lives elsewhere.

    Only packages that actually exist in the checkout are pinned. The directory
    basename is the distribution name for every package here (livekit-agents at
    the root, everything else under livekit-plugins/). Anything not found
    (livekit rtc, livekit-blingfire, livekit-local-inference, …) keeps its PyPI
    specifier.
    """
    for rel in (name, f"livekit-plugins/{name}"):
        if (REPO_ROOT / rel).is_dir():
            return rel
    return None


def pin_to_ref(requirement: str, ref: str) -> str:
    match = REQUIREMENT.match(requirement)
    if match is None:
        return requirement
    path = monorepo_path(match["name"])
    if path is None:
        return requirement
    return f"{match['name']}{match['extras'] or ''} @ {REPO}@{ref}#subdirectory={path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path, help="path to the example directory")
    parser.add_argument(
        "--ref",
        required=True,
        help="git ref in-repo dependencies are pinned to (branch, tag or sha)",
    )
    args = parser.parse_args()

    pyproject = args.example / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    pinned = [pin_to_ref(dep, args.ref) for dep in project["dependencies"]]

    requirements = args.example / "requirements.txt"
    requirements.write_text("\n".join(pinned) + "\n")
    print(f"--- {requirements} rendered at {args.ref} ---")
    print(requirements.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
