#!/usr/bin/env python3
"""Pin an example's in-repo requirements at a git ref, in place.

A deploy build context is the example directory alone, so in-repo requirements
have to resolve over the network. Pinning them at the deployed ref builds the
agent against the code at that ref rather than the latest release on PyPI.

    python scripts/render_example_requirements.py examples/hotel_receptionist --ref main
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO = "git+https://github.com/livekit/agents.git"

# The package name (and optional [extras]) at the start of a requirement,
# e.g. "livekit-agents[evals]>=1.5.7".
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)(?P<extras>\[.*\])?")


def monorepo_path(name: str) -> str | None:
    """Path of `name` within this repo, or None if it lives elsewhere.

    We only pin packages that actually exist in the checkout. The directory
    basename is the distribution name for every package here (livekit-agents at
    the root, everything else under livekit-plugins/). Anything not found
    (livekit rtc, livekit-blingfire, livekit-local-inference, …) keeps its PyPI
    pin.
    """
    for rel in (name, f"livekit-plugins/{name}"):
        if (REPO_ROOT / rel).is_dir():
            return rel
    return None


def pin_to_ref(line: str, ref: str) -> str:
    """Repoint an in-repo requirement at the deployed git ref.

    External requirements, comments and blanks pass through untouched (the
    regex doesn't match a leading '#' or '').
    """
    match = REQUIREMENT.match(line.strip())
    if match is None:
        return line
    path = monorepo_path(match["name"])
    if path is None:
        return line
    return f"{match['name']}{match['extras'] or ''} @ {REPO}@{ref}#subdirectory={path}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", type=Path, help="path to the example directory")
    parser.add_argument(
        "--ref",
        required=True,
        help="git ref in-repo requirements are pinned to (branch, tag or sha)",
    )
    args = parser.parse_args()

    requirements = args.example / "requirements.txt"
    pinned = [pin_to_ref(line, args.ref) for line in requirements.read_text().splitlines()]
    requirements.write_text("\n".join(pinned) + "\n")

    print(f"--- {requirements} pinned to {args.ref} ---")
    print(requirements.read_text(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
