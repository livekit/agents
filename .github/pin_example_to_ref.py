#!/usr/bin/env python3
"""Pin an example's in-repo dependencies to a git ref, in place.

An example resolves its livekit-* dependencies through the workspace root's
``[tool.uv.sources]``, which only applies inside this repo. A deploy build
context is the example directory alone, so appending a git source for every
in-repo dependency is what builds the image against this checkout's code
instead of the latest release on PyPI.

    python .github/pin_example_to_ref.py examples/hotel_receptionist --ref main

Needs Python 3.11+ for tomllib.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/livekit/agents.git"

# The distribution name at the start of a requirement, e.g. "livekit-agents[mcp]>=1.6".
# Extras stay on the dependency itself; a source is keyed by name alone.
REQUIREMENT = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)")


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


def git_sources(dependencies: list[str], ref: str) -> str:
    lines = [
        "",
        "# Written by .github/pin_example_to_ref.py for a deploy: the build context is",
        "# this directory alone, so the in-repo dependencies come from the git ref being",
        f"# deployed ({ref}) rather than from PyPI.",
        "[tool.uv.sources]",
    ]
    for dep in dependencies:
        match = REQUIREMENT.match(dep)
        if match is None:
            continue
        path = monorepo_path(match["name"])
        if path is None:
            continue
        lines.append(
            f'{match["name"]} = {{ git = "{REPO}", rev = "{ref}", subdirectory = "{path}" }}'
        )
    return "\n".join(lines) + "\n"


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
    contents = pyproject.read_text()
    parsed = tomllib.loads(contents)
    if "sources" in parsed.get("tool", {}).get("uv", {}):
        parser.error(f"{pyproject} already declares [tool.uv.sources]")

    sources = git_sources(parsed["project"]["dependencies"], args.ref)
    pyproject.write_text(contents + sources)
    print(f"--- {pyproject} pinned to {args.ref} ---")
    print(sources, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
