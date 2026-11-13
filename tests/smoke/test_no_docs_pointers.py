"""Code states the rule it follows and never points at a page of the repository's docs
(`docs/<path>.md`): the pointer drifts from the page it names. A URL to another site is not a
pointer. Rust sources and manifests are held to the same rule by `rust/check`."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCANNED = (ROOT / "python", ROOT / "tests", ROOT / "rust" / "crates")
# the word before `docs/`, so a URL (`https://...`) can be told apart
POINTER = re.compile(r"(\S*)docs/[\w/-]+\.md(?!\w)")


def _pointers(path: Path) -> list[str]:
    return [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if any("://" not in match[1] for match in POINTER.finditer(line))
    ]


# Catches: a comment or docstring that says "see the docs page" instead of the rule, which goes
# stale when the page moves or changes.
def test_no_python_source_points_at_a_docs_page() -> None:
    files = [
        path
        for root in SCANNED
        for path in sorted(root.rglob("*.py"))
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    found = [pointer for path in files for pointer in _pointers(path)]

    assert found == [], (
        "points at a docs page: state the rule or contract it stands for in a few words\n"
        + "\n".join(found)
    )
