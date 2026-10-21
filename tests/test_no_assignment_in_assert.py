"""Assignment expressions inside ``assert`` are skipped when Python runs with ``-O``.

``assert (x := value) is not None`` never binds ``x`` under ``python -O`` (or
``PYTHONOPTIMIZE``), so any later use of ``x`` raises ``UnboundLocalError``
(https://github.com/livekit/agents/issues/8044). Assign first, then assert.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = [REPO_ROOT / "livekit-agents", REPO_ROOT / "livekit-plugins"]


def _source_files() -> list[Path]:
    files: list[Path] = []
    for source_dir in SOURCE_DIRS:
        files.extend(p for p in source_dir.rglob("*.py") if "tests" not in p.parts)
    return sorted(files)


def _asserts_with_assignment(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        and any(isinstance(child, ast.NamedExpr) for child in ast.walk(node))
    ]


def test_no_assignment_expression_in_assert() -> None:
    files = _source_files()
    assert len(files) > 100

    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in files
        for line in _asserts_with_assignment(path)
    ]
    assert not offenders, (
        "assignment expressions inside assert are skipped under `python -O`; "
        "assign the value before asserting: " + ", ".join(offenders)
    )
