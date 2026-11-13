"""A hermetic test never waits out real time: it awaits the event it needs (`until`), or moves
the core's clock (`_core.skip_host_time`). A real wait is slow on every run and flaky under load.
`tests/fakes.py` (whose `until` polls) and `tests/e2e/` (live services) are outside the rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

TESTS = Path(__file__).parents[1]
OUTSIDE = (TESTS / "fakes.py", TESTS / "e2e")
# the longest sleep allowed: a yield or a short poll, never a duration the test relies on
LONGEST = 0.05
SLEEPS = {"sleep"}
# a line carrying this comment, with its reason, is exempt: the real duration is under test
EXEMPT = "# real wait:"


def _seconds(arg: ast.expr, constants: dict[str, float]) -> float | None:
    """The literal or module-constant duration a sleep is given; ``None`` when computed."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, int | float):
        return float(arg.value)
    if isinstance(arg, ast.Name):
        return constants.get(arg.id)
    return None


def _real_waits(path: Path) -> list[str]:
    found = _sleeps(path.read_text(), path, first_line=1)
    # a script the test runs in a child process is a string: its sleeps count too
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "sleep(" in node.value
        ):
            try:
                found += _sleeps(textwrap.dedent(node.value), path, first_line=node.lineno)
            except SyntaxError:
                continue
    return found


def _sleeps(source: str, path: Path, first_line: int) -> list[str]:
    lines = source.splitlines()
    tree = ast.parse(source)
    constants = {
        target.id: float(node.value.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int | float)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute | ast.Name)
            and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id) in SLEEPS
            and node.args
        ):
            continue
        seconds = _seconds(node.args[0], constants)
        if seconds is None or seconds <= LONGEST:
            continue
        if any(EXEMPT in lines[n - 1] for n in range(node.lineno, (node.end_lineno or 0) + 1)):
            continue
        line = first_line + node.lineno - 1
        found.append(f"{path.relative_to(TESTS.parent)}:{line}: sleeps {seconds:g} s")
    return found


# Catches: a test that sleeps for a duration (a grace period, "long enough for the peer") instead
# of awaiting what it needs, which is slow on every run and fails under load.
def test_no_hermetic_test_sleeps_longer_than_a_poll() -> None:
    files = [
        path
        for path in sorted(TESTS.rglob("*.py"))
        if not any(path == outside or path.is_relative_to(outside) for outside in OUTSIDE)
    ]
    found = [wait for path in files for wait in _real_waits(path)]

    assert found == [], (
        f"sleeps longer than {LONGEST} s: await the event with `until`, skip the core's time "
        f"with `_core.skip_host_time`, or mark the line `{EXEMPT} <why the duration is under "
        "test>`\n" + "\n".join(found)
    )
