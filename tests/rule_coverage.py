"""``pytest.mark.rule("R32", "B15")`` names the checklist rules and bugs a test pins, and
``--coverage-report <path>`` writes them as a markdown table: id, its text, each test, outcome.

``--coverage-checklist <path>`` is the checklist whose table rows (``| R1 | text | ...``) give the
ids and their texts; with it every id is listed, the untested ones too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("rule coverage")
    group.addoption("--coverage-report", metavar="PATH", help="write the rule coverage table here")
    group.addoption(
        "--coverage-checklist", metavar="PATH", help="the checklist with the rule and bug texts"
    )


def register(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "rule(*ids): the checklist rules and bugs a test pins.")
    if path := config.getoption("--coverage-report"):
        config.pluginmanager.register(
            RuleCoverage(Path(path), config.getoption("--coverage-checklist"))
        )


class RuleCoverage:
    """Collects each marked test's outcome and writes the table when the session ends."""

    def __init__(self, report: Path, checklist: str | None) -> None:
        self.report = report
        self.texts: dict[str, str] = {}
        if checklist:
            rows = re.findall(r"^\| ([RB]\d+) \| (.*?) \|", Path(checklist).read_text(), re.M)
            self.texts = dict(rows)
        self.ids: dict[str, tuple[str, ...]] = {}
        self.outcomes: dict[str, str] = {}

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            ids = tuple(i for m in item.iter_markers("rule") for i in m.args)
            if ids:
                self.ids[item.nodeid] = ids

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.nodeid not in self.ids or self.outcomes.get(report.nodeid) == "fail":
            return
        if hasattr(report, "wasxfail"):
            self.outcomes[report.nodeid] = "xfail" if report.skipped else "xpass"
        elif report.failed:
            self.outcomes[report.nodeid] = "fail"
        elif report.skipped:
            self.outcomes[report.nodeid] = "skip"
        elif report.when == "call":
            self.outcomes[report.nodeid] = "pass"

    def pytest_sessionfinish(self) -> None:
        tests: dict[str, list[str]] = {i: [] for i in self.texts}
        for nodeid, ids in self.ids.items():
            for i in ids:
                tests.setdefault(i, []).append(nodeid)
        lines = ["| id | rule or bug | test | outcome |", "|---|---|---|---|"]
        for i in sorted(tests, key=lambda i: (i[0], int(i[1:]))):
            text = self.texts.get(i, "").replace("|", "\\|")
            for nodeid in tests[i] or [""]:
                outcome = self.outcomes.get(nodeid, "not run") if nodeid else "untested"
                lines.append(f"| {i} | {text} | {nodeid and f'`{nodeid}`'} | {outcome} |")
        self.report.write_text("\n".join(lines) + "\n")
