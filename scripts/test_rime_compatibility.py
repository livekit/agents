# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "protobuf>=6.33.5",
#     "rime-api>=0.0.1",
#     "rimelabs-sdk==0.1.0a26",
#     "pytest==9.1.1",
#     "pytest-asyncio==1.4.0",
#     "pytest-asyncio-concurrent==0.5.2",
# ]
# ///
"""Check Rime source compatibility with a published LiveKit core.

Run with uv --no-project --with 'livekit-agents[codecs]==VERSION'.
This does not test package installation.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import distribution, version
from pathlib import Path

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "livekit-plugins/livekit-plugins-rime"


class RequiredChecks:
    """Fail on missing tests, unexpected skips, or changed expected failures."""

    def __init__(self, core_version: str) -> None:
        self.expected_xfails = (
            {
                "tests/test_plugin_rime.py::TestRealtime::"
                "test_standard_session_tool_limit_produces_final_reply_and_restores_tools"
                f"[{steps}]"
                for steps in (0, 1, 2)
            }
            if Version(core_version) < Version("1.8.6")
            else set()
        )
        self.xfails: set[str] = set()
        self.skipped: set[str] = set()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.skipped.add(report.nodeid)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped:
            if getattr(report, "wasxfail", None):
                self.xfails.add(report.nodeid)
            else:
                self.skipped.add(report.nodeid)

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.skipped or self.xfails != self.expected_xfails or not session.testscollected:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            print(
                "\nCompatibility check incomplete: "
                f"skipped={sorted(self.skipped)}, "
                f"expected_xfails={sorted(self.expected_xfails)}, "
                f"actual_xfails={sorted(self.xfails)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-version", required=True)
    parser.add_argument("--junitxml")
    parser.add_argument("--sdk-source", type=Path)
    args = parser.parse_args()
    if Version(args.core_version) < Version("1.7.1"):
        parser.error("this compatibility check starts at LiveKit 1.7.1")

    # Expose Rime source to this process and the import-test subprocesses.
    # Verify below that core comes from the installed distribution.
    sys.path[:0] = [str(PLUGIN), str(ROOT)]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(PLUGIN), str(ROOT), os.environ.get("PYTHONPATH", "")) if path
    )
    from livekit import agents
    from livekit.plugins import rime

    assert version("livekit-agents") == args.core_version
    installed_core = distribution("livekit-agents").locate_file("livekit/agents/__init__.py")
    assert Path(agents.__file__).resolve() == installed_core.resolve()
    assert not installed_core.resolve().is_relative_to(ROOT / "livekit-agents")
    assert Path(rime.__file__).resolve().is_relative_to(PLUGIN)
    print(f"Published core {args.core_version}: {agents.__file__}")
    print(f"Rime source: {rime.__file__}")
    print(f"Rime SDK: {version('rimelabs-sdk')}")
    import rimelabs_sdk

    if args.sdk_source:
        assert Path(rimelabs_sdk.__file__).resolve().is_relative_to(args.sdk_source.resolve())
    print(f"Rime SDK source: {rimelabs_sdk.__file__}")
    files = [
        ROOT / "tests/test_plugin_rime.py",
        ROOT / "tests/test_plugin_rime_reliability.py",
        ROOT / "tests/test_plugin_rime_playback_lifetime.py",
    ]
    command = ["-c", str(ROOT / "pyproject.toml"), "--unit", "--no-concurrent", "-q", "-rx"]
    if args.junitxml:
        command.extend(["--junitxml", args.junitxml])
    return int(
        pytest.main([*command, "--", *map(str, files)], plugins=[RequiredChecks(args.core_version)])
    )


if __name__ == "__main__":
    raise SystemExit(main())
