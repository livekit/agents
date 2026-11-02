"""Surviving Python bootstrap contracts (G-8 and G-9)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.parametrize("failure", ["startup", "drain"])
def test_launcher_failure_exits_nonzero_after_cleanup(failure: str) -> None:
    code = """
        import sys
        from livekit.agents import AgentServer, cli

        class FailingServer(AgentServer):
            async def _run(self, **kwargs):
                if FAILURE == "startup":
                    raise RuntimeError("startup refused")
            async def _drain(self):
                print("drained")
                if FAILURE == "drain":
                    raise RuntimeError("drain refused")
            async def aclose(self):
                print("closed")

        server = FailingServer()
        sys.argv = ["agent.py", "start"]
    """
    code = f"FAILURE = {failure!r}\n" + textwrap.dedent(code)
    code += "\ncli.run_app(server)\n"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert f"{failure} refused" in result.stdout + result.stderr
    assert "drained" in result.stdout and "closed" in result.stdout
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "arguments, status", [("--help", 0), ("start --help", 0), ("download-files", 2)]
)
def test_cli_diagnostics_do_not_run_the_server(arguments: str, status: int) -> None:
    code = textwrap.dedent("""
        import sys
        from livekit.agents import AgentServer, cli

        def fail(*args):
            raise AssertionError("server must not be loaded")

        cli._run_worker = fail
        cli._run_console = fail
    """)
    code += f"sys.argv = ['agent.py', *{arguments!r}.split()]\n"
    code += "cli.run_app(AgentServer())\n"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    if status == 0:
        assert "Usage:" in result.stdout
    else:
        assert "download-files" in result.stderr
