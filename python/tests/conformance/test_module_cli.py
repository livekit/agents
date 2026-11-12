"""The Python module launcher executes the same entrypoint as a direct script launch."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "command", [["start", "--dev"], ["console", "--connect-addr", "127.0.0.1:9"]]
)
def test_module_launch_runs_main_and_preserves_script_imports_and_arguments(
    tmp_path: Path, command: list[str]
) -> None:
    (tmp_path / "settings.py").write_text('MESSAGE = "sibling imported"\n')
    entrypoint = tmp_path / "my agent.py"
    entrypoint.write_text(
        textwrap.dedent("""
            import json
            import sys
            from settings import MESSAGE
            from livekit.agents import AgentServer, cli

            class Server(AgentServer):
                async def _run(self, **kwargs):
                    print(MESSAGE)
                async def _drain(self):
                    print("drained")
                async def aclose(self):
                    print("closed")

            if __name__ == "__main__":
                print("argv=" + json.dumps(sys.argv))
                cli.run_app(Server())
        """)
    )
    result = subprocess.run(
        [sys.executable, "-m", "livekit.agents", command[0], str(entrypoint), *command[1:]],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "argv=" + json.dumps([str(entrypoint), *command]) in result.stdout
    assert "sibling imported" in result.stdout and "closed" in result.stdout


@pytest.mark.parametrize("flags, status", [(["--help"], 0), (["--unknown-option"], 2)])
def test_module_cli_diagnostics_do_not_execute_the_entrypoint(
    tmp_path: Path, flags: list[str], status: int
) -> None:
    entrypoint = tmp_path / "agent.py"
    entrypoint.write_text('raise AssertionError("application must not execute")\n')
    result = subprocess.run(
        [sys.executable, "-m", "livekit.agents", "start", str(entrypoint), *flags],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_module_cli_reports_a_missing_entrypoint(tmp_path: Path) -> None:
    entrypoint = tmp_path / "missing.py"
    result = subprocess.run(
        [sys.executable, "-m", "livekit.agents", "start", str(entrypoint)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "entrypoint is not a file" in result.stderr
    assert "Traceback" not in result.stderr
