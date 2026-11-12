"""The CLI launcher in real OS processes: module launch, failure exits, and one shutdown budget
across drain and the final event-loop cleanup, including signals while a real job process runs.

The agent files stub ``AgentServer``'s private ``_run``, ``_drain`` and ``aclose`` to script a
server that is slow, stuck or failing at a chosen step; the CLI around them is the real one.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .resources import StartJobWorker, gone

SHUTDOWN_BUDGET = """
import asyncio
import os
import time
from livekit.agents import AgentServer, cli


async def ignore_cancellation():
    while True:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            print("cancelled", flush=True)


def saturate_stderr():
    read_end, write_end = os.pipe()
    os.set_blocking(write_end, False)
    try:
        while True:
            os.write(write_end, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(write_end, True)
    os.dup2(write_end, 2)


class Server(AgentServer):
    async def _run(self, **kwargs):
        if BEHAVIOR in ("stuck_task", "saturated_stderr"):
            asyncio.ensure_future(ignore_cancellation())
        if BEHAVIOR == "saturated_stderr":
            saturate_stderr()
        if BEHAVIOR == "stuck_executor":
            asyncio.get_running_loop().run_in_executor(None, time.sleep, 60)
        print("ready", flush=True)
        if BEHAVIOR == "blocked_startup":
            time.sleep(60)
        await asyncio.Event().wait()

    async def _drain(self, timeout):
        if BEHAVIOR == "blocked_drain":
            time.sleep(60)
        await asyncio.wait_for(asyncio.sleep(60 if BEHAVIOR == "slow_drain" else 0.15), timeout)
        print("drained", flush=True)

    async def aclose(self):
        await asyncio.sleep(60 if BEHAVIOR == "blocked_close" else 0.02)


cli.run_app(Server(shutdown_timeout=0.65, drain_timeout=0.25))
time.sleep(1)  # past the deadline: a finished shutdown disarmed it
print("finished", flush=True)
"""


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
@pytest.mark.parametrize(
    ("behavior", "exit_code"),
    [
        ("cooperative", 0),
        ("slow_drain", 0),  # the drain is cut at drain_timeout, then the server closes
        ("blocked_startup", 1),
        ("blocked_drain", 1),
        ("blocked_close", 1),
        ("stuck_task", 1),
        ("stuck_executor", 1),
        ("saturated_stderr", 1),  # the deadline cannot even report itself
    ],
)
async def test_shutdown_ends_within_one_budget_whatever_blocks_it(
    tmp_path: Path, behavior: str, exit_code: int, signal_name: str
) -> None:
    """A signal ends the process within ``shutdown_timeout``: cleanly when the server settles,
    with exit code 1 when anything blocks the drain, the close or the event loop's cleanup."""
    agent = tmp_path / "agent.py"
    agent.write_text(f"BEHAVIOR = {behavior!r}\n" + SHUTDOWN_BUDGET)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-u", str(agent), "start", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        assert process.stdout is not None
        await process.stdout.readuntil(b"ready\n")
        clock = asyncio.get_running_loop().time
        started = clock()

        os.kill(process.pid, getattr(signal, signal_name))
        out, err = await process.communicate()
        elapsed = clock() - started
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
    stdout, stderr = out.decode(), err.decode()

    assert process.returncode == exit_code, stdout + stderr
    if exit_code == 0:
        assert "finished" in stdout
        assert ("drained" in stdout) is (behavior == "cooperative")
    else:
        assert 0.5 <= elapsed < 1.5
        assert "finished" not in stdout
    if behavior != "saturated_stderr":
        assert ("shutdown deadline exceeded" in stderr) is (exit_code == 1)
    if behavior == "stuck_task":
        assert "cancelled" in stdout, "the cleanup never cancelled the task"


MODULE_LAUNCH = """
import json
import sys
from settings import MESSAGE
from livekit.agents import AgentServer, cli

class Server(AgentServer):
    async def _run(self, **kwargs):
        print(MESSAGE)
    async def _drain(self):
        pass
    async def aclose(self):
        print("closed")

if __name__ == "__main__":
    print("argv=" + json.dumps(sys.argv))
    cli.run_app(Server())
"""


@pytest.mark.parametrize(
    "command",
    [["start", "--dev"], ["console", "--connect-addr", "127.0.0.1:9"]],
    ids=lambda c: c[0],
)
def test_python_m_livekit_agents_runs_the_file_as_main_with_its_folder_importable(
    tmp_path: Path, command: list[str]
) -> None:
    """``python -m livekit.agents <cmd> <file>`` runs the file as ``__main__`` with its own argv,
    its sibling modules importable."""
    (tmp_path / "settings.py").write_text('MESSAGE = "sibling imported"\n')
    agent = tmp_path / "my agent.py"
    agent.write_text(MODULE_LAUNCH)

    run = subprocess.run(
        [sys.executable, "-m", "livekit.agents", command[0], str(agent), *command[1:]],
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    assert "argv=" + json.dumps([str(agent), *command]) in run.stdout
    assert "sibling imported" in run.stdout and "closed" in run.stdout


FAILING_SERVER = """
import sys
from livekit.agents import AgentServer, cli

class Server(AgentServer):
    async def _run(self, **kwargs):
        if FAILURE == "startup":
            raise RuntimeError("startup refused")
    async def _drain(self, timeout=None):
        print("drained")
        if FAILURE == "drain":
            raise RuntimeError("drain refused")
    async def aclose(self):
        print("closed")

cli.run_app(Server())
"""


@pytest.mark.parametrize("failure", ["startup", "drain"])
def test_a_server_failing_at_startup_or_drain_exits_nonzero_after_closing(
    tmp_path: Path, failure: str
) -> None:
    """A server raising at startup or while draining still drains and closes, names its error,
    and exits nonzero."""
    agent = tmp_path / "agent.py"
    agent.write_text(f"FAILURE = {failure!r}\n" + textwrap.dedent(FAILING_SERVER))

    run = subprocess.run([sys.executable, str(agent), "start"], capture_output=True, text=True)

    assert run.returncode != 0, run.stdout + run.stderr
    assert f"{failure} refused" in run.stdout + run.stderr
    assert "drained" in run.stdout and "closed" in run.stdout


@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("module_launch", [False, True], ids=["script", "python_m"])
def test_a_signal_drains_and_reaps_a_running_job_process(
    job_worker: StartJobWorker, shutdown_signal: int, module_launch: bool
) -> None:
    """A signal to a worker running a job in its own process ends that job (its shutdown
    callbacks run), reaps the process and exits 0."""
    w = job_worker(
        "cli_shutdown", module_launch=module_launch, shutdown_timeout=5.0, drain_timeout=0.4
    )
    job = json.loads(w.marker("room-0.started"))

    os.kill(w.proc.pid, shutdown_signal)

    assert w.wait(timeout=8) == 0, w.tail()
    assert (w.dir / "room-0.shutdown").exists(), w.tail()
    assert gone(job["pid"], 3), "the CLI left its job process alive"
