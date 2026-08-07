"""Graceful-shutdown signal handling tests for the worker CLI.

A signal handler that raises would surface the exception inside whatever frame
the main thread happens to be executing — e.g. a user request_fnc doing blocking
I/O inside a job-request task — instead of at the run_until_complete boundary,
losing the shutdown entirely (https://github.com/livekit/agents/issues/6724).
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.unit

_WORKER_SCRIPT = """
import asyncio
import time

from livekit.agents import AgentServer, JobContext
from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_worker

server = AgentServer(
    ws_url="ws://127.0.0.1:1",  # unreachable: the worker retries forever
    api_key="devkey",
    api_secret="devsecret",
    max_retry=100000,
    num_idle_processes=0,
)


@server.rtc_session()
async def entry(ctx: JobContext) -> None:
    pass


_orig_run = server.run


async def _run(*args, **kwargs):
    async def _block_loop() -> None:
        await asyncio.sleep(0.1)
        print("BLOCKING", flush=True)
        # a synchronous call inside a task blocks the event loop on the main
        # thread, the state a blocking user request_fnc puts the worker in
        time.sleep(3)
        print("UNBLOCKED", flush=True)

    asyncio.ensure_future(_block_loop())
    return await _orig_run(*args, **kwargs)


server.run = _run

_run_worker(server, proto.CliArgs(log_level="DEBUG", simulation=True))
print("CLEAN_EXIT", flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM semantics differ on Windows")
def test_sigterm_during_blocked_event_loop_shuts_down_worker() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        assert proc.stdout is not None
        deadline = time.monotonic() + 30
        for line in proc.stdout:
            if "BLOCKING" in line:
                break
            if time.monotonic() > deadline:
                pytest.fail("worker subprocess never reached the blocking section")
        else:
            pytest.fail(f"worker subprocess exited early (rc={proc.wait()})")

        # deliver a single SIGTERM while the event loop is blocked; the worker
        # must still drain and exit once the blocking call returns
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 0, f"worker exited with {proc.returncode}:\n{out}"
    assert "CLEAN_EXIT" in out
    assert "Task exception was never retrieved" not in out
    assert "_ExitCli" not in out
