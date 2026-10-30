"""Bounded native ownership of the CLI's final event-loop cleanup."""

import asyncio
import contextlib
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.no_concurrent]


async def _run_cleanup(code: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-c", code]
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        assert process.stdout is not None
        # Framework imports have their own allowance; cleanup keeps its twelve-second bound.
        prefix = await asyncio.wait_for(process.stdout.readuntil(b"cleanup started\n"), timeout=30)
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=12)
        assert process.returncode is not None
        return subprocess.CompletedProcess(
            command, process.returncode, (prefix + stdout).decode(), stderr.decode()
        )
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.communicate()


@pytest.mark.parametrize(
    "blocking",
    [
        "task",
        "executor",
        pytest.param(
            "stderr", marks=pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipe flags")
        ),
    ],
)
def test_stuck_loop_shutdown_exits_nonzero_while_retaining_cleanup(blocking: str) -> None:
    code = textwrap.dedent("""
        import asyncio
        import time
        from livekit.agents import cli, _core

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def stuck():
            while True:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    print("cancellation served", flush=True)

        async def prepare():
            if BLOCKING != "executor":
                loop.create_task(stuck())
            else:
                loop.run_in_executor(None, time.sleep, 60)
            await asyncio.sleep(0)

        loop.run_until_complete(prepare())
        if BLOCKING == "stderr":
            import os
            read_end, write_end = os.pipe()
            os.set_blocking(write_end, False)
            try:
                while True:
                    os.write(write_end, b"x" * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(write_end, True)
            os.dup2(write_end, 2)
            print("stderr saturated", flush=True)
        print("cleanup started", flush=True)
        cli._close_loop(loop, _core.Shutdown(0.5, 0.0))
        print("cleanup falsely completed", flush=True)
    """)
    result = asyncio.run(_run_cleanup(f"BLOCKING = {blocking!r}\n" + code))
    assert "cleanup started" in result.stdout
    if blocking != "executor":
        assert "cancellation served" in result.stdout
    assert "cleanup falsely completed" not in result.stdout
    assert result.returncode == 1
    if blocking == "stderr":
        assert "stderr saturated" in result.stdout
    elif sys.platform != "win32":
        assert result.stderr.count("shutdown deadline exceeded") == 1


def test_cooperative_loop_shutdown_disarms_the_native_deadline() -> None:
    code = textwrap.dedent("""
        import asyncio
        import time
        from livekit.agents import cli, _core

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def pending():
            try:
                await asyncio.sleep(60)
            finally:
                print("task settled", flush=True)

        loop.create_task(pending())
        loop.run_until_complete(asyncio.sleep(0))
        print("cleanup started", flush=True)
        cli._close_loop(loop, _core.Shutdown(0.5, 0.0))
        print("loop closed", loop.is_closed(), flush=True)
        time.sleep(0.75)
    """)
    result = asyncio.run(_run_cleanup(code))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task settled" in result.stdout and "loop closed True" in result.stdout
    assert "shutdown deadline exceeded" not in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX termination signals")
@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGINT"])
@pytest.mark.parametrize("module_launch", [False, True])
@pytest.mark.parametrize(
    "behavior",
    ["cooperative", "drain_timeout", "blocked_loop", "blocked_cleanup", "blocked_startup"],
)
async def test_signals_share_one_budget_across_drain_and_cleanup(
    tmp_path: Path, behavior: str, signal_name: str, module_launch: bool
) -> None:
    """Native signal capture starts the deadline even before Python can respond."""
    import os
    import signal

    code = textwrap.dedent("""
        import asyncio
        import sys
        import time
        from livekit.agents import AgentServer, cli

        class Server(AgentServer):
            async def _run(self, **kwargs):
                print("ready", flush=True)
                if BEHAVIOR == "blocked_startup":
                    time.sleep(60)
                await asyncio.Event().wait()

            async def _drain(self):
                print("draining", flush=True)
                if BEHAVIOR == "blocked_loop":
                    time.sleep(60)
                await asyncio.sleep(60 if BEHAVIOR == "drain_timeout" else 0.15)
                print("drained", flush=True)

            async def aclose(self):
                print("closing", flush=True)
                await asyncio.sleep(60 if BEHAVIOR == "blocked_cleanup" else 0.02)
                print("closed", flush=True)

        cli.run_app(Server(shutdown_timeout=0.65, drain_timeout=0.25))
        print("finished", flush=True)
    """)
    entrypoint = tmp_path / "agent.py"
    entrypoint.write_text(f"BEHAVIOR = {behavior!r}\n" + code)
    command = (
        ["-m", "livekit.agents", "start", str(entrypoint)]
        if module_launch
        else [str(entrypoint), "start"]
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert process.stdout is not None
        await asyncio.wait_for(process.stdout.readuntil(b"ready\n"), timeout=30)
        clock = asyncio.get_running_loop().time
        started = clock()
        os.kill(process.pid, getattr(signal, signal_name))
        await asyncio.sleep(0.03)
        os.kill(process.pid, getattr(signal, signal_name))
        output, errors = await asyncio.wait_for(process.communicate(), timeout=4)
        stdout, stderr = output.decode(), errors.decode()
        elapsed = clock() - started
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
    if behavior != "blocked_startup":
        assert "draining" in stdout
    if behavior in {"cooperative", "drain_timeout"}:
        assert process.returncode == 0, stdout + stderr
        assert "closed" in stdout and "finished" in stdout
        assert "shutdown deadline exceeded" not in stderr
        if behavior == "drain_timeout":
            assert "drained" not in stdout
    else:
        assert process.returncode == 1, stdout + stderr
        assert "shutdown deadline exceeded" in stderr
        assert 0.5 <= elapsed < 1.5
        assert "finished" not in stdout
