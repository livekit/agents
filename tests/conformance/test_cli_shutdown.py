"""Bounded native ownership of the CLI's final event-loop cleanup."""

import asyncio
import contextlib
import subprocess
import sys
import textwrap

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
        from livekit.agents import cli

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
        cli._close_loop(loop)
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
        assert result.stderr.count("incomplete event-loop cleanup") == 1


def test_cooperative_loop_shutdown_disarms_the_native_deadline() -> None:
    code = textwrap.dedent("""
        import asyncio
        import time
        from livekit.agents import cli

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
        cli._close_loop(loop)
        print("loop closed", loop.is_closed(), flush=True)
        time.sleep(5.5)
    """)
    result = asyncio.run(_run_cleanup(code))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "task settled" in result.stdout and "loop closed True" in result.stdout
    assert "incomplete event-loop cleanup" not in result.stderr
