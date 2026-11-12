"""The event-loop stall detector: a blocked loop is reported with its stack, a closed watch is no
longer checked, and its py-spy sampler never takes the process down.

Each case runs in a fresh interpreter: the detector and its sampler are process-wide, start at the
first watch, and a sampler fault aborts the whole process rather than failing a test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# A watched asyncio loop that blocks in `block_the_loop_here`, then stops its watch.
_BLOCKED_LOOP = """
import asyncio, logging, time
from livekit.agents import telemetry

logging.basicConfig(level=logging.INFO)
telemetry._install()

def block_the_loop_here():
    time.sleep(0.4)

async def main():
    monitor = telemetry._LoopMonitor()
    monitor.start("AJ_stall", whole_process=True)
    await asyncio.sleep(0.1)
    {before_block}
    block_the_loop_here()
    await asyncio.sleep(0.1)
    monitor.stop()

asyncio.run(main())
"""

# Threads start, recurse and exit nonstop around a watch's sampler.
_THREAD_CHURN = """
import threading, time
from livekit.agents import _core

stopping = False

def recurse(depth):
    if depth == 0:
        time.sleep(0)  # hand the GIL over: stacks change while the sampler reads them
        return 0
    return recurse(depth - 1) + 1

def churn():
    while not stopping:
        threads = [threading.Thread(target=recurse, args=(50,)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

def open_watch():
    return _core._core_LoopWatch(
        threading.get_native_id(), parent=None, job="AJ_churn", whole_process=True
    )

{program}
"""


def _run(program: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_a_blocked_loop_is_reported_with_the_blocking_function() -> None:
    """A loop blocked past the threshold logs one warning naming the function that blocked it."""
    result = _run(_BLOCKED_LOOP.format(before_block="pass"))

    assert result.returncode == 0, result.stderr
    assert result.stderr.count("event loop blocked for") == 1, result.stderr
    assert "at block_the_loop_here (" in result.stderr, result.stderr


def test_a_closed_watch_reports_nothing() -> None:
    """Once its watch is closed, a loop that blocks is no longer reported."""
    result = _run(_BLOCKED_LOOP.format(before_block="monitor.stop()"))

    assert result.returncode == 0, result.stderr
    assert "event loop blocked" not in result.stderr, result.stderr


@pytest.mark.skipif(
    sys.platform != "linux", reason="macOS frameworks abort a forked child of a threaded process"
)
def test_a_forked_child_reports_its_own_blocked_loop() -> None:
    """A child forked from a watched process starts its own detector and reports its stalls."""
    fork_a_watched_process = """
import os, threading, time
from livekit.agents import _core

watch = _core._core_LoopWatch(
    threading.get_native_id(), parent=None, job="AJ_parent", whole_process=True
)
time.sleep(0.5)  # the parent's detector has attached and samples
if (child := os.fork()) != 0:
    watch.close()
    os._exit(os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]))
_core.init(_core.Role.Job)
"""
    result = _run(fork_a_watched_process + _BLOCKED_LOOP.format(before_block="pass"))

    assert result.returncode == 0, result.stderr
    assert "at block_the_loop_here (" in result.stderr, result.stderr


def test_the_sampler_attaches_beside_a_thread_deeper_than_py_spy_reads() -> None:
    """A thread more than 4096 frames deep when the sampler attaches never aborts the process."""
    result = _run(
        """
        import sys, threading, time
        from livekit.agents import _core

        sys.setrecursionlimit(10_000)
        deep, release = threading.Event(), threading.Event()

        def recurse(depth):
            if depth == 0:
                deep.set()
                release.wait()
                return
            recurse(depth - 1)

        threading.Thread(target=recurse, args=(5_000,), daemon=True).start()
        deep.wait()
        watch = _core._core_LoopWatch(
            threading.get_native_id(), parent=None, job="AJ_deep", whole_process=True
        )
        time.sleep(0.3)
        watch.close()
        release.set()
        """
    )

    assert result.returncode == 0, result.stderr


def test_the_sampler_survives_threads_starting_and_exiting_while_it_attaches() -> None:
    """Sampling a process whose threads come and go never aborts it."""
    program = """
for _ in range(4):
    threading.Thread(target=churn, daemon=True).start()
time.sleep(0.05)  # the churn is under way when the sampler attaches
watch = open_watch()
deadline = time.monotonic() + 0.5
while time.monotonic() < deadline:
    recurse(100)
watch.close()
"""
    # Torn reads while attaching aborted about half of these children.
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _THREAD_CHURN.format(program=program)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]

    stderrs = [child.communicate(timeout=30)[1] for child in children]

    crashed = [
        stderr for child, stderr in zip(children, stderrs, strict=True) if child.returncode != 0
    ]
    assert not crashed, f"{len(crashed)}/{len(children)} children aborted:\n{crashed[0]}"


def test_exiting_with_a_watch_open_never_aborts_the_process() -> None:
    """The interpreter can finalize while the sampler is still reading its threads."""
    result = _run(
        _THREAD_CHURN.format(
            program="""
watch = open_watch()
time.sleep(0.15)  # attached while the process is quiet
for _ in range(4):
    threading.Thread(target=churn, daemon=True).start()
time.sleep(0.35)
"""
        )
    )

    assert result.returncode == 0, result.stderr


def test_a_forked_child_can_log_after_its_parent_reported_a_stall() -> None:
    """A forked child's first core log line never takes it down (it once trapped on macOS)."""
    result = _run(
        """
        import logging, os, threading, time
        from livekit.agents import _core, telemetry

        logging.basicConfig()
        watch = _core._core_LoopWatch(
            threading.get_native_id(), parent=None, job=None, whole_process=False
        )
        time.sleep(0.4)
        watch.beat()
        time.sleep(0.1)
        _core.flush_logs()  # the stall's report started the core's log drain
        if (child := os.fork()) == 0:
            telemetry._install()  # logs that the core was inherited across a fork
            _core.flush_logs()
            os._exit(0)
        os._exit(os.waitstatus_to_exitcode(os.waitpid(child, 0)[1]))
        """
    )

    assert result.returncode == 0, result.stderr
    assert "event loop blocked for" in result.stderr, result.stderr
    assert "inherited across a fork" in result.stderr, result.stderr
