"""Job runners as the worker builds them, and the process-tree evidence their cleanup checks.

A job runner (``_core.JobSupervisor`` over a ``JobLauncher``) is how the worker runs one job in
its own process; the tests that need a single runner build one here instead of a whole server.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Required, TypeAlias, TypedDict

import psutil

from livekit.agents import _core
from livekit.agents.ipc import JobLauncher, JobRunnerOptions
from livekit.agents.job import JobAcceptArguments, RunningJobInfo

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from livekit.agents import JobContext
    from livekit.protocol import agent

from collections.abc import Awaitable, Callable

OwnRunner: TypeAlias = Callable[[_core.JobSupervisor], _core.JobSupervisor]
"""The ``own_executor`` fixture: registers a runner the test starts; teardown closes and reaps it."""


class ProcessRecord(TypedDict, total=False):
    """One process as psutil saw it."""

    pid: Required[int]
    created: float
    status: str
    threads: int
    handles: int
    rss: int
    uss: int
    command: list[str]
    error: str


def job_runner(
    entrypoint: Callable[[JobContext], Awaitable[None]],
    *,
    session_end_fnc: Callable[[JobContext], Awaitable[None]] | None = None,
    session_end_timeout: float = 1,
    mp_ctx: BaseContext | None = None,
    preload_modules: tuple[str, ...] = (),
    user_arguments: object = None,
    thread: bool = False,
    initialize_timeout: float = 10,
    close_timeout: float = 10,
    ping_interval: float = 2.5,
    ping_timeout: float = 60,
    high_ping_threshold: float = 0.5,
    env: tuple[tuple[str, str], ...] = (),
) -> _core.JobSupervisor:
    """One job runner over a fresh launcher, with these ``JobRunnerOptions``; ``env`` is the
    server environment its jobs see (their LiveKit API key and secret)."""
    return _core.JobSupervisor(
        JobLauncher(
            job_entrypoint_fnc=entrypoint,
            session_end_fnc=session_end_fnc,
            simulation_end_fnc=None,
            session_end_timeout=session_end_timeout,
            preload_modules=preload_modules,
            mp_ctx=mp_ctx,
            user_arguments=user_arguments,
            loop=asyncio.get_running_loop(),
            env=env,
        ),
        JobRunnerOptions(
            thread=thread,
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            high_ping_threshold=high_ping_threshold,
        ),
    )


def running_job(job: agent.Job, *, fake_job: bool = True) -> RunningJobInfo:
    """The worker's record of `job`, as a dispatch would hand it over."""
    return RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="", identity="", metadata=""),
        job=job,
        url="",
        token="",
        worker_id="",
        fake_job=fake_job,
        received_at=None,
        accepted_at=None,
        assigned_at=None,
    )


def process_snapshot(process: psutil.Process) -> ProcessRecord:
    record: ProcessRecord = {"pid": process.pid}
    with process.oneshot():
        try:
            record["created"] = process.create_time()
            record["status"] = process.status()
            record["threads"] = process.num_threads()
            record["handles"] = process.num_fds()
            memory = process.memory_full_info()
            record["rss"], record["uss"] = memory.rss, memory.uss
            record["command"] = process.cmdline()
        except psutil.Error as error:
            record["error"] = repr(error)
    return record


def tree_snapshot(pid: int) -> list[ProcessRecord]:
    try:
        process = psutil.Process(pid)
        return [
            process_snapshot(process),
            *(process_snapshot(p) for p in process.children(recursive=True)),
        ]
    except psutil.NoSuchProcess:
        return []


def process_group_snapshot(pid: int) -> list[ProcessRecord]:
    records = []
    for process in psutil.process_iter():
        with contextlib.suppress(ProcessLookupError, PermissionError, psutil.Error):
            if os.getpgid(process.pid) == pid:
                records.append(process_snapshot(process))
    return records


def living_processes(records: list[ProcessRecord]) -> list[ProcessRecord]:
    living = []
    for record in records:
        with contextlib.suppress(psutil.NoSuchProcess):
            process = psutil.Process(record["pid"])
            if process.create_time() == record.get("created"):
                living.append(process_snapshot(process))
    return living


def terminate_processes(records: list[ProcessRecord]) -> None:
    for record in reversed(records):
        with contextlib.suppress(psutil.Error):
            process = psutil.Process(record["pid"])
            if process.create_time() == record.get("created"):
                process.kill()


async def close_owned_executor(executor: _core.JobSupervisor, artifact_dir: Path) -> None:
    """Capture evidence before cleanup; a missed deadline remains a failure after emergency reaping."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(executor.pid) if executor.pid else []
    artifact = artifact_dir / f"executor-{executor.id}.json"
    artifact.write_text(json.dumps({"before": before}, indent=2))
    failure: BaseException | None = None
    try:
        async with asyncio.timeout(5):
            await executor.aclose()
            await executor.join()
    except BaseException as error:
        failure = error
        observed = {record["pid"]: record for record in before}
        if executor.pid:
            observed.update({record["pid"]: record for record in tree_snapshot(executor.pid)})
        before = list(observed.values())
        artifact.write_text(json.dumps({"failure": repr(error), "before": before}, indent=2))
        terminate_processes(before)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(executor.join(), timeout=5)
    finally:
        after: list[ProcessRecord] = []
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            after = living_processes(before)
            if not after or asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.01)
        artifact.write_text(
            json.dumps(
                {"failure": repr(failure) if failure else None, "before": before, "after": after},
                indent=2,
            )
        )
    if failure:
        raise AssertionError(
            f"IPC fixture missed cleanup deadline; evidence: {artifact}"
        ) from failure
    if after:
        raise AssertionError(
            f"IPC fixture retained processes (including zombies); evidence: {artifact}"
        )


def run_owned_process(
    command: list[str], artifact_dir: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a command with a watchdog, retaining its process tree before any emergency cleanup."""
    import signal
    import time

    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "stdout.txt"
    artifact = artifact_dir / "processes.json"
    known: dict[int, ProcessRecord] = {}
    failure: BaseException | None = None
    with output.open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                known.update({record["pid"]: record for record in tree_snapshot(process.pid)})
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    process.wait(timeout=min(0.02, remaining))
                except subprocess.TimeoutExpired:
                    pass
            known.update({record["pid"]: record for record in process_group_snapshot(process.pid)})
            cleanup_deadline = time.monotonic() + 5
            while living_processes(list(known.values())):
                if time.monotonic() >= cleanup_deadline:
                    raise AssertionError("owned descendants outlived the subprocess")
                time.sleep(0.01)
        except BaseException as error:
            failure = error
        finally:
            before: list[ProcessRecord] = []
            try:
                known.update(
                    {record["pid"]: record for record in process_group_snapshot(process.pid)}
                )
                before = living_processes(list(known.values()))
            except BaseException as error:
                failure = failure or error
            artifact.write_text(
                json.dumps(
                    {
                        "command": command,
                        "failure": repr(failure) if failure else None,
                        "before": before,
                        "observed": list(known.values()),
                    },
                    indent=2,
                )
            )
            # Inspection failures still retain the session's kernel ownership for emergency cleanup.
            if before or failure:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                terminate_processes(list(known.values()))
            process.wait(timeout=5)
            cleanup_deadline = time.monotonic() + 5
            after = living_processes(list(known.values()))
            while after and time.monotonic() < cleanup_deadline:
                time.sleep(0.01)
                after = living_processes(list(known.values()))
            artifact.write_text(
                json.dumps(
                    {
                        "command": command,
                        "exit_code": process.returncode,
                        "failure": repr(failure) if failure else None,
                        "before": before,
                        "observed": list(known.values()),
                        "after": after,
                    },
                    indent=2,
                )
            )
    if failure:
        raise AssertionError(f"process watchdog failed; evidence: {artifact}") from failure
    if after:
        raise AssertionError(f"process watchdog retained owned processes; evidence: {artifact}")
    return subprocess.CompletedProcess(command, process.returncode, output.read_text(), None)
