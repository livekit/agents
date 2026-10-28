"""Shared ownership and pre-cleanup evidence for real worker process tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from livekit.agents.ipc import JobExecutor


def process_snapshot(process: psutil.Process) -> dict[str, Any]:
    record: dict[str, Any] = {"pid": process.pid}
    with process.oneshot():
        try:
            record["created"] = process.create_time()
            record["status"] = process.status()
            record["threads"] = process.num_threads()
            record["native_thread_ids"] = [thread.id for thread in process.threads()]
            if process.pid == os.getpid():
                import threading

                record["python_threads"] = [
                    {"id": thread.native_id, "name": thread.name}
                    for thread in threading.enumerate()
                ]
            record["handles"] = process.num_handles() if os.name == "nt" else process.num_fds()
            memory = process.memory_full_info()
            record.update(rss=memory.rss, uss=memory.uss, command=process.cmdline())
        except psutil.Error as error:
            record["error"] = repr(error)
    return record


def tree_snapshot(pid: int) -> list[dict[str, Any]]:
    try:
        process = psutil.Process(pid)
        return [
            process_snapshot(process),
            *(process_snapshot(p) for p in process.children(recursive=True)),
        ]
    except psutil.NoSuchProcess:
        return []


def process_group_snapshot(pid: int) -> list[dict[str, Any]]:
    if os.name == "nt":
        return tree_snapshot(pid)
    records = []
    for process in psutil.process_iter():
        with contextlib.suppress(ProcessLookupError, PermissionError, psutil.Error):
            if os.getpgid(process.pid) == pid:
                records.append(process_snapshot(process))
    return records


def living_processes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    living = []
    for record in records:
        with contextlib.suppress(psutil.NoSuchProcess):
            process = psutil.Process(record["pid"])
            if process.create_time() == record.get("created"):
                living.append(process_snapshot(process))
    return living


def terminate_processes(records: list[dict[str, Any]]) -> None:
    for record in reversed(records):
        with contextlib.suppress(psutil.Error):
            process = psutil.Process(record["pid"])
            if process.create_time() == record.get("created"):
                process.kill()


async def close_owned_executor(executor: JobExecutor, artifact_dir: Path) -> None:
    """Capture evidence before cleanup; a missed deadline remains a failure after emergency reaping."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    before = tree_snapshot(executor.pid) if executor.pid else []
    artifact = artifact_dir / f"executor-{executor.id}.json"
    artifact.write_text(json.dumps({"before": before}, indent=2))
    failure: BaseException | None = None
    try:
        async with asyncio.timeout(5):
            await executor.aclose()
            if executor.started:
                await executor.join()
    except BaseException as error:
        failure = error
        observed = {record["pid"]: record for record in before}
        if executor.pid:
            observed.update({record["pid"]: record for record in tree_snapshot(executor.pid)})
        before = list(observed.values())
        artifact.write_text(json.dumps({"failure": repr(error), "before": before}, indent=2))
        terminate_processes(before)
        if executor.started:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(executor.join(), timeout=5)
    finally:
        after: list[dict[str, Any]] = []
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
    known: dict[int, dict[str, Any]] = {}
    failure: BaseException | None = None
    with output.open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name != "nt",
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
            before: list[dict[str, Any]] = []
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
                if os.name != "nt":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
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
