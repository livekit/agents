"""A real ``AgentServer`` whose jobs run in the core's job processes, run as a script by
``resources.JobWorker``: unregistered, jobs simulated with ``fake_job``. Its entrypoint leaves
marker files (``<room>.started``, ``<room>.shutdown``) and log lines on stdout. Job processes
import the entrypoint from this file, as they import a user's agent file.

Driving it unregistered with simulated jobs uses the server's private hooks (``_run``,
``_simulate_job``): the same pool and IPC as a dispatched job, without a LiveKit server.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import time
from multiprocessing import current_process
from pathlib import Path

import psutil

from livekit.agents import AgentServer, JobContext, _core, cli

SCENARIO = os.environ.get("IT_SCENARIO", "")
MARKERS = Path(os.environ.get("IT_DIR", "."))

# a job process imports this file before it answers the initialize request: the first one to do
# so in the `slow_init` scenario takes longer than the server's initialize_process_timeout
if SCENARIO == "slow_init" and current_process().name == "job_proc":
    try:
        (MARKERS / "slow_init").open("x").close()
        time.sleep(60)
    except FileExistsError:
        pass


def _write(name: str, data: dict[str, object]) -> None:
    tmp = MARKERS / f".{name}"
    tmp.write_text(json.dumps(data))
    tmp.rename(MARKERS / name)


async def _entrypoint(ctx: JobContext) -> None:
    room = ctx.job.room.name
    info: dict[str, object] = {"pid": os.getpid(), "job_id": ctx.job.id}

    async def on_shutdown(reason: str) -> None:
        _write(f"{room}.shutdown", {**info, "reason": reason, **extra})

    ctx.add_shutdown_callback(on_shutdown)
    extra: dict[str, object] = {}
    _write(f"{room}.started", info)

    scenario = SCENARIO if room == "room-0" else "ok"
    if scenario == "crash":
        os._exit(7)
    elif scenario == "idle":
        await asyncio.sleep(600)
    elif scenario == "cli_shutdown":
        # A connected job normally returns its entrypoint and stays alive until shutdown.
        # The fake room has the same lifetime without requiring a LiveKit server.
        return
    elif scenario == "memory":
        hog = bytearray(800 * 1024 * 1024)
        for i in range(0, len(hog), 4096):
            hog[i] = 1
        await asyncio.sleep(600)
    elif scenario == "forkserver":
        parent = psutil.Process(os.getppid())
        extra.update(parent=parent.pid, parent_threads=parent.num_threads())

    logging.getLogger("it.job").info("it job says hello", extra={"it_field": room})
    print(f"it print from {room}", flush=True)
    ctx.shutdown("it done")


async def _drive(server: AgentServer, jobs: int) -> None:
    started = asyncio.Event()
    server._on_started.append(started.set)
    run = asyncio.create_task(server._run(devmode=False, unregistered=True))
    readiness = asyncio.create_task(started.wait())
    try:
        await asyncio.wait({run, readiness}, timeout=60, return_when=asyncio.FIRST_COMPLETED)
        if run.done():
            run.result()
        if not started.is_set():
            raise TimeoutError("the worker did not become ready")
        for i in range(jobs):
            room = f"room-{i}"
            await server._simulate_job(room, fake_job=True)
            # Start the next job only after the previous one's actual lifecycle settled.
            while True:
                began = (MARKERS / f"{room}.started").exists() or (
                    MARKERS / f"{room}.shutdown"
                ).exists()
                if began and not server._active_jobs:
                    break
                await asyncio.sleep(0.05)
    finally:
        readiness.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await readiness
        await server.aclose()
        await run


def _worker_main() -> None:
    from livekit.agents._generated.cli import CliArgsStart, CliParseResultParsed
    from livekit.agents._generated.worker_options import ProcessOptions
    from livekit.agents.job import JobExecutorType

    options = json.loads(os.environ["IT_OPTS"])
    if "job_executor_type" in options:
        options["job_executor_type"] = JobExecutorType(options["job_executor_type"])
    options.setdefault("num_idle_processes", 1)
    options.setdefault("multiprocessing_context", "spawn")
    options.setdefault("listeners", {"health": {"port": 0}})

    # the IPC options are private: set them where the CLI does
    tuning = {f.name for f in dataclasses.fields(ProcessOptions)}
    process = ProcessOptions(**{k: v for k, v in options.items() if k in tuning})
    server = AgentServer()
    server._update_options(process, **{k: v for k, v in options.items() if k not in tuning})
    server.rtc_session()(_entrypoint)
    if SCENARIO == "cli_shutdown":
        job_task: asyncio.Task[None] | None = None

        def start_job() -> None:
            nonlocal job_task
            job_task = asyncio.create_task(server._simulate_job("room-0", fake_job=True))

        server._on_started.append(start_job)
        parsed = _core.parse_cli(["agent", "start"])
        assert isinstance(parsed, CliParseResultParsed) and isinstance(parsed.args, CliArgsStart)
        raise SystemExit(cli._run_worker(server, parsed.args, unregistered=True))
    cli.setup_logging("debug")
    asyncio.run(_drive(server, int(os.environ["IT_JOBS"])))


if __name__ == "__main__":
    _worker_main()
