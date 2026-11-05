"""Job process start methods compared: ``forkserver`` against ``spawn``, each with the
framework's real preload (``ipc/_preload.py``, which the forkserver runs once and a spawned
process runs itself), as the worker starts its job processes.

For each method it measures:

- cold start: one process at a time, from ``start()`` to initialized (ready for a job);
- memory: ``--warm`` processes initialized side by side, their summed PSS (Linux) or USS
  (elsewhere, where psutil has no PSS; RSS where reading the pages is denied), the forkserver
  itself included;
- burst: ``--burst`` processes started at once, as jobs beyond the warm pool are, each one's
  time to initialized (median and max).

    python livekit-agents-next/scripts/proc_bench.py
    python livekit-agents-next/scripts/proc_bench.py --warm 20 --burst 10 --methods forkserver

The numbers are only representative on Linux, the platform the forkserver is the default on.
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import statistics
import time
from typing import Any

import psutil

from livekit.agents import _core
from livekit.agents.ipc import JobLauncher, JobRunnerOptions, configure_forkserver


async def _entrypoint(ctx: Any) -> None:
    ctx.shutdown("bench")


def _executor(method: str) -> _core.JobSupervisor:
    launcher = JobLauncher(
        job_entrypoint_fnc=_entrypoint,
        session_end_fnc=None,
        simulation_end_fnc=None,
        session_end_timeout=10,
        mp_ctx=mp.get_context(method),
        loop=asyncio.get_running_loop(),
    )
    options = JobRunnerOptions(initialize_timeout=120, close_timeout=10, high_ping_threshold=1)
    return _core.JobSupervisor(launcher, options)


async def _ready(method: str) -> tuple[_core.JobSupervisor, float]:
    proc = _executor(method)
    started = time.perf_counter()
    await proc.start()
    await proc.initialize()
    return proc, time.perf_counter() - started


def _footprint_mb(pids: list[int]) -> tuple[float, str]:
    """The summed footprint: PSS, else USS, else RSS (where reading another process's pages is
    denied, as on macOS without the task_for_pid entitlement)."""
    total, metric = 0.0, "rss"
    for pid in pids:
        try:
            info = psutil.Process(pid).memory_full_info()
            metric = "pss" if hasattr(info, "pss") else "uss"
            total += getattr(info, metric)
        except psutil.AccessDenied:
            total += psutil.Process(pid).memory_info().rss
    return total / (1024 * 1024), metric


def _forkserver_pid() -> list[int]:
    me = psutil.Process()
    return [c.pid for c in me.children() if "forkserver" in " ".join(c.cmdline())]


async def bench(method: str, cold: int, warm: int, burst: int) -> dict[str, Any]:
    if method == "forkserver":
        configure_forkserver(mp.get_context(method), [])
    # the first process also starts the forkserver: not counted
    first, _ = await _ready(method)
    await first.aclose()

    colds = []
    for _ in range(cold):
        proc, elapsed = await _ready(method)
        colds.append(elapsed)
        await proc.aclose()

    procs = [p for p, _ in await asyncio.gather(*(_ready(method) for _ in range(warm)))]
    pids = [p.pid for p in procs if p.pid] + (_forkserver_pid() if method == "forkserver" else [])
    memory, metric = _footprint_mb(pids)
    await asyncio.gather(*(p.aclose() for p in procs))

    burst_procs = await asyncio.gather(*(_ready(method) for _ in range(burst)))
    bursts = [t for _, t in burst_procs]
    await asyncio.gather(*(p.aclose() for p, _ in burst_procs))
    return {
        "cold start s (median)": round(statistics.median(colds), 3),
        f"memory MB ({metric}), {warm} warm": round(memory, 1),
        f"burst of {burst} s (median)": round(statistics.median(bursts), 3),
        f"burst of {burst} s (max)": round(max(bursts), 3),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["forkserver", "spawn"])
    parser.add_argument("--cold", type=int, default=5)
    parser.add_argument("--warm", type=int, default=10)
    parser.add_argument("--burst", type=int, default=5)
    args = parser.parse_args()
    for method in args.methods:
        print(method, await bench(method, args.cold, args.warm, args.burst), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
