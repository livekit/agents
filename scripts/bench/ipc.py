"""Measure and verify real IPC process churn through the production pool and supervisor.

Run with ``PYTHONPATH=livekit-agents-next:. uv run --no-sync python -m scripts.bench.ipc``.
Jobs use the framework's simulated room; no model, network service or credentials are needed.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

import psutil

from livekit.agents import JobContext, _core
from livekit.agents.ipc import JobExecutor, ProcPool
from livekit.agents.ipc.launch import configure_forkserver
from livekit.agents.job import JobAcceptArguments, JobExecutorType, JobProcess, RunningJobInfo
from livekit.protocol import agent
from scripts.bench.metrics import LoopLag, distribution

PoolEvent: TypeAlias = Literal[
    "process_created", "process_started", "process_ready", "process_closed", "process_job_launched"
]


def require(condition: bool, message: str) -> None:
    # Unlike assert, correctness checks must also run under python -O.
    if not condition:
        raise AssertionError(message)


def same_process_alive(pid: int, birth: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def child_arguments(process: JobProcess) -> tuple[Path, str]:
    value = process.user_arguments
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("directory"), str)
        or not isinstance(value.get("executor_id"), str)
    ):
        raise TypeError("IPC benchmark child arguments were corrupted")
    return Path(value["directory"]), value["executor_id"]


async def entrypoint(context: JobContext) -> None:
    """Acknowledge the actual payload, then wait for the parent's completion/kill race."""
    started_ns = time.monotonic_ns()
    directory, identity = child_arguments(context.proc)
    job_id = context.job.id
    require(job_id.startswith("AJ_bench_"), f"unexpected job identity: {job_id}")
    claim = directory / "starts" / job_id
    try:
        claim.mkdir()
    except FileExistsError:
        atomic_json(
            directory / "violations" / f"{job_id}-{os.getpid()}.json",
            {
                "error": "duplicate entrypoint",
                "job_id": job_id,
                "pid": os.getpid(),
            },
        )
        raise
    atomic_json(
        claim / "receipt.json",
        {
            "executor_id": identity,
            "pid": os.getpid(),
            "job_id": job_id,
            "metadata_sha256": hashlib.sha256(context.job.metadata.encode()).hexdigest(),
            "started_ns": started_ns,
        },
    )
    while not (directory / "release" / job_id).exists():
        await asyncio.sleep(0.005)
    context.shutdown("IPC benchmark completed")


class ObservedExecutor(JobExecutor):
    """Observe completion without replacing any process/supervisor behavior."""

    completions: int = 0

    async def _core_completed(self, completion: Any) -> None:
        await super()._core_completed(completion)
        self.completions += 1


@dataclass
class Ledger:
    directory: Path
    created: list[ObservedExecutor] = field(default_factory=list)
    events: Counter[tuple[str, str]] = field(default_factory=Counter)
    ready_ns: dict[str, int] = field(default_factory=dict)
    created_ns: dict[str, int] = field(default_factory=dict)
    births: dict[str, tuple[int, float]] = field(default_factory=dict)
    unready: set[str] = field(default_factory=set)
    live: set[str] = field(default_factory=set)
    peak_unready: int = 0
    peak_live: int = 0
    violations: list[str] = field(default_factory=list)
    submitted: dict[str, tuple[str, int]] = field(default_factory=dict)
    acknowledged: set[str] = field(default_factory=set)
    cancelled: set[str] = field(default_factory=set)
    forbidden: set[str] = field(default_factory=set)
    dispatch_ns: list[int] = field(default_factory=list)
    job_start_ns: list[int] = field(default_factory=list)

    def event(self, event: str, process: JobExecutor) -> None:
        identity = process.id
        self.events[event, identity] += 1
        if event == "process_started" and process.pid is not None:
            try:
                self.births[identity] = (process.pid, psutil.Process(process.pid).create_time())
            except psutil.NoSuchProcess:
                pass  # The OS may already have reaped a deliberately killed runner.
        if event == "process_created":
            self.unready.add(identity)
            self.live.add(identity)
            self.peak_unready = max(self.peak_unready, len(self.unready))
            self.peak_live = max(self.peak_live, len(self.live))
        if event == "process_ready":
            self.ready_ns[identity] = time.monotonic_ns()
            self.unready.discard(identity)
        if event == "process_closed":
            self.unready.discard(identity)
            self.live.discard(identity)
        # EventEmitter catches callback exceptions, so record errors for the final oracle.
        if self.events[event, identity] > 1:
            self.violations.append(f"{event} delivered twice for {identity}")
        if event != "process_closed" and self.events["process_closed", identity]:
            self.violations.append(f"{event} after close for {identity}")

    def make_executor(self, arguments: dict[str, Any]) -> ObservedExecutor:
        process = ObservedExecutor(**arguments)
        process.user_arguments = {"directory": str(self.directory), "executor_id": process.id}
        self.created_ns[process.id] = time.monotonic_ns()
        self.created.append(process)
        return process

    def verify(self) -> dict[str, int]:
        require(not self.violations, str(self.violations))
        require(not list((self.directory / "violations").iterdir()), "duplicate job execution")
        identities: set[str] = set()
        receipts = list((self.directory / "starts").glob("*/receipt.json"))
        started: set[str] = set()
        for path in receipts:
            receipt = json.loads(path.read_text())
            job_id, identity = receipt["job_id"], receipt["executor_id"]
            require(job_id in self.submitted, f"unsubmitted job executed: {job_id}")
            require(job_id not in self.forbidden, f"rejected job executed: {job_id}")
            require(identity not in identities, f"process executed two jobs: {identity}")
            identities.add(identity)
            started.add(job_id)
            digest, submitted_ns = self.submitted[job_id]
            require(receipt["metadata_sha256"] == digest, f"payload corruption: {job_id}")
            require(receipt["started_ns"] >= submitted_ns, f"invalid start time: {job_id}")
            process = next(p for p in self.created if p.id == identity)
            require(process.pid == receipt["pid"], f"wrong process received {job_id}")
        require(
            self.acknowledged <= started,
            f"acknowledged but never ran: {self.acknowledged - started}",
        )
        require(started <= self.acknowledged | self.cancelled, "unexpected job execution")
        for process in self.created:
            require(
                self.events["process_closed", process.id] == 1, f"process not retired: {process.id}"
            )
            require(
                process.completions == int(process.pid is not None),
                f"bad completion count: {process.id}",
            )
            if process.pid is not None:
                require(process.exitcode is not None, f"missing exit observation: {process.id}")
        for pid, birth in self.births.values():
            require(
                not same_process_alive(pid, birth), f"child still exists (including zombies): {pid}"
            )
        return {
            "executors_created": len(self.created),
            "processes_spawned": sum(p.pid is not None for p in self.created),
            "peak_owned_executors": self.peak_live,
            "peak_unready_executors": self.peak_unready,
            "forced_kills": sum(p.killed for p in self.created),
            "graceful_exits": sum(p.exitcode == 0 for p in self.created),
            "acknowledged_jobs": len(self.acknowledged),
            "cancelled_requests": len(self.cancelled),
            "cancelled_jobs_started": len(started & self.cancelled),
            "verified_entrypoints": len(started),
        }


class ObservedPool(ProcPool):
    def __init__(self, ledger: Ledger, **kwargs: Any) -> None:
        self.ledger = ledger
        super().__init__(**kwargs)
        events: tuple[PoolEvent, ...] = (
            "process_created",
            "process_started",
            "process_ready",
            "process_closed",
            "process_job_launched",
        )
        for event in events:
            self.on(event, lambda process, event=event: ledger.event(event, process))

    def _core_create(self) -> JobExecutor:
        return self.ledger.make_executor(self._executor_args)


async def eventually(predicate: Callable[[], bool], timeout: float, description: str) -> None:
    try:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)
    except TimeoutError as error:
        raise AssertionError(description) from error


def options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "job_entrypoint_fnc": entrypoint,
        "session_end_fnc": None,
        "simulation_end_fnc": None,
        "initialize_timeout": args.timeout,
        "close_timeout": 2.0,
        "session_end_timeout": 1.0,
        "job_executor_type": JobExecutorType.PROCESS,
        "mp_ctx": mp.get_context(args.start_method),
        "memory_warn_mb": 0,
        "memory_limit_mb": 0,
        "http_proxy": None,
        "env": [("LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE", "0")],
        "loop": asyncio.get_running_loop(),
    }


def new_ledger(directory: Path) -> Ledger:
    for name in ["starts", "release", "violations"]:
        (directory / name).mkdir(parents=True)
    return Ledger(directory)


def job(ledger: Ledger, args: argparse.Namespace, label: str) -> RunningJobInfo:
    job_id = f"AJ_bench_{args.seed}_{label}"
    payload = hashlib.sha256(job_id.encode()).hexdigest()
    metadata = json.dumps(
        {
            "nonce": job_id,
            "payload": (payload * math.ceil(args.payload_bytes / 64))[: args.payload_bytes],
        }
    )
    ledger.submitted[job_id] = (hashlib.sha256(metadata.encode()).hexdigest(), time.monotonic_ns())
    return RunningJobInfo(
        accept_arguments=JobAcceptArguments(name="ipc-bench", identity=job_id, metadata=""),
        job=agent.Job(id=job_id, metadata=metadata),
        url="",
        token="",
        worker_id="ipc-bench",
        fake_job=True,
    )


async def accepted(
    pool: ProcPool, ledger: Ledger, info: RunningJobInfo, timeout: float
) -> JobExecutor:
    started = time.monotonic_ns()
    await asyncio.wait_for(pool.launch_job(info), timeout)
    ledger.dispatch_ns.append(time.monotonic_ns() - started)
    ledger.acknowledged.add(info.job.id)
    path = ledger.directory / "starts" / info.job.id / "receipt.json"
    await eventually(path.exists, timeout, f"entrypoint missing: {info.job.id}")
    receipt = json.loads(path.read_text())
    ledger.job_start_ns.append(receipt["started_ns"] - ledger.submitted[info.job.id][1])
    process = pool.get_by_job_id(info.job.id)
    if process is None:
        raise AssertionError(f"acknowledged job has no owner: {info.job.id}")
    return process


async def finish(
    ledger: Ledger, info: RunningJobInfo, process: JobExecutor, timeout: float
) -> None:
    (ledger.directory / "release" / info.job.id).touch()
    await asyncio.wait_for(process.join(), timeout)
    await eventually(lambda: process.exitcode is not None, timeout, "completion missing after join")
    require(process.exitcode == 0, f"normal job failed: {info.job.id}, exit={process.exitcode}")


def idle(pool: ProcPool, ledger: Ledger) -> list[JobExecutor]:
    return [p for p in pool.processes if p.id in ledger.ready_ns and p.running_job is None]


async def cleanup(pool: ProcPool, ledger: Ledger, timeout: float) -> None:
    errors: list[BaseException] = []
    try:
        await asyncio.wait_for(pool.aclose(), timeout)
    except Exception as error:
        errors.append(error)
    finally:
        # Also join processes removed from pool.processes, including abandoned launch callers.
        async def stop(process: JobExecutor) -> None:
            if process.started:
                try:
                    await asyncio.wait_for(process.kill(), timeout)
                finally:
                    await asyncio.wait_for(process.join(), timeout)

        results = await asyncio.gather(*(stop(p) for p in ledger.created), return_exceptions=True)
        errors.extend(error for error in results if isinstance(error, BaseException))
        try:
            await eventually(
                lambda: (
                    not pool.processes
                    and all(
                        (p.pid is None or p.completions == 1)
                        and ledger.events["process_closed", p.id] == 1
                        for p in ledger.created
                    )
                ),
                timeout,
                "process retirement/completion callbacks did not settle",
            )
        except Exception as error:
            errors.append(error)
    if errors:
        raise BaseExceptionGroup("IPC benchmark cleanup failures", errors)


async def run_round(directory: Path, args: argparse.Namespace, index: int) -> dict[str, Any]:
    ledger = new_ledger(directory)
    pool = ObservedPool(ledger, num_idle_processes=args.warm, **options(args))
    tasks: list[asyncio.Task[Any]] = []
    begun = time.monotonic_ns()
    try:
        await asyncio.wait_for(pool.start(), args.timeout)
        await eventually(
            lambda: len(idle(pool, ledger)) >= args.warm, args.timeout, "warm pool unavailable"
        )
        warmup_ns = time.monotonic_ns() - begun
        warm_pids = {p.pid for p in idle(pool, ledger)}
        warm_jobs = [job(ledger, args, f"{index}_warm_{i}") for i in range(args.warm)]
        processes = await asyncio.gather(
            *(accepted(pool, ledger, info, args.timeout) for info in warm_jobs)
        )
        require({p.pid for p in processes} <= warm_pids, "warm dispatch spawned a cold process")
        warm_dispatch = ledger.dispatch_ns.copy()
        # The first job remains alive: duplicate admission must not run its entrypoint twice.
        try:
            await pool.launch_job(warm_jobs[0])
        except RuntimeError as error:
            require("already admitted" in str(error), f"wrong duplicate error: {error}")
        else:
            raise AssertionError("duplicate job admitted")
        await asyncio.gather(
            *(
                finish(ledger, info, process, args.timeout)
                for info, process in zip(warm_jobs, processes, strict=True)
            )
        )
        replenish = time.monotonic_ns()
        await eventually(
            lambda: len(idle(pool, ledger)) >= args.warm,
            args.timeout,
            "warm pool did not replenish",
        )
        replenishment_ns = time.monotonic_ns() - replenish
        semaphore = asyncio.Semaphore(args.concurrency)

        async def churn(number: int) -> None:
            async with semaphore:
                info = job(ledger, args, f"{index}_churn_{number}")
                action = (number + args.seed) % 4
                if action == 0:
                    launch = asyncio.create_task(pool.launch_job(info))
                    tasks.append(launch)
                    await asyncio.sleep(0)
                    if launch.cancel():
                        with contextlib.suppress(asyncio.CancelledError):
                            await launch
                        ledger.cancelled.add(info.job.id)
                        return
                    await launch
                    ledger.acknowledged.add(info.job.id)
                    path = ledger.directory / "starts" / info.job.id / "receipt.json"
                    await eventually(
                        path.exists, args.timeout, f"late entrypoint missing: {info.job.id}"
                    )
                    process = pool.get_by_job_id(info.job.id)
                    if process is None:
                        raise AssertionError("completed launch lost its process")
                else:
                    process = await accepted(pool, ledger, info, args.timeout)
                if action == 1:
                    await asyncio.wait_for(
                        asyncio.gather(
                            process.kill(), process.aclose(), process.kill(), process.join()
                        ),
                        args.timeout,
                    )
                elif action == 2:
                    (ledger.directory / "release" / info.job.id).touch()
                    await asyncio.wait_for(
                        asyncio.gather(process.aclose(), process.join(), process.aclose()),
                        args.timeout,
                    )
                else:
                    await finish(ledger, info, process, args.timeout)

        churn_started = time.monotonic_ns()
        churn_tasks = [asyncio.create_task(churn(i)) for i in range(args.jobs)]
        tasks.extend(churn_tasks)
        await asyncio.gather(*churn_tasks)
        churn_ns = time.monotonic_ns() - churn_started
        await eventually(
            lambda: len(idle(pool, ledger)) >= args.warm and pool._jobs_waiting_for_process == 0,
            args.timeout,
            "pool failed to recover after cancellation and kill races",
        )
        held = job(ledger, args, f"{index}_drain")
        await accepted(pool, ledger, held, args.timeout)
        drain = asyncio.create_task(pool.join_jobs())
        tasks.append(drain)
        await asyncio.sleep(0.02)
        require(not drain.done(), "drain finished while a job was still running")
        refused = job(ledger, args, f"{index}_refused")
        ledger.forbidden.add(refused.job.id)
        try:
            await pool.launch_job(refused)
        except RuntimeError as error:
            require("draining" in str(error), f"wrong drain refusal: {error}")
        else:
            raise AssertionError("job admitted during drain")
        close_started = time.monotonic_ns()
        abandoned = asyncio.create_task(pool.aclose())
        tasks.append(abandoned)
        await asyncio.sleep(0)
        abandoned.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await abandoned
        await asyncio.wait_for(asyncio.gather(pool.aclose(), pool.aclose(), drain), args.timeout)
        shutdown_ns = time.monotonic_ns() - close_started
        # Verify before emergency cleanup: fallback kill/join must never make a broken pool
        # look correct. Notification delivery may follow the native close acknowledgement.
        require(not pool.processes, "pool close returned with retained processes")
        require(pool._jobs_waiting_for_process == 0, "cancelled requests retained admission")
        await eventually(
            lambda: all(
                (p.pid is None or p.completions == 1) and ledger.events["process_closed", p.id] == 1
                for p in ledger.created
            ),
            args.timeout,
            "completion/close notifications missing after pool shutdown",
        )
        verified = ledger.verify()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await cleanup(pool, ledger, args.timeout)
    require(not pool.processes, "pool retained processes after close")
    require(pool._jobs_waiting_for_process == 0, "cancelled requests retained admission")
    return {
        **verified,
        "warmup_ms": warmup_ns / 1e6,
        "process_create_to_ready": asdict(
            distribution(
                [ready - ledger.created_ns[identity] for identity, ready in ledger.ready_ns.items()]
            )
        ),
        "warm_dispatch": asdict(distribution(warm_dispatch)),
        "dispatch_ack": asdict(distribution(ledger.dispatch_ns)),
        "entrypoint_start": asdict(distribution(ledger.job_start_ns)),
        "replenishment_ms": replenishment_ns / 1e6,
        "churn_requests_per_second": args.jobs / (churn_ns / 1e9),
        "shutdown_ms": shutdown_ns / 1e6,
    }


def descriptors() -> int:
    process = psutil.Process()
    if sys.platform == "win32":
        return process.num_handles()
    return process.num_fds()


async def benchmark(directory: Path, args: argparse.Namespace) -> dict[str, Any]:
    _core.init(_core.Role.Worker)
    os.environ["LIVEKIT_AGENTS_PRELOAD_LOCAL_INFERENCE"] = "0"
    configure_forkserver(mp.get_context(args.start_method), ())
    # Exercise the whole path before the resource baseline: native runtime, forkserver/resource
    # tracker and logging infrastructure are process-owned, not per-round leaks.
    print("IPC benchmark: untimed calibration", file=sys.stderr, flush=True)
    await run_round(directory / "calibration", args, -1)
    gc.collect()
    await asyncio.sleep(0.1)
    baseline = descriptors()
    baseline_children = {
        (p.pid, p.create_time()) for p in psutil.Process().children(recursive=True)
    }
    # A calibration leak must not become an allowed descendant in the baseline. Only the
    # multiprocessing helpers explicitly owned by this interpreter may remain alive.
    helper_pids: set[int | None] = set()
    if os.name != "nt":
        from multiprocessing import resource_tracker

        helper_pids.add(getattr(resource_tracker._resource_tracker, "_pid", None))
    if args.start_method == "forkserver":
        from multiprocessing import forkserver

        helper_pids.add(getattr(forkserver._forkserver, "_forkserver_pid", None))
    require(all(pid in helper_pids for pid, _ in baseline_children), "calibration leaked a child")
    baseline_tasks = asyncio.all_tasks()
    lag = LoopLag()
    heartbeat = asyncio.create_task(lag.run())
    await lag.ready.wait()
    rounds = []
    try:
        for index in range(args.rounds):
            print(f"IPC benchmark: round {index + 1}/{args.rounds}", file=sys.stderr, flush=True)
            async with asyncio.timeout(
                args.timeout * (4 + math.ceil(args.jobs / args.concurrency))
            ):
                rounds.append(await run_round(directory / f"round-{index}", args, index))
            gc.collect()
            await asyncio.sleep(0.1)
            require(descriptors() <= baseline + args.max_fd_growth, "parent descriptor/handle leak")
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
    await eventually(
        lambda: not (asyncio.all_tasks() - baseline_tasks),
        args.timeout,
        "asyncio task leak",
    )
    children = {(p.pid, p.create_time()) for p in psutil.Process().children(recursive=True)}
    require(
        children <= baseline_children,
        f"unexpected surviving descendants: {children - baseline_children}",
    )
    return {
        "correct": True,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "start_method": args.start_method,
        "seed": args.seed,
        "warm_processes": args.warm,
        "concurrency": args.concurrency,
        "jobs_per_round": args.jobs,
        "payload_bytes": args.payload_bytes,
        "local_model_preload": False,
        "parent_handles_before": baseline,
        "parent_handles_after": descriptors(),
        "loop_lag": asdict(distribution(lag.samples)),
        "rounds": rounds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--warm", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--payload-bytes", type=int, default=1024)
    parser.add_argument("--start-method", choices=["spawn", "forkserver"], default="spawn")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-fd-growth", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        min(args.rounds, args.warm, args.concurrency) < 1
        or args.jobs < 4
        or args.warm > 32
        or args.concurrency > 32
    ):
        parser.error(
            "rounds must be positive, jobs at least 4; warm/concurrency must be between 1 and 32"
        )
    if (
        not 0 <= args.payload_bytes <= 1024 * 1024
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.max_fd_growth < 0
    ):
        parser.error("invalid payload size, timeout or descriptor allowance")
    if args.start_method == "forkserver" and sys.platform != "linux":
        parser.error("the production forkserver executor is supported on Linux only")
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("livekit.agents").setLevel(logging.ERROR)
    directory = Path(tempfile.mkdtemp(prefix="lk-ipc-bench-"))
    try:
        result = asyncio.run(benchmark(directory, args))
        if args.output:
            atomic_json(args.output, result)
        print(json.dumps(result, indent=2))
    except BaseException:
        print(f"IPC benchmark failed; child receipts retained at {directory}", file=sys.stderr)
        raise
    shutil.rmtree(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
