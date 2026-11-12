"""Ownership of test processes, in-process servers and the recording service."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

import httpx
import psutil
import pytest
from aiohttp import web
from pydantic import BaseModel

from livekit.agents import AgentServer

T = TypeVar("T")
STARTUP_SECONDS = 30.0
REQUEST_SECONDS = 30.0
CLOSE_SECONDS = 5.0


async def finish_cleanup(work: Awaitable[T]) -> T:
    """Settle owned cleanup even when its caller is cancelled again."""
    task = asyncio.ensure_future(work)
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


@dataclass
class Child:
    process: asyncio.subprocess.Process
    log: Path
    reaped: bool = False

    async def stop(self) -> None:
        # Kill the group even if the leader exited with descendants still running.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            async with asyncio.timeout(CLOSE_SECONDS):
                await self.process.wait()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            async with asyncio.timeout(CLOSE_SECONDS):
                await self.process.wait()
        finally:
            # macOS answers EPERM for a group whose leader is already reaped
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGKILL)
        self.reaped = True

    async def ready(self, url: str, *, timeout: float = STARTUP_SECONDS) -> None:
        async with httpx.AsyncClient(timeout=1.0, trust_env=False) as http:
            async with asyncio.timeout(timeout):
                while True:
                    if self.process.returncode is not None:
                        raise RuntimeError(
                            f"worker exited {self.process.returncode}; log: {self.log}\n"
                            + self.log.read_text()[-8000:]
                        )
                    try:
                        response = await http.get(url)
                    except httpx.TransportError:
                        pass
                    else:
                        response.raise_for_status()
                        return
                    await asyncio.sleep(0.02)


@asynccontextmanager
async def child_process(
    *command: str, log: Path, env: dict[str, str] | None = None, pass_fds: tuple[int, ...] = ()
) -> AsyncIterator[Child]:
    with log.open("wb") as output:
        spawning = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                stdout=output,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
                pass_fds=pass_fds,
            )
        )
        child: Child | None = None
        try:
            child = Child(await asyncio.shield(spawning), log)
            yield child
        finally:

            async def stop() -> None:
                owned = child or Child(await spawning, log)
                await owned.stop()

            await finish_cleanup(stop())


class ServiceCall(BaseModel):
    worker: str
    thread: str
    operation: str
    terms: str = ""


@dataclass
class RecordingService:
    """An application service; its calls are business evidence, independent of agent logs."""

    url: str = ""
    calls: list[ServiceCall] = field(default_factory=list)
    entered: dict[tuple[str, str], asyncio.Event] = field(default_factory=dict)
    releases: dict[str, asyncio.Event] = field(default_factory=dict)

    async def record(self, request: web.Request) -> web.Response:
        call = ServiceCall.model_validate(await request.json())
        self.calls.append(call)
        self.entered.setdefault((call.thread, call.operation), asyncio.Event()).set()
        if call.operation == "wait":
            async with asyncio.timeout(REQUEST_SECONDS):
                await self.releases.setdefault(call.thread, asyncio.Event()).wait()
        return web.Response(text="booked " + call.terms if call.operation == "book" else "found")

    async def wait_for_call(self, thread: str, operation: str = "turn") -> None:
        async with asyncio.timeout(REQUEST_SECONDS):
            await self.entered.setdefault((thread, operation), asyncio.Event()).wait()

    def release(self, thread: str) -> None:
        self.releases.setdefault(thread, asyncio.Event()).set()


@asynccontextmanager
async def recording_service() -> AsyncIterator[RecordingService]:
    service = RecordingService()
    app = web.Application()
    app.router.add_post("/call", service.record)
    runner = web.AppRunner(app, shutdown_timeout=CLOSE_SECONDS)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        service.url = f"http://{host}:{port}"
        yield service
    finally:
        for event in service.releases.values():
            event.set()
        await finish_cleanup(runner.cleanup())


@asynccontextmanager
async def serving(server: AgentServer) -> AsyncIterator[str]:
    """``server`` served in process as ``lk agent console`` serves it: unregistered, its routes
    and text doors on its own loopback listener, which vouches for every caller. Yields the
    base URL ``http://127.0.0.1:<port>/agents/<agent>/<deployment>``."""
    server._update_options(listeners={"health": {"port": 0}})
    run = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        url = None
        for _ in range(200):
            with contextlib.suppress(RuntimeError):
                url = server._config.delegation_url("").rstrip("/")
                break
            if run.done():
                run.result()
            await asyncio.sleep(0.05)
        if url is None:
            raise TimeoutError("the public listener did not start")
        yield url
    finally:
        await server.aclose()
        await run


JOB_WORKER = Path(__file__).with_name("job_worker.py")


class StartJobWorker(Protocol):
    """The ``job_worker`` fixture: ``job_worker("crash", jobs=2, **server_options)``."""

    def __call__(
        self, scenario: str, jobs: int = 1, *, module_launch: bool = False, **options: object
    ) -> JobWorker: ...


class JobWorker:
    """``job_worker.py`` run in a subprocess, its stdout collected as JSON log lines."""

    def __init__(
        self,
        tmp_path: Path,
        scenario: str,
        jobs: int,
        *,
        module_launch: bool = False,
        **options: object,
    ) -> None:
        self.dir = tmp_path
        env = {
            **os.environ,
            "IT_SCENARIO": scenario,
            "IT_DIR": str(tmp_path),
            "IT_JOBS": str(jobs),
            "IT_OPTS": json.dumps(options),
            "PYTHONUNBUFFERED": "1",
        }
        self.out = (tmp_path / "stdout").open("w+")
        command = (
            [sys.executable, "-m", "livekit.agents", "start", str(JOB_WORKER)]
            if module_launch
            else [sys.executable, str(JOB_WORKER)]
        )
        self.proc = subprocess.Popen(
            command,
            env=env,
            stdout=self.out,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).parents[2]),
            start_new_session=True,
        )
        self.owned: dict[int, float] = {}

    def lines(self) -> list[dict[str, object]]:
        """The worker's stdout, one JSON log record per line (``{"raw": line}`` otherwise)."""
        lines: list[dict[str, object]] = []
        for line in (self.dir / "stdout").read_text().splitlines():
            try:
                lines.append(json.loads(line))
            except ValueError:
                lines.append({"raw": line})
        return lines

    def logged(self, message: str) -> list[dict[str, object]]:
        return [rec for rec in self.lines() if message in str(rec.get("message", rec.get("raw")))]

    def marker(self, name: str, timeout: float = 30.0) -> str:
        path = self.dir / name
        deadline = time.monotonic() + timeout
        while not path.exists():
            self.snapshot()
            if self.proc.poll() is not None:
                pytest.fail(f"worker exited ({self.proc.returncode}) before {name}:\n{self.tail()}")
            if time.monotonic() > deadline:
                pytest.fail(f"no {name} after {timeout}s:\n{self.tail()}")
            time.sleep(0.05)
        return path.read_text()

    def wait(self, timeout: float = 60.0) -> int:
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            pytest.fail(f"worker still running after {timeout}s:\n{self.tail()}")

    def tail(self) -> str:
        return "\n".join(json.dumps(x) for x in self.lines()[-40:])

    def snapshot(self) -> None:
        """Remember every process in the worker's tree, so close can prove each was reaped."""
        try:
            root = psutil.Process(self.proc.pid)
            for process in [root, *root.children(recursive=True)]:
                with contextlib.suppress(psutil.NoSuchProcess):
                    self.owned[process.pid] = process.create_time()
        except psutil.NoSuchProcess:
            pass

    def close(self) -> None:
        self.snapshot()
        artifact = self.dir / "process-cleanup.json"
        # The session includes forkserver/tracker and descendants even after logical-parent death.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait(timeout=5)
        deadline = time.monotonic() + 10
        retained: list[dict[str, object]] = []
        while time.monotonic() < deadline:
            retained = []
            for pid, created in self.owned.items():
                with contextlib.suppress(psutil.NoSuchProcess):
                    process = psutil.Process(pid)
                    if process.create_time() == created:
                        retained.append({"pid": pid, "status": process.status()})
            if not retained:
                break
            time.sleep(0.02)
        artifact.write_text(json.dumps({"known": self.owned, "after": retained}, indent=2))
        self.out.close()
        assert not retained, f"owned processes remain (zombies are not reaped): {artifact}"


def gone(pid: int, timeout: float) -> bool:
    """``pid`` exited and was reaped: a zombie still holds a process-table entry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            psutil.Process(pid).status()
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.1)
    return False
