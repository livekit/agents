"""How fast an agent is ready when `lk agent dev`, `lk agent start` or `lk agent console`
launches it.

Run with ``make bench-startup``.

Each launch is the real ``python main.py ...`` process, talking to local stand-ins for LiveKit
and for the console. Launches are cold: Python finds no bytecode cache and writes none, so the
stdlib, the framework and the app compile from source every time. ``--warm`` uses the cache,
as a dev reload does.

  interpreter       python -c pass
  framework import  import livekit.agents
  app import        the app module: its agents, plugins and models
  <mode> ready      launch to registered (dev, start), or to the console connection (console)
  <mode> stop       SIGTERM to exit; a dev reload pays it before every launch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aiohttp import WSMsgType, web

from livekit.protocol import agent

MODES = ("dev", "start", "console")
TIMEOUT = 30.0

IMPORTS = """
import json, time

started = time.perf_counter_ns()
import livekit.agents
framework = time.perf_counter_ns()
import {module}
done = time.perf_counter_ns()

print(json.dumps([framework - started, done - framework]))
"""


class StandIns:
    """LiveKit and the console, served for the whole bench."""

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/agent", self._worker)
        self._livekit = web.AppRunner(app, handle_signals=False)
        await self._livekit.setup()
        await web.TCPSite(self._livekit, "127.0.0.1", 0).start()

        self._console = await asyncio.start_server(self._console_call, "127.0.0.1", 0)

    async def close(self) -> None:
        await self._livekit.cleanup()
        self._console.close()

    def command(self, mode: str) -> tuple[list[str], dict[str, str]]:
        """The arguments and environment that launch `mode` against the stand-ins."""
        if mode == "console":
            port = self._console.sockets[0].getsockname()[1]
            return ["console", "--connect-addr", f"127.0.0.1:{port}"], {}

        port = self._livekit.addresses[0][1]
        argv = ["start", "--dev"] if mode == "dev" else ["start"]
        return argv, {"LIVEKIT_URL": f"ws://127.0.0.1:{port}"}

    def expect(self) -> asyncio.Future[int]:
        """Resolves with the time the next agent is ready."""
        self._ready: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        return self._ready

    def _mark_ready(self) -> None:
        if not self._ready.done():
            self._ready.set_result(time.monotonic_ns())

    async def _worker(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        async for frame in ws:
            if frame.type != WSMsgType.BINARY:
                continue

            message = agent.WorkerMessage()
            message.ParseFromString(frame.data)

            if message.HasField("register"):
                self._mark_ready()
                registered = agent.RegisterWorkerResponse(worker_id="AW_bench")
                await ws.send_bytes(agent.ServerMessage(register=registered).SerializeToString())

        return ws

    async def _console_call(self, reader: asyncio.StreamReader, _: asyncio.StreamWriter) -> None:
        self._mark_ready()
        while await reader.read(65536):
            pass


def environment(warm: bool, **extra: str) -> dict[str, str]:
    env = os.environ | {
        "PYTHONUNBUFFERED": "1",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
    }
    if not warm:
        env |= {"PYTHONPYCACHEPREFIX": os.devnull, "PYTHONDONTWRITEBYTECODE": "1"}
    return env | extra


def time_interpreter(warm: bool) -> int:
    started = time.monotonic_ns()
    subprocess.run([sys.executable, "-c", "pass"], env=environment(warm), check=True)
    return time.monotonic_ns() - started


def time_imports(app: Path, warm: bool) -> tuple[int, int]:
    """(framework, app module), timed inside one interpreter."""
    probe = subprocess.run(
        [sys.executable, "-c", IMPORTS.format(module=app.stem)],
        cwd=app.parent,
        env=environment(warm),
        capture_output=True,
        text=True,
        check=True,
    )
    framework, module = json.loads(probe.stdout)
    return framework, module


async def time_launch(app: Path, mode: str, warm: bool, stand_ins: StandIns) -> tuple[int, int]:
    """(launch to ready, SIGTERM to exit)."""
    argv, extra = stand_ins.command(mode)
    ready = stand_ins.expect()

    with tempfile.TemporaryFile() as log:
        started = time.monotonic_ns()
        process = await asyncio.create_subprocess_exec(
            sys.executable, app.name, *argv,
            cwd=app.parent, env=environment(warm, **extra), stdout=log, stderr=log,
        )  # fmt: skip

        try:
            exited = asyncio.ensure_future(process.wait())
            await asyncio.wait({ready, exited}, timeout=TIMEOUT, return_when="FIRST_COMPLETED")
            exited.cancel()
            if not ready.done():
                raise RuntimeError(f"{mode}: the agent exited or was not ready in {TIMEOUT}s")

            stopping = time.monotonic_ns()
            process.terminate()
            await asyncio.wait_for(process.wait(), TIMEOUT)

            return ready.result() - started, time.monotonic_ns() - stopping

        except BaseException:
            log.seek(0)
            print(log.read().decode()[-4000:], file=sys.stderr)
            raise

        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


async def bench(app: Path, modes: list[str], runs: int, warm: bool) -> dict[str, list[int]]:
    """Nanoseconds per measurement.

    Each round measures everything once, so machine load drifts across all of them alike.
    A warm bench discards its first round, which fills the cache the others reuse.
    """
    stand_ins = StandIns()
    await stand_ins.start()

    samples: dict[str, list[int]] = {}
    rounds = runs + 1 if warm else runs

    try:
        for index in range(rounds):
            print(f"startup bench: round {index + 1}/{rounds}", file=sys.stderr)

            framework, module = time_imports(app, warm)
            measured = {
                "interpreter": time_interpreter(warm),
                "framework import": framework,
                "app import": module,
            }

            for mode in modes:
                measured[f"{mode} ready"], measured[f"{mode} stop"] = await time_launch(
                    app, mode, warm, stand_ins
                )

            if warm and index == 0:
                continue

            for name, value in measured.items():
                samples.setdefault(name, []).append(value)

    finally:
        await stand_ins.close()

    return samples


def report(app: Path, samples: dict[str, list[int]], warm: bool) -> None:
    cache = "warm" if warm else "cold"
    print(f"{app} (python {sys.version.split()[0]}, {cache} bytecode cache)")
    print(f"  {'':<20}{'median':>9}{'min':>9}{'max':>9}  ms")

    for name, ns in samples.items():
        ms = [value / 1e6 for value in ns]
        print(f"  {name:<20}{statistics.median(ms):>9.1f}{min(ms):>9.1f}{max(ms):>9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--app",
        type=Path,
        default=Path(__file__).parent / "startup_app" / "main.py",
        help="the app's main.py",
    )
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warm", action="store_true", help="use the bytecode cache")
    parser.add_argument("--json", action="store_true", help="print the samples as JSON")
    args = parser.parse_args()

    app = args.app.resolve()
    samples = asyncio.run(bench(app, args.modes, args.runs, args.warm))

    if args.json:
        print(json.dumps(samples, indent=2))
    else:
        report(app, samples, args.warm)


if __name__ == "__main__":
    main()
