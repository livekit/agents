"""Ownership of test processes, recording services and real RTC peers."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

import httpx
from aiohttp import web
from pydantic import BaseModel

if TYPE_CHECKING:
    from livekit import rtc

T = TypeVar("T")
STARTUP_SECONDS = 30.0
REQUEST_SECONDS = 30.0
CLOSE_SECONDS = 5.0
DEBOUNCE_SECONDS = 0.5


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
        with contextlib.suppress(ProcessLookupError):
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
            with contextlib.suppress(ProcessLookupError):
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


@dataclass(frozen=True)
class AudioReceived:
    at: float
    samples: int


class Caller:
    """A real receiving peer; timestamps are at the caller, never the worker journal."""

    def __init__(self) -> None:
        from livekit import rtc

        self.room = rtc.Room()
        self.audio: list[AudioReceived] = []
        self.readers: list[asyncio.Task[None]] = []
        self.closing = False

        @self.room.on("track_subscribed")
        def subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if not self.closing and track.kind == rtc.TrackKind.KIND_AUDIO:
                self.readers.append(asyncio.create_task(self._receive(track)))

    async def _receive(self, track: rtc.Track) -> None:
        from livekit import rtc

        stream = rtc.AudioStream(track)
        try:
            async for event in stream:
                self.audio.append(AudioReceived(time.monotonic(), event.frame.samples_per_channel))
        finally:
            await stream.aclose()

    async def close(self) -> None:
        self.closing = True
        try:
            async with asyncio.timeout(CLOSE_SECONDS):
                await self.room.disconnect()
        finally:
            for task in self.readers:
                task.cancel()
            outcomes = await asyncio.gather(*self.readers, return_exceptions=True)
            for outcome in outcomes:
                if isinstance(outcome, BaseException) and not isinstance(
                    outcome, asyncio.CancelledError
                ):
                    raise outcome
