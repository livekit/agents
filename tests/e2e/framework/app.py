"""Owned subprocess applications and public text clients, shared across feature stories."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from livekit.agents.a2a import AgentClient

from ..resources import STARTUP_SECONDS, Child, child_process, finish_cleanup
from ..sfu import KEY, SECRET

ROOT = Path(__file__).resolve().parents[3]

if TYPE_CHECKING:
    from .voice import VoiceClient


class RunningApp:
    """An ordinary SDK application process; restarting preserves its environment and data."""

    def __init__(self, module: str, directory: Path, env: dict[str, str], voice: bool) -> None:
        self.module, self.directory, self.env, self.voice = module, directory, env, voice
        self.agent_name = env["LIVEKIT_AGENT_NAME"]
        self.url = ""
        self.child: Child | None = None
        self._process: AbstractAsyncContextManager[Child] | None = None
        self._attempt = 0

    def endpoint(self, name: str) -> str:
        return f"{self.url}/{name}"

    async def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("application already started")
        self._attempt += 1
        marker = self.directory / f"ready-{self._attempt}.json"
        self._process = child_process(
            sys.executable,
            "-m",
            "tests.e2e.framework.runner",
            self.module,
            str(marker),
            "voice" if self.voice else "text",
            log=self.directory / f"app-{self._attempt}.log",
            env=self.env,
        )
        self.child = await self._process.__aenter__()
        async with asyncio.timeout(STARTUP_SECONDS):
            while not marker.exists():
                if self.child.process.returncode is not None:
                    raise RuntimeError(self.child.log.read_text())
                await asyncio.sleep(0.01)
        self.url = json.loads(marker.read_text())["url"]

    async def stop(self) -> None:
        if self._process is not None:
            process, self._process = self._process, None
            await process.__aexit__(None, None, None)

    async def kill(self) -> None:
        if self.child is not None:
            await self.child.kill()
        await self.stop()

    async def restart(self) -> None:
        await self.stop()
        await self.start()


class E2E:
    """Fixture facade over the existing subprocess, SFU and client infrastructure."""

    def __init__(self, directory: Path, sfu: Callable[[], str]) -> None:
        self.directory, self._sfu = directory, sfu
        self._resources = AsyncExitStack()
        self._apps = 0

    async def start_app(
        self, module: str, *, env: Mapping[str, str] | None = None, voice: bool = False
    ) -> RunningApp:
        self._apps += 1
        directory = self.directory / f"app-{self._apps}"
        directory.mkdir()
        child_env = dict(os.environ)
        # Each app uses private local credentials/data; exported project routing cannot affect it.
        for name in (
            "LIVEKIT_URL",
            "LIVEKIT_AGENT_NAME_OVERRIDE",
            "LIVEKIT_AGENT_DEPLOYMENT",
            "LIVEKIT_AGENT_VERSION",
        ):
            child_env.pop(name, None)
        child_env.update(
            LIVEKIT_API_KEY=KEY,
            LIVEKIT_API_SECRET=SECRET,
            LIVEKIT_AGENT_NAME="e2e-" + uuid.uuid4().hex,
            LIVEKIT_AGENT_THREADS_DIR=str(directory / "threads"),
            PYTHONPATH=os.pathsep.join((str(ROOT / "python/livekit-agents"), str(ROOT))),
        )
        if voice:
            child_env["LIVEKIT_URL"] = self._sfu()
        child_env.update(env or {})
        app = RunningApp(module, directory, child_env, voice)
        self._resources.push_async_callback(app.stop)
        await app.start()
        return app

    @asynccontextmanager
    async def text_client(
        self, app: RunningApp, endpoint: str = "desk"
    ) -> AsyncIterator[AgentClient]:
        # The local unregistered listener vouches for callers, just like the console listener.
        client = AgentClient(app.endpoint(endpoint))
        try:
            yield client
        finally:
            await client.aclose()

    def voice_client(self, app: RunningApp) -> AbstractAsyncContextManager[VoiceClient]:
        from .voice import connected_voice

        return connected_voice(app)

    async def aclose(self) -> None:
        await finish_cleanup(self._resources.aclose())
