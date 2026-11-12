"""Host fixtures: a local SQLite storage, the A2A peer, an A2A door served in process and a job
run the way the worker runs one."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from livekit.agents import AgentServer, _core
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import SQLiteStorage, Storage
from livekit.protocol import agent

from .a2a_door import PROJECT_ENV, serving
from .a2a_peer import peer  # noqa: F401  (re-exported so pytest discovers the fixture)
from .ipc import Entrypoint, job_runner, running_job

A2AHandler = Callable[[A2ASessionContext], Awaitable[object]]
RunJob = Callable[..., Awaitable[None]]


@pytest.fixture
async def thread_storage(tmp_path: Path) -> AsyncIterator[Storage]:
    storage = SQLiteStorage(
        directory=tmp_path / "storage",
        lease_ttl=timedelta(seconds=2),
        renew_interval=timedelta(seconds=0.5),
        holder="worker-a",
    )
    yield storage
    await storage.aclose()


@pytest.fixture
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A developer's exported LiveKit environment does not reach the in-process servers."""
    for name in PROJECT_ENV:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def no_livekit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def a2a_door(
    thread_storage: Storage, no_project_env: None
) -> Callable[[A2AHandler], AbstractAsyncContextManager[httpx.AsyncClient]]:
    """Serves a handler as the ``desk`` endpoint; the client's base URL is the endpoint's.

    async with a2a_door(handler) as door:
        response = await door.post("/v1", json=message_send("hello"))
    """

    @asynccontextmanager
    async def serve(handler: A2AHandler) -> AsyncIterator[httpx.AsyncClient]:
        server = AgentServer(
            storage=thread_storage,
            agent_name="front-desk",
            api_key="APItest",
            api_secret="secret-secret-secret-secret-secret",
        )
        server.a2a_session(endpoint="desk", description="Answers front desk questions.")(handler)
        async with serving(server) as base_url:
            async with httpx.AsyncClient(base_url=f"{base_url}/desk", timeout=30.0) as client:
                yield client

    return serve


@pytest.fixture
async def run_job() -> AsyncIterator[RunJob]:
    """Runs one job to its end in a job thread, as the worker does after a dispatch:
    ``await run_job(entrypoint, agent.Job(id="AJ_x"), session_end=, env=, fake_job=True)``."""
    runners: list[_core.JobSupervisor] = []

    async def run(
        entrypoint: Entrypoint,
        job: agent.Job,
        *,
        session_end: Entrypoint | None = None,
        env: dict[str, str] | None = None,
        fake_job: bool = True,
    ) -> None:
        runner = job_runner(entrypoint, session_end=session_end, env=env)
        runners.append(runner)
        await runner.start()
        await runner.initialize()
        await runner.launch_job(running_job(job, fake_job=fake_job))
        await runner.join()

    yield run
    for runner in runners:
        async with asyncio.timeout(10):
            await runner.aclose()
            await runner.join()
