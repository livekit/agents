"""Fixture-owned workers, services, threads, storages and job runners. Cleanup runs on failure
and cancellation and reaps every child. A missing live prerequisite fails the run."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest

from livekit import rtc
from livekit.agents import _core
from livekit.agents.storage import SQLiteStorage, Storage

from .drivers import A2ACaller, Workers
from .processes import close_owned_executor
from .resources import (
    REQUEST_SECONDS,
    Child,
    JobWorker,
    RecordingService,
    StartJobWorker,
    child_process,
    finish_cleanup,
    recording_service,
)
from .sfu import (  # noqa: F401  (fixtures for the room modules)
    collect_closed_rooms,
    connected,
    sfu,
)
from .storages import LOCAL_LEASE_TTL, agentstore_configured, project_storage
from .v1_clients import compat_worker  # noqa: F401  (fixture for the client compat modules)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def live_credentials() -> tuple[str, str, str]:
    names = ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        pytest.fail("selected E2E requires " + ", ".join(missing))
    return os.environ[names[0]], os.environ[names[1]], os.environ[names[2]]


@pytest.fixture
async def service() -> AsyncIterator[RecordingService]:
    async with recording_service() as recording:
        yield recording


@pytest.fixture
async def workers(
    service: RecordingService, live_credentials: tuple[str, str, str], tmp_path: Path
) -> AsyncIterator[Workers]:
    """Two real worker processes (``text_worker.py``) on their own loopback listeners."""
    from livekit.agents.a2a import Credentials

    _, key, secret = live_credentials
    token = Credentials(key, secret).project_token()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "livekit-agents"), str(ROOT)))
    stack = AsyncExitStack()
    try:
        children: list[Child] = []
        callers: list[A2ACaller] = []
        for name in ("worker-a", "worker-b"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            url = f"http://127.0.0.1:{port}/agents/e2e/_"
            child = await stack.enter_async_context(
                child_process(
                    sys.executable,
                    "-m",
                    "tests_new.e2e.text_worker",
                    "--port",
                    str(port),
                    "--worker",
                    name,
                    "--service",
                    service.url,
                    log=tmp_path / f"{name}.log",
                    env=env,
                )
            )
            await child.ready(url + "/ready")
            children.append(child)
            caller = A2ACaller(url, token)
            stack.push_async_callback(caller.close)
            callers.append(caller)
        yield Workers(callers[0], callers[1], children[0], children[1])
    finally:
        await finish_cleanup(stack.aclose())


@asynccontextmanager
async def owned_thread() -> AsyncIterator[str]:
    from livekit.agents.storage import AgentDBStorage, ThreadNotFound

    storage = AgentDBStorage()

    async def create() -> str:
        async with asyncio.timeout(REQUEST_SECONDS):
            return (await storage.threads.create_thread(scope="e2e-" + uuid.uuid4().hex)).thread_id

    creating = asyncio.create_task(create())
    try:
        yield await asyncio.shield(creating)
    finally:

        async def delete() -> None:
            created = await creating
            async with asyncio.timeout(REQUEST_SECONDS):
                try:
                    await storage.threads.delete_thread(created)
                except ThreadNotFound:
                    pass

        await finish_cleanup(delete())


@pytest.fixture
async def thread(live_credentials: tuple[str, str, str]) -> AsyncIterator[str]:
    """A fresh AgentDB thread, deleted at teardown."""
    async with owned_thread() as thread_id:
        yield thread_id


@pytest.fixture(params=["sqlite", "agentdb"])
async def thread_storage_worker(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[Callable[[str], Storage]]:
    """``thread_storage_worker("worker-c")`` is one more worker on the same storage; the AgentDB
    case skips when the environment names no project."""
    made: list[Storage] = []
    if request.param == "agentdb":
        if (why := agentstore_configured()) is not None:
            pytest.skip(f"no agentstore project in the environment: {why}")

        def make(holder: str) -> Storage:
            made.append(project_storage(holder))
            return made[-1]
    else:
        directory = tmp_path_factory.mktemp("storage")

        def make(holder: str) -> Storage:
            made.append(
                SQLiteStorage(
                    directory=directory,
                    lease_ttl=timedelta(seconds=LOCAL_LEASE_TTL),
                    renew_interval=timedelta(seconds=LOCAL_LEASE_TTL / 4),
                    holder=holder,
                )
            )
            return made[-1]

    yield make

    for storage in made:
        await storage.aclose()


@pytest.fixture
def thread_storages(
    thread_storage_worker: Callable[[str], Storage],
) -> tuple[Storage, Storage]:
    """Two workers sharing one storage, each with a holder of its own."""
    return thread_storage_worker("worker-a"), thread_storage_worker("worker-b")


@pytest.fixture
def no_project_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """In-process servers address themselves on loopback: a developer's exported project
    environment must not shape their URLs, names or versions."""
    for name in (
        "LIVEKIT_URL",
        "LIVEKIT_AGENT_NAME",
        "LIVEKIT_AGENT_NAME_OVERRIDE",
        "LIVEKIT_AGENT_DEPLOYMENT",
        "LIVEKIT_AGENT_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def own_executor(
    tmp_path: Path,
) -> AsyncIterator[Callable[[_core.JobSupervisor], _core.JobSupervisor]]:
    """Registers the job runners a test starts; closes each, with cleanup evidence, at teardown."""
    executors: list[_core.JobSupervisor] = []

    def register(executor: _core.JobSupervisor) -> _core.JobSupervisor:
        executors.append(executor)
        return executor

    yield register
    failures: list[Exception] = []
    for executor in executors:
        try:
            await close_owned_executor(executor, tmp_path / "ipc-cleanup")
        except Exception as error:
            failures.append(error)
    if failures:
        raise ExceptionGroup("IPC fixture cleanup failures", failures)


@pytest.fixture
def job_worker(tmp_path: Path) -> Iterator[StartJobWorker]:
    """``job_worker("crash", jobs=2, **server_options)`` starts one; teardown reaps its tree."""
    started: list[JobWorker] = []

    def start(
        scenario: str, jobs: int = 1, *, module_launch: bool = False, **options: object
    ) -> JobWorker:
        worker = JobWorker(tmp_path, scenario, jobs, module_launch=module_launch, **options)
        started.append(worker)
        return worker

    yield start
    for worker in started:
        worker.close()


class AvatarWorker:
    """An avatar worker's end of the agent's data-stream audio, in a room as ``avatar``: keeps
    each segment the agent streams in ``segments``, counts ``lk.clear_buffer`` calls and reports
    a segment played when ``played()`` is called."""

    def __init__(self, room: rtc.Room, agent_identity: str) -> None:
        self.room = room
        self.agent_identity = agent_identity
        self.segments: asyncio.Queue[tuple[rtc.ByteStreamInfo, bytes]] = asyncio.Queue()
        self.clears = 0
        self._reads: set[asyncio.Task[None]] = set()
        room.register_byte_stream_handler("lk.audio_stream", self._receive)
        room.local_participant.register_rpc_method("lk.clear_buffer", self._clear)

    def _receive(self, reader: rtc.ByteStreamReader, sender: str) -> None:
        async def read() -> None:
            self.segments.put_nowait((reader.info, b"".join([chunk async for chunk in reader])))

        task = asyncio.get_running_loop().create_task(read())
        self._reads.add(task)
        task.add_done_callback(self._reads.discard)

    async def _clear(self, data: rtc.RpcInvocationData) -> str:
        self.clears += 1
        return "ok"

    async def played(
        self, segment: rtc.ByteStreamInfo, position: float, *, interrupted: bool = False
    ) -> None:
        await self.room.local_participant.perform_rpc(
            destination_identity=self.agent_identity,
            method=f"lk.playback_finished/{segment.stream_id}/{segment.name}",
            payload=json.dumps({"playback_position": position, "interrupted": interrupted}),
        )

    async def aclose(self) -> None:
        for task in self._reads:
            task.cancel()
        await asyncio.gather(*self._reads, return_exceptions=True)


JoinAvatar = Callable[[str, str], AbstractAsyncContextManager[AvatarWorker]]


@pytest.fixture
def join_avatar() -> JoinAvatar:
    """``async with join_avatar(sfu, room_name) as avatar:`` joins an avatar worker for the
    room's ``agent`` participant."""

    @asynccontextmanager
    async def join(url: str, room_name: str) -> AsyncIterator[AvatarWorker]:
        async with connected(url, room_name, "avatar") as room:
            worker = AvatarWorker(room, "agent")
            try:
                yield worker
            finally:
                await worker.aclose()

    return join
