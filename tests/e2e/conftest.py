"""Fixture-owned resources; selecting E2E makes missing prerequisites an error."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import pytest

from .drivers import A2ACaller, Workers
from .resources import (
    REQUEST_SECONDS,
    Caller,
    Child,
    RecordingService,
    child_process,
    finish_cleanup,
    recording_service,
)


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
    from livekit.agents.a2a import Credentials

    _, key, secret = live_credentials
    token = Credentials(key, secret).project_token()
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join((str(root / "livekit-agents-next"), str(root)))
    stack = AsyncExitStack()
    try:
        children: list[Child] = []
        callers: list[A2ACaller] = []
        for name in ("worker-a", "worker-b"):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            url = f"http://127.0.0.1:{port}/agents/e2e/default"
            child = await stack.enter_async_context(
                child_process(
                    sys.executable,
                    "-m",
                    "tests.e2e.worker",
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
    async with owned_thread() as thread_id:
        yield thread_id


@pytest.fixture
async def other_thread(live_credentials: tuple[str, str, str]) -> AsyncIterator[str]:
    async with owned_thread() as thread_id:
        yield thread_id


@pytest.fixture
async def room(live_credentials: tuple[str, str, str]) -> AsyncIterator[str]:
    from livekit import api

    url, key, secret = live_credentials
    name = "e2e-" + uuid.uuid4().hex
    client = api.LiveKitAPI(url, key, secret)
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            await client.room.create_room(api.CreateRoomRequest(name=name))
        yield name
    finally:

        async def close() -> None:
            try:
                async with asyncio.timeout(REQUEST_SECONDS):
                    try:
                        await client.room.delete_room(api.DeleteRoomRequest(room=name))
                    except api.TwirpError as error:
                        if error.code != "not_found":
                            raise
            finally:
                await client.aclose()

        await finish_cleanup(close())


@pytest.fixture
async def caller(room: str, live_credentials: tuple[str, str, str]) -> AsyncIterator[Caller]:
    from livekit import api

    url, key, secret = live_credentials
    token = (
        api.AccessToken(key, secret)
        .with_identity("e2e-caller")
        .with_grants(api.VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    peer = Caller()
    try:
        async with asyncio.timeout(REQUEST_SECONDS):
            await peer.room.connect(url, token)
        yield peer
    finally:
        await finish_cleanup(peer.close())
