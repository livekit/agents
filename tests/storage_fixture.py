"""The storages the storage and door tests run against: a local ``SQLiteStorage`` in the test's
own directory, and LiveKit Cloud's agentstore when the environment names a project
(``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET``); without one the AgentDB cases
skip. ``thread_storage`` is one storage; ``thread_storages`` two with holders of their own,
which is two workers sharing it; ``thread_storage_worker`` the factory behind both.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from datetime import timedelta

import pytest

from livekit.agents.storage import AgentDBStorage, SQLiteStorage, Storage

AGENTDB_LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10"))
"""The AgentDB lane lease's lifetime in seconds, as ``AgentDBStorage`` reads it."""
LOCAL_LEASE_TTL = 2.0
"""The local lease's lifetime: short, so a test that lets a lease lapse stays fast."""
LEASE_MARGIN = 1.0
"""What a test sleeps past the lease's lifetime before it counts the lease as lapsed."""
PROJECT_URL = os.environ.get("LIVEKIT_URL", "")
"""The project's URL, read at import: a test module may remove ``LIVEKIT_URL`` so that its
servers address themselves locally, and the storage still needs it."""


def project_storage(holder: str = "") -> AgentDBStorage:
    """An ``AgentDBStorage`` on ``PROJECT_URL``, which it reads only while it is built."""
    with pytest.MonkeyPatch.context() as mp:
        if PROJECT_URL:
            mp.setenv("LIVEKIT_URL", PROJECT_URL)
        return AgentDBStorage(holder=holder)


def agentstore_configured() -> str | None:
    """``None`` when the environment names a project, else why the AgentDB cases skip."""
    try:
        project_storage()
    except ValueError as e:
        return str(e)
    return None


def lease_ttl(storage: Storage) -> float:
    return LOCAL_LEASE_TTL if isinstance(storage, SQLiteStorage) else AGENTDB_LEASE_TTL


async def new_session(storage: Storage, agent_name: str) -> str:
    """A fresh thread id, its database created with it."""
    return (await storage.threads.create_thread()).thread_id


async def let_the_lease_lapse(storage: Storage) -> None:
    await asyncio.sleep(lease_ttl(storage) + LEASE_MARGIN)


@pytest.fixture(params=["sqlite", "agentdb"])
async def thread_storage_worker(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> AsyncIterator[Callable[[str], Storage]]:
    """``thread_storage_worker("worker-c")`` is one more worker on the same storage."""
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
def thread_storage(thread_storage_worker: Callable[[str], Storage]) -> Storage:
    return thread_storage_worker("worker-a")


@pytest.fixture
def thread_storages(
    thread_storage_worker: Callable[[str], Storage],
) -> tuple[Storage, Storage]:
    return thread_storage_worker("worker-a"), thread_storage_worker("worker-b")
