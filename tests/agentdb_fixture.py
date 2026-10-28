"""The agentstore the storage tests run against.

By default the fake of ``fake_agentstore.py``, started once per session on loopback. With
``AGENTSTORE_PROJECT=1``, the project of ``LIVEKIT_URL``, ``LIVEKIT_API_KEY`` and
``LIVEKIT_API_SECRET`` (staging), and every test skips when the project is not set.
``agentdb_storage`` is one ``AgentDBStorage``; ``agentdb_workers`` two of them with holders of their
own, which is two workers of one project; ``agentdb_worker`` the factory behind both. Threads are
minted by the threads service, never reused across tests.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable
from datetime import timedelta

import pytest

from livekit.agents.a2a import pump
from livekit.agents.storage import AgentDBStorage

from . import fake_agentstore

ON_PROJECT = os.environ.get("AGENTSTORE_PROJECT") == "1"
"""Whether the tests run against the project's agentstore rather than the fake."""
LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10" if ON_PROJECT else "2"))
"""The lane lease's lifetime in seconds: what ``AgentDBStorage`` reads from the environment on
the project, short on the fake so that a test waits seconds for a lease to lapse."""
LEASE_MARGIN = 1.0
"""What a test sleeps past ``LEASE_TTL`` before it counts a lease as lapsed."""
PROJECT_URL = os.environ.get("LIVEKIT_URL", "") if ON_PROJECT else ""
"""The project's URL, read at import: a test module may remove ``LIVEKIT_URL`` so that its
servers address themselves locally, and the storage still needs it."""


def project_storage(holder: str = "") -> AgentDBStorage:
    """An ``AgentDBStorage`` on the fake or on ``PROJECT_URL``, which it reads only while it is
    built."""
    with pytest.MonkeyPatch.context() as mp:
        if not ON_PROJECT:
            mp.setenv("LIVEKIT_URL", fake_agentstore.serve())
            return AgentDBStorage(
                api_key="devkey",
                api_secret="secret",
                lease_ttl=timedelta(seconds=LEASE_TTL),
                renew_interval=timedelta(seconds=LEASE_TTL / 4),
                holder=holder,
            )
        if PROJECT_URL:
            mp.setenv("LIVEKIT_URL", PROJECT_URL)
        return AgentDBStorage(holder=holder)


def agentstore_configured() -> str | None:
    """``None`` when the storage can be built (always on the fake), else why the tests skip."""
    try:
        project_storage()
    except ValueError as e:
        return str(e)
    return None


async def new_session(storage: AgentDBStorage, agent_name: str) -> str:
    """A fresh thread id from the threads service, its database created with it."""
    return (await storage.threads.create_thread()).thread_id


async def let_the_lease_lapse() -> None:
    await asyncio.sleep(LEASE_TTL + LEASE_MARGIN)


@pytest.fixture
async def agentdb_worker() -> AsyncIterator[Callable[[str], AgentDBStorage]]:
    """``agentdb_worker("worker-c")`` is one more worker against the same project."""
    if (why := agentstore_configured()) is not None:
        pytest.skip(f"no agentstore project in the environment: {why}")

    def make(holder: str) -> AgentDBStorage:
        return project_storage(holder)

    yield make

    # an A2A response ends before its turn has closed the durable stream (a2a/pump.py); that
    # close is a request to agent-db now, so it is given its time before the leak check
    # a turn may start the close of a stream as it ends
    for _ in range(3):
        if turns := [task for task in pump.running if not task.done()]:
            await asyncio.wait(turns, timeout=5)


@pytest.fixture
def agentdb_storage(agentdb_worker: Callable[[str], AgentDBStorage]) -> AgentDBStorage:
    return agentdb_worker("worker-a")


@pytest.fixture
def agentdb_workers(
    agentdb_worker: Callable[[str], AgentDBStorage],
) -> tuple[AgentDBStorage, AgentDBStorage]:
    return agentdb_worker("worker-a"), agentdb_worker("worker-b")
