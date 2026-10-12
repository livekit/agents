"""The agentstore the storage tests run against, through LiveKit's public API.

A staging project from the environment: ``LIVEKIT_URL``, with ``LIVEKIT_API_KEY`` and
``LIVEKIT_API_SECRET``. ``agentdb_storage`` is one ``AgentDBStorage``; ``agentdb_workers`` two
of them with holders of their own, which is two workers of one project;
``agentdb_worker`` the factory behind both. Every test skips when the project is not set. Threads
are minted by the threads service, never reused across tests.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable

import pytest

from livekit.agents.a2a import pump
from livekit.agents.storage import AgentDBStorage

LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10"))
"""The lane lease's lifetime in seconds, as ``AgentDBStorage`` reads it from the environment."""
LEASE_MARGIN = 1.0
"""What a test sleeps past ``LEASE_TTL`` before it counts a lease as lapsed."""


def agentstore_configured() -> str | None:
    """``None`` when the environment names a project, else why the tests skip."""
    try:
        AgentDBStorage()
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
        return AgentDBStorage(holder=holder)

    yield make

    # an A2A response ends before its turn has closed the durable stream (a2a/pump.py); that
    # close is a request to agent-db now, so it is given its time before the leak check
    turns = [task for task in pump.running if not task.done()]
    if turns:
        await asyncio.wait(turns, timeout=5)


@pytest.fixture
def agentdb_storage(agentdb_worker: Callable[[str], AgentDBStorage]) -> AgentDBStorage:
    return agentdb_worker("worker-a")


@pytest.fixture
def agentdb_workers(
    agentdb_worker: Callable[[str], AgentDBStorage],
) -> tuple[AgentDBStorage, AgentDBStorage]:
    return agentdb_worker("worker-a"), agentdb_worker("worker-b")
