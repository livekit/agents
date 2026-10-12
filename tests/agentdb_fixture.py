"""The local agent-db the storage tests run against (``examples/README.md``, "Environment").

``agentdb_storage`` is one ``AgentDBStorage`` built from the ``LIVEKIT_AGENTDB_*`` variables with
the local stack's defaults; ``agentdb_workers`` two of them with holders of their own, which is
two workers of one project; ``agentdb_worker`` the factory behind both. Every test skips when the
data plane port does not answer. Threads are minted by the threads service, never reused across tests.
"""

from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import AsyncIterator, Callable
from urllib.parse import urlparse

import pytest

from livekit.agents.a2a import pump
from livekit.agents.storage import AgentDBStorage

DATA_PLANE_URL = os.environ.get("LIVEKIT_AGENTDB_URL", "ws://localhost:7781/v1/db")
STREAMS_URL = os.environ.get("LIVEKIT_AGENTDB_STREAMS_URL", "http://localhost:7780")
MANAGEMENT_URL = os.environ.get("LIVEKIT_AGENTDB_MANAGEMENT_URL", "http://localhost:7783")
THREADS_URL = os.environ.get("LIVEKIT_AGENT_THREADS_URL", "http://localhost:7793")
API_KEY = os.environ.get("LIVEKIT_AGENTDB_API_KEY", "devkey")
API_SECRET = os.environ.get("LIVEKIT_AGENTDB_API_SECRET", "secretsecretsecretsecretsecretsecret")

LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10"))
"""The lane lease's lifetime in seconds, as ``AgentDBStorage`` reads it from the environment."""
LEASE_MARGIN = 1.0
"""What a test sleeps past ``LEASE_TTL`` before it counts a lease as lapsed."""


def data_plane_is_up(timeout: float = 0.5) -> bool:
    parsed = urlparse(DATA_PLANE_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def new_session(storage: AgentDBStorage, agent_name: str) -> str:
    """A fresh thread id from the threads service, its database created with it."""
    return (await storage.threads.create_thread()).thread_id


async def let_the_lease_lapse() -> None:
    await asyncio.sleep(LEASE_TTL + LEASE_MARGIN)


@pytest.fixture
async def agentdb_worker() -> AsyncIterator[Callable[[str], AgentDBStorage]]:
    """``agentdb_worker("worker-c")`` is one more worker against the same local agent-db."""
    if not data_plane_is_up():
        pytest.skip(f"no agent-db data plane at {DATA_PLANE_URL}; start the local stack")

    def make(holder: str) -> AgentDBStorage:
        storage = AgentDBStorage(
            DATA_PLANE_URL,
            streams_url=STREAMS_URL,
            management_url=MANAGEMENT_URL,
            threads_url=THREADS_URL,
            api_key=API_KEY,
            api_secret=API_SECRET,
            holder=holder,
        )
        return storage

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
