"""The storages two workers share: a local ``SQLiteStorage``, or the project's AgentDB when the
environment names one (``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, ``LIVEKIT_API_SECRET``)."""

from __future__ import annotations

import asyncio
import os

import pytest

from livekit.agents.storage import AgentDBStorage, SQLiteStorage, Storage

AGENTDB_LEASE_TTL = float(os.environ.get("LIVEKIT_AGENTDB_LEASE_TTL", "10"))
"""The AgentDB partition lease's lifetime in seconds, as ``AgentDBStorage`` reads it."""
LOCAL_LEASE_TTL = 2.0
"""The local lease's lifetime: short, so a test that lets a lease lapse stays fast."""
LEASE_MARGIN = 1.0
PROJECT_URL = os.environ.get("LIVEKIT_URL", "")
"""Read at import: a test may remove ``LIVEKIT_URL`` so its servers address themselves locally,
and the storage still needs it."""


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


async def let_the_lease_lapse(storage: Storage) -> None:
    ttl = LOCAL_LEASE_TTL if isinstance(storage, SQLiteStorage) else AGENTDB_LEASE_TTL
    await asyncio.sleep(ttl + LEASE_MARGIN)
