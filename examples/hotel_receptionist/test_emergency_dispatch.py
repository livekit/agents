from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB, NotFound

pytestmark = pytest.mark.unit


def _db() -> HotelDB:
    return HotelDB.from_bytes(build_seed_bytes(TODAY))


def _dispatch_rows(db: HotelDB) -> list[tuple[str, str, str]]:
    return db.connection.execute(
        "SELECT room_id, kind, situation FROM emergency_dispatches"
    ).fetchall()


@pytest.mark.asyncio
async def test_dispatch_to_known_room_records_the_dispatch() -> None:
    db = _db()
    code = await db.dispatch_emergency(room="402", kind="medical", situation="guest collapsed")
    assert code.startswith("EMG-")
    assert _dispatch_rows(db) == [("RM_402", "medical", "guest collapsed")]


@pytest.mark.asyncio
async def test_unknown_room_raises_so_the_agent_reconfirms() -> None:
    db = _db()
    with pytest.raises(NotFound):
        await db.dispatch_emergency(room="408", kind="medical", situation="husband unresponsive")
    assert _dispatch_rows(db) == []


@pytest.mark.asyncio
async def test_bad_kind_is_rejected_distinctly_from_a_bad_room() -> None:
    db = _db()
    with pytest.raises(ValueError):
        await db.dispatch_emergency(room="402", kind="noise", situation="loud neighbour")
    assert _dispatch_rows(db) == []
