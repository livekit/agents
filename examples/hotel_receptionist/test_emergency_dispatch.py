from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB

pytestmark = pytest.mark.unit


def _db() -> HotelDB:
    return HotelDB.from_bytes(build_seed_bytes(TODAY))


def _dispatch_rows(db: HotelDB) -> list[tuple[str, str, str]]:
    return db.connection.execute(
        "SELECT room_id, kind, situation FROM emergency_dispatches"
    ).fetchall()


@pytest.mark.asyncio
async def test_dispatch_to_known_room_reports_room_on_file() -> None:
    db = _db()
    code, room_on_file = await db.dispatch_emergency(
        room="402", kind="medical", situation="guest collapsed"
    )
    assert code.startswith("EMG-")
    assert room_on_file is True
    assert _dispatch_rows(db) == [("RM_402", "medical", "guest collapsed")]


@pytest.mark.asyncio
async def test_unknown_room_never_blocks_the_dispatch() -> None:
    # A panicked caller's room number may not check out against inventory - the
    # dispatch still happens on their word, flagged so the agent re-confirms the
    # room while staff is already moving.
    db = _db()
    code, room_on_file = await db.dispatch_emergency(
        room="408", kind="medical", situation="husband unresponsive"
    )
    assert code.startswith("EMG-")
    assert room_on_file is False
    assert _dispatch_rows(db) == [("RM_408", "medical", "husband unresponsive")]


@pytest.mark.asyncio
async def test_bad_kind_is_still_rejected() -> None:
    db = _db()
    with pytest.raises(ValueError):
        await db.dispatch_emergency(room="402", kind="noise", situation="loud neighbour")
    assert _dispatch_rows(db) == []
