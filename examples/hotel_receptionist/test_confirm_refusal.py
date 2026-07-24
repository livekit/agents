from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from book_restaurant import BookRestaurantTask
from book_room import BookRoomTask
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB

from livekit.agents import ToolError

pytestmark = pytest.mark.unit

# Confirmation codes are "HTL-XXXXXXXXXXXX" / "RES-XXXX". The agent invented
# HTL-9R3L5K1MZX2Y off the back of a refusal that only described the next step,
# so the refusal text must not contain anything code-shaped for it to copy.
_CODE_SHAPED = re.compile(r"\b[A-Z]{3}-[A-Z0-9]{4,}\b")


def _db() -> HotelDB:
    return HotelDB.from_bytes(build_seed_bytes(TODAY))


def _room_bookings(db: HotelDB) -> int:
    (count,) = next(db.connection.execute("SELECT COUNT(*) FROM hotel_bookings"))
    return count


def _restaurant_reservations(db: HotelDB) -> int:
    (count,) = next(db.connection.execute("SELECT COUNT(*) FROM restaurant_reservations"))
    return count


async def _partial_room_task(db: HotelDB) -> BookRoomTask:
    """A draft carrying the stay and the room but none of the identity fields -
    the exact state the agent was in when it spoke a code for nothing."""
    task = BookRoomTask(db)
    check_in = TODAY + timedelta(days=30)
    await task.set_stay(check_in=check_in, check_out=check_in + timedelta(days=2), guests=2)
    await task.choose_room(room_type="king", extras=[])
    return task


@pytest.mark.asyncio
async def test_confirm_booking_refusal_states_that_nothing_was_booked() -> None:
    db = _db()
    before = _room_bookings(db)
    task = await _partial_room_task(db)

    with pytest.raises(ToolError) as excinfo:
        await task.confirm_booking()

    assert _room_bookings(db) == before, "a refused confirm_booking wrote a booking row"
    assert not task.done()
    message = excinfo.value.message
    assert "NOT booked" in message
    assert "no confirmation code exists" in message
    assert not _CODE_SHAPED.search(message), message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clear",
    ["_check_in", "_room_type", "_first_name", "_email", "_phone", "_card_last4"],
)
async def test_every_missing_room_field_refuses_the_same_way(clear: str) -> None:
    # Whichever field is missing, the refusal has to carry the outcome - the
    # branch of _status() it lands on must never be the whole message.
    db = _db()
    task = await _partial_room_task(db)
    task._first_name, task._last_name = "Aaron", "Delgado"
    task._email, task._phone, task._card_last4 = "a@b.com", "4155550190", "4242"
    setattr(task, clear, None)

    with pytest.raises(ToolError) as excinfo:
        await task.confirm_booking()

    message = excinfo.value.message
    assert "NOT booked" in message
    assert not _CODE_SHAPED.search(message), message


@pytest.mark.asyncio
async def test_confirm_reservation_refusal_states_that_nothing_was_reserved() -> None:
    db = _db()
    before = _restaurant_reservations(db)
    task = BookRestaurantTask(db)
    on_date = TODAY + timedelta(days=30)
    await task.set_party(on_date=on_date, party_size=2)

    with pytest.raises(ToolError) as excinfo:
        await task.confirm_reservation()

    assert _restaurant_reservations(db) == before
    assert not task.done()
    message = excinfo.value.message
    assert "NOT reserved" in message
    assert "no confirmation code exists" in message
    assert not _CODE_SHAPED.search(message), message


@pytest.mark.asyncio
async def test_status_stays_a_progress_string() -> None:
    # The refusal wording is additive: the success/progress path keeps its own
    # rendering, which the read-back and the next-step directives depend on.
    task = BookRoomTask(_db())
    assert "NOT booked" not in task._status()
    task._check_in, task._check_out = date(2026, 7, 10), date(2026, 7, 13)
    task._guests, task._room_type = 2, "king"
    task._first_name, task._last_name = "Aaron", "Delgado"
    task._email, task._phone, task._card_last4 = "a@b.com", "4155550190", "4242"
    task._quoted_total = 87360
    status = task._status()
    assert "NOT booked" not in status
    assert "call confirm_booking()" in status
