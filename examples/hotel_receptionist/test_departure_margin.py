from __future__ import annotations

import os
import sys
from datetime import time

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools_services import _departure_margin_note


def test_three_hour_margin() -> None:
    note = _departure_margin_note(pickup=time(14, 30), departure=time(17, 40))
    assert "about 3 hours" in note
    assert "5:40" in note  # the departure it was checked against, spoken
    assert "margin" in note.lower() or "before" in note.lower()


def test_half_hour_granularity() -> None:
    note = _departure_margin_note(pickup=time(14, 30), departure=time(16, 0))
    assert "about 1 and a half hours" in note


def test_tight_margin_is_flagged() -> None:
    note = _departure_margin_note(pickup=time(16, 30), departure=time(17, 40))
    assert "TIGHT" in note


def test_pickup_not_before_departure_is_flagged() -> None:
    note = _departure_margin_note(pickup=time(18, 0), departure=time(17, 40))
    assert "not before" in note.lower()
    assert "re-check" in note.lower()


# --- departure time round-trips through the DB --------------------------------


@pytest.mark.asyncio
async def test_departure_time_is_stored_and_looked_up() -> None:
    from datetime import date

    from fake_data.seed import build_seed_bytes
    from hotel_db import TODAY, HotelDB

    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        flight_date = date(2026, 6, 11)
        assert await db.latest_flight_departure(room="401", flight_date=flight_date) is None
        await db.request_flight_reconfirmation(
            room="401",
            airline="Iberia",
            flight_number="IB 6174",
            flight_date=flight_date,
            booking_reference="QX4R7T",
            seat_check=True,
            departure_time=time(17, 40),
        )
        assert await db.latest_flight_departure(room="401", flight_date=flight_date) == time(17, 40)
        # a different day's pickup has nothing to check against
        assert await db.latest_flight_departure(room="401", flight_date=date(2026, 6, 12)) is None
    finally:
        await db.aclose()
