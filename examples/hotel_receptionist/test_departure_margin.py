from __future__ import annotations

import os
import sys
from datetime import date, time, timedelta, timezone
from types import SimpleNamespace

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import HotelReceptionistAgent
from common import Userdata
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB
from instructions import build_instructions
from tools_services import NumberedRoom, _departure_margin_note

from livekit.agents.llm.utils import build_legacy_openai_schema


def test_instructions_require_immediate_booking_reference_readback() -> None:
    assert "read that booking reference back in your very next reply" in build_instructions()


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


@pytest.mark.asyncio
async def test_tz_aware_times_are_stored_naive() -> None:
    # the LLM sometimes passes "17:40:00Z"-style times; pydantic keeps the tzinfo
    # and a raw isoformat() then stores "17:40:00+00:00", which diverges from the
    # plain local wall-clock time everything else (and the graders) expects.
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    try:
        flight_date = TODAY + timedelta(days=2)
        await db.request_flight_reconfirmation(
            room="401",
            airline="Iberia",
            flight_number="IB 6174",
            flight_date=flight_date,
            booking_reference="QX4R7T",
            seat_check=True,
            departure_time=time(17, 40, tzinfo=timezone.utc),
        )
        stored = db.connection.execute(
            "SELECT departure_time FROM flight_reconfirmations ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        assert stored == "17:40:00"
        assert await db.latest_flight_departure(room="401", flight_date=flight_date) == time(17, 40)

        await db.book_airport_car(
            room="401",
            pickup_date=flight_date,
            pickup_time=time(14, 30, tzinfo=timezone.utc),
            passengers=2,
        )
        stored = db.connection.execute(
            "SELECT pickup_time FROM airport_cars ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        assert stored == "14:30:00"
    finally:
        await db.aclose()


@pytest.mark.asyncio
async def test_airport_car_uses_explicit_departure_and_echoes_booking_reference() -> None:
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    ctx = SimpleNamespace(userdata=Userdata(db=db))
    agent = HotelReceptionistAgent()
    car_tool = next(tool for tool in agent.tools if tool.__name__ == "book_airport_car")
    flight_tool = next(
        tool for tool in agent.tools if tool.__name__ == "request_flight_reconfirmation"
    )
    flight_date = TODAY + timedelta(days=2)
    room = NumberedRoom(401)

    try:
        car_result = await car_tool(
            ctx=ctx,
            room=room,
            pickup_date=flight_date,
            pickup_time=time(14, 30),
            flight_departure_time=time(17, 40),
            passengers=2,
        )
        assert "about 3 hours" in car_result
        assert db.connection.execute("SELECT COUNT(*) FROM airport_cars").fetchone()[0] == 1

        flight_result = await flight_tool(
            ctx=ctx,
            room=room,
            airline="Iberia",
            flight_number="IB 6174",
            flight_date=flight_date,
            booking_reference="Q-X-4-R-7-T",
            seat_check=True,
            departure_time=time(17, 40),
        )
        assert "airline booking reference captured is Q, X, 4, R, 7, T" in flight_result
        assert await db.latest_flight_departure(room="401", flight_date=flight_date) == time(17, 40)
    finally:
        await db.aclose()


def test_reconfirmation_schema_requires_departure_time_but_accepts_null() -> None:
    # Required-but-nullable: the model must confront the field on every call (so it
    # asks the caller), while a caller who doesn't know the time never blocks the tool.
    agent = HotelReceptionistAgent()
    flight_tool = next(
        tool for tool in agent.tools if tool.__name__ == "request_flight_reconfirmation"
    )
    schema = build_legacy_openai_schema(flight_tool)
    parameters = schema["function"]["parameters"]

    assert "departure_time" in parameters["required"]
    prop = parameters["properties"]["departure_time"]
    assert "null" in str(prop)
    assert "ASK the caller" in prop["description"]


def test_airport_car_schema_requires_flight_departure_time() -> None:
    agent = HotelReceptionistAgent()
    car_tool = next(tool for tool in agent.tools if tool.__name__ == "book_airport_car")
    schema = build_legacy_openai_schema(car_tool)
    parameters = schema["function"]["parameters"]

    assert "flight_departure_time" in parameters["required"]
    assert parameters["properties"]["flight_departure_time"]["format"] == "time"
    assert "ASK the caller" in parameters["properties"]["flight_departure_time"]["description"]
