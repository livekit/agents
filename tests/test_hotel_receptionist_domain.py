from __future__ import annotations

from collections.abc import Iterator
from datetime import date, time, timedelta
from types import SimpleNamespace

import pytest

from examples.hotel_receptionist.common import Userdata
from examples.hotel_receptionist.fake_data.seed import populate
from examples.hotel_receptionist.hotel import (
    PRICING,
    ViewRate,
    cancellation_penalty_applies,
    format_date,
    speak_usd,
)
from examples.hotel_receptionist.hotel_db import HotelDB
from examples.hotel_receptionist.modify_booking import ModifyBookingTask
from examples.hotel_receptionist.tools_restaurant import RestaurantToolsMixin
from examples.hotel_receptionist.tools_rooms import RoomToolsMixin
from livekit.agents import ToolError

pytestmark = pytest.mark.unit

TODAY = date(2026, 6, 8)


@pytest.fixture
def hotel_db() -> Iterator[HotelDB]:
    db = HotelDB.empty(TODAY)
    populate(db, TODAY)
    try:
        yield db
    finally:
        db.close()


def tool_context(db: HotelDB) -> SimpleNamespace:
    return SimpleNamespace(userdata=Userdata(db=db, today=TODAY))


def test_speak_usd_uses_numeric_dollar_figures() -> None:
    assert speak_usd(8500) == "$85"
    assert speak_usd(10010) == "$100.10"


def test_date_format_is_portable() -> None:
    assert format_date(TODAY) == "Monday, June 8"


def test_cancellation_penalty_uses_calendar_days() -> None:
    assert PRICING.cancellation_window_days == 2
    assert cancellation_penalty_applies(check_in=TODAY + timedelta(days=1), today=TODAY)
    assert not cancellation_penalty_applies(check_in=TODAY + timedelta(days=2), today=TODAY)


async def test_business_center_can_be_booked_at_any_hour(hotel_db: HotelDB) -> None:
    code, service, total = await hotel_db.book_business_center(
        service_id="meeting_room",
        guest_name="Sam Lee",
        guest_phone="415-555-0100",
        on_date=TODAY,
        at_time=time(2),
        duration_hours=1,
    )

    assert code.startswith("BIZ-")
    assert service.name == "Meeting room"
    assert total == 4000


async def test_mirror_failure_does_not_fail_a_committed_action(hotel_db: HotelDB) -> None:
    async def fail() -> None:
        raise RuntimeError("mirror unavailable")

    hotel_db.on_change = fail
    code, _, _ = await hotel_db.book_business_center(
        service_id="meeting_room",
        guest_name="Sam Lee",
        guest_phone="415-555-0100",
        on_date=TODAY,
        at_time=time(2),
        duration_hours=1,
    )

    stored = hotel_db.connection.execute(
        "SELECT code FROM business_center_bookings WHERE code = ?", (code,)
    ).fetchone()
    assert stored == (code,)


async def test_modifying_extras_preserves_smoking_room(hotel_db: HotelDB) -> None:
    existing = await hotel_db.find_booking(last_name="Smith", confirmation_code="HTL-AB12")
    assert existing is not None
    assert existing.smoking

    task = ModifyBookingTask(hotel_db, existing, TODAY)
    await task.choose_room(room_type=existing.room_type, extras=[])
    await task.confirm_changes()

    updated = await hotel_db.find_booking(last_name="Smith", confirmation_code="HTL-AB12")
    assert updated is not None
    assert updated.smoking
    assert updated.extras == []


async def test_modification_can_explicitly_change_to_non_smoking(hotel_db: HotelDB) -> None:
    existing = await hotel_db.find_booking(last_name="Smith", confirmation_code="HTL-AB12")
    assert existing is not None
    assert existing.smoking

    task = ModifyBookingTask(hotel_db, existing, TODAY)
    await task.choose_room(
        room_type=existing.room_type,
        extras=existing.extras,
        smoking_room=False,
    )
    await task.confirm_changes()

    updated = await hotel_db.find_booking(last_name="Smith", confirmation_code="HTL-AB12")
    assert updated is not None
    assert not updated.smoking


async def test_room_availability_keeps_each_view_with_its_rate_range(
    hotel_db: HotelDB,
) -> None:
    availability = await hotel_db.list_room_types_available(
        check_in=TODAY + timedelta(days=40),
        check_out=TODAY + timedelta(days=42),
        guests=2,
        smoking=False,
    )

    king = next(option for option in availability if option.type == "king")
    assert king.view_rates == (
        ViewRate(view="city", minimum=24000, maximum=24000),
        ViewRate(view="ocean", minimum=26000, maximum=28000),
    )


async def test_room_availability_formats_view_specific_prices(hotel_db: HotelDB) -> None:
    result = await RoomToolsMixin(instructions="").check_room_availability(
        tool_context(hotel_db),
        TODAY + timedelta(days=40),
        TODAY + timedelta(days=42),
        2,
        "non_smoking",
        "king",
    )

    assert result == "king: city $240/night; ocean $260–$280/night"


async def test_booking_selects_the_lowest_rate_it_advertises() -> None:
    db = HotelDB.empty(TODAY)
    try:
        db.connection.executemany(
            "INSERT INTO hotel_rooms "
            "(id, type, nightly_rate, max_occupancy, smoking, pets_allowed, room_view) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("RM_101", "king", 30000, 2, 0, 0, "ocean"),
                ("RM_102", "king", 25000, 2, 0, 0, "ocean"),
            ],
        )

        booking = await db.book_room(
            room_type="king",
            smoking=False,
            guests=2,
            check_in=TODAY + timedelta(days=1),
            check_out=TODAY + timedelta(days=2),
            first_name="Sam",
            last_name="Lee",
            email="sam@example.com",
            phone="415-555-0100",
            card_last4="4242",
            extras=[],
            view="ocean",
        )

        assert booking.room_id == "RM_102"
        assert booking.nightly_rate == 25000
    finally:
        db.close()


async def test_room_browse_rejects_past_check_in_but_allows_today(
    hotel_db: HotelDB,
) -> None:
    ctx = tool_context(hotel_db)
    tool = RoomToolsMixin(instructions="")

    with pytest.raises(ToolError, match="check-in can't be in the past"):
        await tool.check_room_availability(
            ctx,
            TODAY - timedelta(days=1),
            TODAY + timedelta(days=1),
            2,
            "no_preference",
            "any",
        )

    await tool.check_room_availability(
        ctx,
        TODAY,
        TODAY + timedelta(days=1),
        2,
        "no_preference",
        "any",
    )


async def test_restaurant_browse_rejects_past_date_but_allows_today(
    hotel_db: HotelDB,
) -> None:
    ctx = tool_context(hotel_db)
    tool = RestaurantToolsMixin(instructions="")

    with pytest.raises(ToolError, match="date can't be in the past"):
        await tool.check_restaurant_availability(ctx, TODAY - timedelta(days=1), 2)

    await tool.check_restaurant_availability(ctx, TODAY, 2)


async def test_in_house_booking_can_keep_its_original_check_in(hotel_db: HotelDB) -> None:
    existing = await hotel_db.find_booking(last_name="García", confirmation_code="HTL-EF56")
    assert existing is not None
    assert existing.check_in < TODAY

    task = ModifyBookingTask(hotel_db, existing, TODAY)
    result = await task.set_stay(existing.check_in, existing.check_out + timedelta(days=1), 3)

    assert result.startswith("stay updated")
