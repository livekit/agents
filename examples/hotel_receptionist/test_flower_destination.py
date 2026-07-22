from __future__ import annotations

import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB
from tools_services import _resolve_flower_destination

from livekit.agents import ToolError


@pytest.fixture
def db() -> HotelDB:
    return HotelDB.from_bytes(build_seed_bytes(TODAY))


def test_location_only() -> None:
    assert _resolve_flower_destination("Penthouse Suite", None) == "Penthouse Suite"


def test_recipient_only() -> None:
    assert _resolve_flower_destination(None, "Diane Okafor") == "Diane Okafor"


def test_location_wins_when_both_are_given() -> None:
    # the observed failure: "Penthouse Suite, Diane Okafor" jammed into one field.
    # with both known, the stored destination is the location alone.
    assert _resolve_flower_destination("Penthouse Suite", "Diane Okafor") == "Penthouse Suite"


def test_values_are_trimmed() -> None:
    assert _resolve_flower_destination("  412 ", None) == "412"


def test_blank_location_falls_back_to_recipient() -> None:
    assert _resolve_flower_destination("   ", "Diane Okafor") == "Diane Okafor"


def test_neither_raises_tool_error() -> None:
    with pytest.raises(ToolError):
        _resolve_flower_destination(None, None)
    with pytest.raises(ToolError):
        _resolve_flower_destination("  ", "")


# --- DB layer: order_flowers destination rules + amend_florist_order ---

import asyncio  # noqa: E402
from datetime import timedelta  # noqa: E402

from hotel_db import NotFound, Unavailable  # noqa: E402

_ORDER_KWARGS = {
    "arrangement_id": "roses",
    "guest_name": "Marcus Webb",
    "guest_phone": "415-555-0182",
    "card_message": "Happy ten years.",
}


def _order(db: HotelDB, **overrides):
    kwargs = {**_ORDER_KWARGS, "on_date": TODAY + timedelta(days=2), **overrides}
    return asyncio.run(db.order_flowers(**kwargs))


def _row(db: HotelDB, code: str) -> tuple:
    for row in db.connection.execute(
        "SELECT room_id, recipient_name, delivery_instructions FROM florist_orders"
        " WHERE code = :code",
        {"code": code},
    ):
        return row
    raise AssertionError(f"no florist_orders row for {code}")


def test_order_to_room_stores_room_and_no_recipient(db: HotelDB) -> None:
    code, _, _ = _order(db, room_id="RM_304")
    assert _row(db, code) == ("RM_304", None, "")


def test_order_to_recipient_stores_name_and_no_room(db: HotelDB) -> None:
    code, _, _ = _order(db, recipient_name="Theodore Lansing")
    assert _row(db, code) == (None, "Theodore Lansing", "")


def test_room_wins_when_both_are_given(db: HotelDB) -> None:
    # path-independence: whether the agent also collected a name must not
    # change the stored state.
    code, _, _ = _order(db, room_id="RM_PH", recipient_name="Diane Okafor")
    assert _row(db, code) == ("RM_PH", None, "")


def test_unknown_room_is_not_found(db: HotelDB) -> None:
    with pytest.raises(NotFound):
        _order(db, room_id="RM_412")


def test_no_destination_is_unavailable(db: HotelDB) -> None:
    with pytest.raises(Unavailable):
        _order(db)
    with pytest.raises(Unavailable):
        _order(db, room_id="  ", recipient_name="")


def test_instructions_stored_with_order(db: HotelDB) -> None:
    code, _, _ = _order(db, room_id="RM_304", delivery_instructions="as early as possible")
    assert _row(db, code) == ("RM_304", None, "as early as possible")


def test_amend_sets_instructions(db: HotelDB) -> None:
    code, _, _ = _order(db, room_id="RM_304")
    asyncio.run(db.amend_florist_order(code=code, delivery_instructions="before noon"))
    assert _row(db, code) == ("RM_304", None, "before noon")


def test_amend_unknown_code_is_not_found(db: HotelDB) -> None:
    with pytest.raises(NotFound):
        asyncio.run(db.amend_florist_order(code="FLR-NOPE", delivery_instructions="x"))


# --- tool layer: destination validation ---

from tools_services import _florist_destination  # noqa: E402

from livekit.agents import ToolError  # noqa: E402


def test_destination_room_phrase_resolves(db: HotelDB) -> None:
    assert _florist_destination(db, "room 304", None) == ("RM_304", None)
    assert _florist_destination(db, "the penthouse suite", "Diane Okafor") == ("RM_PH", None)


def test_destination_recipient_only(db: HotelDB) -> None:
    assert _florist_destination(db, None, "Theodore Lansing") == (None, "Theodore Lansing")
    assert _florist_destination(db, "  ", " Theodore Lansing ") == (None, "Theodore Lansing")


def test_destination_placeholder_raises(db: HotelDB) -> None:
    # a bare stand-in word must bounce even when a recipient is also present -
    # the agent said "room" without the caller naming one.
    with pytest.raises(ToolError):
        _florist_destination(db, "room", "Diane Okafor")


def test_destination_unknown_room_raises(db: HotelDB) -> None:
    with pytest.raises(ToolError, match="doesn't match"):
        _florist_destination(db, "412", None)


def test_destination_neither_raises(db: HotelDB) -> None:
    with pytest.raises(ToolError, match="ask the caller"):
        _florist_destination(db, None, None)
    with pytest.raises(ToolError):
        _florist_destination(db, "", "  ")
