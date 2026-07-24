from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pytest
from fake_data.seed import build_seed_bytes
from hotel_db import TODAY, HotelDB, speak_room
from pydantic import TypeAdapter, ValidationError


@pytest.fixture
def db() -> HotelDB:
    return HotelDB.from_bytes(build_seed_bytes(TODAY))


def test_speak_room() -> None:
    assert speak_room("RM_304") == "room 304"
    assert speak_room("RM_PH") == "the penthouse suite"


def test_seed_has_exactly_one_penthouse(db: HotelDB) -> None:
    # PenthouseSuite carries no id because RM_PH is the only penthouse; this
    # guards the assumption against future seed edits.
    rows = list(db.connection.execute("SELECT id FROM hotel_rooms WHERE type = 'penthouse'"))
    assert rows == [("RM_PH",)]


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
    code, _, _ = _order(db, room_id="RM_304", delivery_instructions="as_early_as_possible")
    assert _row(db, code) == ("RM_304", None, "as_early_as_possible")


def test_amend_sets_instructions(db: HotelDB) -> None:
    code, _, _ = _order(db, room_id="RM_304")
    asyncio.run(db.amend_florist_order(code=code, delivery_instructions="before_noon_if_possible"))
    assert _row(db, code) == ("RM_304", None, "before_noon_if_possible")


def test_amend_accepts_code_as_spoken_to_the_caller(db: HotelDB) -> None:
    # The model only ever sees the code via _speak_code, which uppercases it,
    # so the code it passes back to amend_florist_order is the uppercase form.
    code, _, _ = _order(db, room_id="RM_304")
    asyncio.run(
        db.amend_florist_order(code=code.upper(), delivery_instructions="before_noon_if_possible")
    )
    assert _row(db, code) == ("RM_304", None, "before_noon_if_possible")


def test_amend_unknown_code_is_not_found(db: HotelDB) -> None:
    with pytest.raises(NotFound):
        asyncio.run(db.amend_florist_order(code="FLR-NOPE", delivery_instructions="x"))


# --- tool layer: destination validation ---

from tools_services import (  # noqa: E402
    NumberedRoom,
    PenthouseSuite,
    Room,
    _florist_destination,
    room_to_id,
)

from livekit.agents import ToolError, function_tool  # noqa: E402
from livekit.agents.llm.utils import prepare_function_arguments  # noqa: E402


@function_tool
async def _echo_room(room: Room) -> str:
    """Return a room.

    Args:
        room: A numbered room, or the penthouse.
    """
    return room_to_id(room)


def test_destination_neither_raises(db: HotelDB) -> None:
    with pytest.raises(ToolError, match="ask the caller"):
        _florist_destination(None, None)
    with pytest.raises(ToolError):
        _florist_destination(None, "  ")


def test_room_models_keep_distinct_types_with_scalar_json() -> None:
    adapter = TypeAdapter(Room)

    numbered = adapter.validate_python(206)
    assert isinstance(numbered, NumberedRoom)
    assert adapter.dump_python(numbered) == 206
    assert room_to_id(numbered) == "RM_206"

    penthouse = adapter.validate_python("penthouse")
    assert isinstance(penthouse, PenthouseSuite)
    assert adapter.dump_python(penthouse) == "penthouse"
    assert room_to_id(penthouse) == "RM_PH"


def test_room_models_reject_arbitrary_room_phrases() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Room).validate_python("suite 206")


@pytest.mark.parametrize(
    ("arguments", "expected_type", "expected_id"),
    [
        ('{"room": 206}', NumberedRoom, "RM_206"),
        ('{"room": "penthouse"}', PenthouseSuite, "RM_PH"),
    ],
)
def test_tool_arguments_preserve_room_model_type(
    arguments: str,
    expected_type: type[NumberedRoom] | type[PenthouseSuite],
    expected_id: str,
) -> None:
    args, _ = prepare_function_arguments(fnc=_echo_room, json_arguments=arguments)
    assert isinstance(args[0], expected_type)
    assert room_to_id(args[0]) == expected_id


from benchmark import diff_databases  # noqa: E402


def _db_with_order(**overrides: Any) -> HotelDB:
    fresh = HotelDB.from_bytes(build_seed_bytes(TODAY))
    _order(fresh, **overrides)
    return fresh


def test_diff_catches_delivery_instruction_divergence() -> None:
    a = _db_with_order(room_id="RM_304", delivery_instructions="as_early_as_possible")
    b = _db_with_order(room_id="RM_304", delivery_instructions="")
    diffs = diff_databases(a.connection, b.connection)
    assert any("delivery_instructions" in d for d in diffs)


def test_diff_catches_destination_divergence() -> None:
    a = _db_with_order(room_id="RM_PH")
    b = _db_with_order(recipient_name="Diane Okafor")
    diffs = diff_databases(a.connection, b.connection)
    assert any("florist_orders" in d for d in diffs)
