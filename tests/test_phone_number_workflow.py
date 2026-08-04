from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from examples.hotel_receptionist import tools_restaurant
from examples.hotel_receptionist.book_restaurant import (
    BookRestaurantTask,
    RestaurantReservationNotCreatedError,
)
from examples.hotel_receptionist.fake_data.seed import build_seed_bytes
from examples.hotel_receptionist.hotel_db import HotelDB
from examples.hotel_receptionist.tools_restaurant import RestaurantToolsMixin
from livekit.agents import beta

pytestmark = pytest.mark.unit

TODAY = date(2026, 6, 8)


class _DeclinedPhoneTask:
    last_extra_instructions = ""

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_extra_instructions = kwargs["extra_instructions"]

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _decline() -> None:
            raise beta.workflows.PhoneNumberCaptureDeclinedError("caller declined")

        return _decline().__await__()


class _FailedRestaurantTask:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _fail() -> None:
            raise RestaurantReservationNotCreatedError("phone number required")

        return _fail().__await__()


class _RestaurantAgent(RestaurantToolsMixin):
    def __init__(self) -> None:
        super().__init__(instructions="test")


@pytest.mark.asyncio
async def test_restaurant_phone_refusal_ends_without_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(beta.workflows, "GetPhoneNumberTask", _DeclinedPhoneTask)
    db = HotelDB.from_bytes(build_seed_bytes(TODAY), TODAY)
    before = db.connection.execute("SELECT count(*) FROM restaurant_reservations").fetchone()
    task = BookRestaurantTask(db, TODAY)

    output = await task.open_phone_dialog()

    after = db.connection.execute("SELECT count(*) FROM restaurant_reservations").fetchone()
    assert before == after
    assert task.done()
    task_error = task._AgentTask__fut.exception()  # type: ignore[attr-defined]
    assert isinstance(task_error, RestaurantReservationNotCreatedError)
    assert output is not None and "never tell the caller the table is reserved" in output
    assert "call\n`decline_phone_number_capture` immediately" in (
        _DeclinedPhoneTask.last_extra_instructions
    )
    await db.aclose()


@pytest.mark.asyncio
async def test_restaurant_tool_reports_refusal_as_not_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools_restaurant, "BookRestaurantTask", _FailedRestaurantTask)
    agent = _RestaurantAgent()
    context = SimpleNamespace(userdata=SimpleNamespace(db=object(), today=TODAY))

    output = await agent.start_restaurant_booking(context)  # type: ignore[arg-type]

    assert output is not None
    assert output.startswith("No reservation was created")
    assert "table is not reserved" in output
