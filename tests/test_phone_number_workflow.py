from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from livekit.agents import Agent, beta

pytestmark = pytest.mark.unit

_HOTEL_EXAMPLE = Path(__file__).parents[1] / "examples" / "hotel_receptionist"
sys.path.insert(0, str(_HOTEL_EXAMPLE))

import tools_restaurant  # noqa: E402
from book_restaurant import (  # noqa: E402
    BookRestaurantTask,
    RestaurantReservationNotCreatedError,
)
from fake_data.seed import build_seed_bytes  # noqa: E402
from hotel_db import TODAY, HotelDB  # noqa: E402
from tools_restaurant import RestaurantToolsMixin  # noqa: E402


class _DeclinedPhoneTask:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _decline() -> None:
            raise beta.workflows.PhoneNumberCaptureDeclinedError("caller declined")

        return _decline().__await__()


class _FailedRestaurantTask:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _fail() -> None:
            raise RestaurantReservationNotCreatedError("phone number required")

        return _fail().__await__()


class _RestaurantAgent(RestaurantToolsMixin, Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")


async def test_restaurant_phone_refusal_ends_without_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(beta.workflows, "GetPhoneNumberTask", _DeclinedPhoneTask)
    db = HotelDB.from_bytes(build_seed_bytes(TODAY))
    before = db.connection.execute("SELECT count(*) FROM restaurant_reservations").fetchone()
    task = BookRestaurantTask(db)

    output = await task.open_phone_dialog()

    after = db.connection.execute("SELECT count(*) FROM restaurant_reservations").fetchone()
    assert before == after
    assert output is not None
    assert task.done()
    task_error = task._AgentTask__fut.exception()  # type: ignore[attr-defined]
    assert isinstance(task_error, RestaurantReservationNotCreatedError)
    await db.aclose()


async def test_restaurant_tool_reports_refusal_as_not_reserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tools_restaurant, "BookRestaurantTask", _FailedRestaurantTask)
    agent = _RestaurantAgent()
    context = SimpleNamespace(userdata=SimpleNamespace(db=object()))

    output = await agent.start_restaurant_booking(context)  # type: ignore[arg-type]

    assert output is not None
