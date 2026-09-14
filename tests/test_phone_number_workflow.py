from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from livekit.agents import beta
from livekit.agents.llm.tool_context import ToolError

pytestmark = pytest.mark.unit

_HOTEL_EXAMPLE = Path(__file__).parents[1] / "examples" / "hotel_receptionist"
sys.path.insert(0, str(_HOTEL_EXAMPLE))

from book_restaurant import BookRestaurantTask  # noqa: E402
from fake_data.seed import build_seed_bytes  # noqa: E402
from hotel_db import TODAY, HotelDB  # noqa: E402


class _DeclinedPhoneTask:
    def __init__(self, **kwargs: Any) -> None:
        pass

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _decline() -> None:
            raise beta.workflows.PhoneNumberCaptureDeclinedError("caller declined")

        return _decline().__await__()


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
    assert isinstance(task_error, ToolError)
    await db.aclose()
