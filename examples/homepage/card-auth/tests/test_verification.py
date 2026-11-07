import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from livekit.agents.llm import ToolError

from bank import FLAGGED, LOSS_PREVENTION_CASE, MAX_ATTEMPTS, Bank
from card_services import BANK_TZ, CardServices

SAM = {
    "full_name": "Sam Ortiz",
    "date_of_birth": "1984-03-03",
    "card_last_four": "1111",
    "request": "other",
}


CTX = SimpleNamespace(function_call=SimpleNamespace(call_id="call_tool"))


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
def agent(published: dict) -> CardServices:

    async def publish(key, value):
        published.setdefault(f"{key}_history", []).append(value)
        published[key] = value

    return CardServices(Bank(publish=publish))


async def verify(agent, **overrides) -> dict:
    return json.loads(await agent.verify_identity(**{**SAM, **overrides}))


def in_days(days: int) -> str:
    return (datetime.now(BANK_TZ).date() + timedelta(days=days)).isoformat()


async def test_any_name_and_birthday_with_the_demo_card_verify(agent, published):
    assert await verify(agent) == {"verified": True}
    assert published["status_history"] == ["authenticated"]
    assert published["card"]["holder"] == "Sam Ortiz"
    assert published["attempts"] == [
        {
            "submitted": {
                "name": "Sam Ortiz",
                "dateOfBirth": "1984-03-03",
                "lastFour": "1111",
            },
            "authenticated": True,
        }
    ]


@pytest.mark.parametrize("card", ["0000", "4417", "1112"])
async def test_any_other_card_fails_without_saying_why(agent, published, card):
    assert await verify(agent, card_last_four=card) == {"verified": False}
    assert published["status"] == "unauthenticated"
    assert "card" not in published


async def test_three_failed_checks_lock_the_caller_out(agent, published):
    for _ in range(MAX_ATTEMPTS - 1):
        assert await verify(agent, card_last_four="0000") == {"verified": False}
    assert "lockedOut" not in published

    last = await verify(agent, card_last_four="0000")
    assert last == {"verified": False, "locked_out": True}
    assert published["lockedOut"] is True
    with pytest.raises(ToolError):
        await verify(agent)


async def test_nothing_happens_on_the_card_before_verification(agent, published):
    await verify(agent, card_last_four="0000")
    with pytest.raises(ToolError):
        await agent.look_up_recent_charges()
    with pytest.raises(ToolError):
        await agent.lock_card()
    with pytest.raises(ToolError):
        await agent.set_travel_notice("Japan", in_days(10))
    assert "card" not in published


async def test_a_charge_they_do_not_recognize_locks_the_card(agent, published):
    await verify(agent)
    charge = json.loads(await agent.look_up_recent_charges())["recent_charge"]
    assert charge["where"] == FLAGGED.place
    assert published["alert"]["madeByCardholder"] is None

    result = json.loads(await agent.report_purchase(made_by_cardholder=False))
    assert result["loss_prevention_case"] == LOSS_PREVENTION_CASE
    assert published["alert"]["madeByCardholder"] is False
    assert published["card"]["state"] == "locked"
    assert published["card"]["caseId"] == LOSS_PREVENTION_CASE
