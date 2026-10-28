import json
from datetime import datetime, timedelta

import pytest
from livekit.agents.llm import ToolError

from bank import FLAGGED, LOSS_PREVENTION_CASE, MAX_ATTEMPTS, Bank
from card_services import BANK_TZ, GREETING, CardServices

SAM = {
    "full_name": "Sam Ortiz",
    "date_of_birth": "1984-03-03",
    "card_last_four": "1111",
    "request": "other",
}


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, published: dict) -> CardServices:
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")

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
    assert published["status_history"] == ["authenticating", "authenticated"]
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


@pytest.mark.parametrize("card", ["11 11", "1-1-1-1", "ending in 1111"])
async def test_spoken_formats_of_the_card_are_accepted(agent, card):
    assert await verify(agent, card_last_four=card) == {"verified": True}


@pytest.mark.parametrize(
    "unusable",
    [
        {"full_name": " "},
        {"date_of_birth": "April 22, 1990"},
        {"date_of_birth": "2999-01-01"},
        {"card_last_four": "111"},
    ],
)
async def test_details_that_cannot_be_right_are_asked_again_not_checked(
    agent, published, unusable
):
    with pytest.raises(ToolError):
        await verify(agent, **unusable)
    assert "attempts" not in published


async def test_a_failed_check_can_be_retried(agent, published):
    await verify(agent, card_last_four="0000")
    assert await verify(agent) == {"verified": True}
    assert [a["authenticated"] for a in published["attempts"]] == [False, True]


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


async def test_requests_wait_for_verification(agent, published):
    for request in ("charge", "lock"):
        result = await verify(agent, card_last_four="0000", request=request)
        assert result == {"verified": False}
    assert "card" not in published
    assert "alert" not in published


async def test_a_charge_request_comes_back_with_the_charge(agent, published):
    result = await verify(agent, request="charge")
    assert result["verified"] is True
    assert result["recent_charge"]["where"] == FLAGGED.place
    assert published["alert"]["madeByCardholder"] is None


async def test_a_lock_request_locks_the_card_once_verified(agent, published):
    result = await verify(agent, request="lock")
    assert result["verified"] is True
    assert result["card"].startswith("locked")
    assert published["card"]["state"] == "locked"


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


async def test_a_lost_card_gets_locked(agent, published):
    await verify(agent)
    said = await agent.lock_card()
    assert "1111" in said
    assert published["card"]["state"] == "locked"
    assert published["card"]["caseId"] is None
    with pytest.raises(ToolError):
        await agent.set_travel_notice("Japan", in_days(10))


async def test_a_travel_notice_needs_no_flagged_purchase(agent, published):
    await verify(agent)
    said = await agent.set_travel_notice("Japan", in_days(10))
    assert "Japan" in said
    assert published["card"]["travel"] == {
        "destination": "Japan",
        "until": in_days(10),
    }
    assert "alert" not in published


def test_the_card_question_explains_the_demo_cards(agent):
    assert "four ones for a success, or four zeros for a failure" in agent.instructions
    assert "Can I get your name?" in GREETING
