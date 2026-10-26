import json
from datetime import date, datetime, timedelta

import pytest
from livekit.agents.llm import ToolError

from bank import FLAGGED, LOSS_PREVENTION_CASE, Bank
from card_services import (
    BANK_TZ,
    FraudAlert,
    outbound_caller_script,
    outbound_greeting,
)


@pytest.fixture
def published() -> dict:
    return {}


@pytest.fixture
async def bank(published: dict) -> Bank:
    async def publish(key, value):
        published[key] = value

    bank = Bank(publish=publish, lookup_seconds=0)
    await bank.open_alert()
    return bank


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, bank: Bank) -> FraudAlert:
    monkeypatch.setenv("LIVEKIT_API_KEY", "unused")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "unused")
    return FraudAlert(bank)


def today() -> date:
    return datetime.now(BANK_TZ).date()


async def confirm(agent) -> dict:
    return json.loads(await agent.verify_identity("Sam Ortiz", "1984-03-03"))


async def test_any_name_and_birthday_confirm_whoever_picked_up(agent, published):
    assert published["alert"]["checks"] == []
    assert published["card"]["holder"] is None

    confirmed = await confirm(agent)
    assert confirmed["verified"] is True
    assert confirmed["flagged_purchase"]["where"] == FLAGGED.place
    assert published["alert"]["checks"] == [True]
    assert published["card"]["holder"] == "Sam Ortiz"
    assert published["status"] == "authenticated"


@pytest.mark.parametrize("born", ["2999-01-01", "March 3rd"])
async def test_a_date_that_cannot_be_a_birthday_is_asked_again(agent, published, born):
    with pytest.raises(ToolError):
        await agent.verify_identity("Sam Ortiz", born)
    assert published["alert"]["checks"] == []
    assert "status" not in published


async def test_nothing_is_recorded_before_verification(agent):
    with pytest.raises(ToolError):
        await agent.report_purchase(made_by_cardholder=False)


async def test_a_purchase_they_did_not_make_locks_the_card(agent, published):
    await confirm(agent)
    result = json.loads(await agent.report_purchase(made_by_cardholder=False))
    assert result["card"] == "locked"
    assert result["loss_prevention_case"] == LOSS_PREVENTION_CASE
    assert published["alert"]["madeByCardholder"] is False
    assert published["card"]["state"] == "locked"
    assert published["card"]["caseId"] == LOSS_PREVENTION_CASE
    with pytest.raises(ToolError):
        await agent.set_travel_notice("Italy", str(today() + timedelta(days=7)))


async def test_a_purchase_they_made_takes_a_travel_notice(agent, published):
    await confirm(agent)
    await agent.report_purchase(made_by_cardholder=True)
    assert published["card"]["state"] == "active"
    back = today() + timedelta(days=14)
    said = await agent.set_travel_notice("Italy", back.isoformat())
    assert "Italy" in said
    assert published["card"]["travel"] == {
        "destination": "Italy",
        "until": back.isoformat(),
    }


async def test_a_travel_notice_needs_a_return_date_that_makes_sense(agent):
    await confirm(agent)
    await agent.report_purchase(made_by_cardholder=True)
    with pytest.raises(ToolError):
        await agent.set_travel_notice("Italy", "2020-01-01")
    with pytest.raises(ToolError):
        await agent.set_travel_notice("Italy", str(today() + timedelta(days=400)))


def test_the_greeting_explains_the_demo_and_asks_who_picked_up():
    greeting = outbound_greeting()
    assert "demo" in greeting
    assert "one-one-one-one" in greeting
    assert greeting.endswith("Who am I speaking with?")


def test_the_voice_model_only_learns_the_purchase_from_the_check(agent):
    assert FLAGGED.merchant not in agent.instructions
    assert FLAGGED.amount not in agent.instructions


async def test_script_follows_the_call_from_verification_to_the_travel_notice(
    agent, bank
):
    assert "date of birth" in outbound_caller_script(bank)
    await confirm(agent)
    assert "that was them" in outbound_caller_script(bank)
    await agent.report_purchase(made_by_cardholder=True)
    assert "two weeks" in outbound_caller_script(bank)
    await agent.set_travel_notice("Italy", str(today() + timedelta(days=14)))
    assert "goodbye" in outbound_caller_script(bank)
