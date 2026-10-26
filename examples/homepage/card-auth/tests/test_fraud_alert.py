import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pytest
from livekit.agents.llm import ToolError

from bank import FLAGGED, LOSS_PREVENTION_CASE, ON_FILE, Bank
from card_services import (
    BANK_TZ,
    CallState,
    FraudAlert,
    outbound_caller_script,
    outbound_greeting,
)

JORDAN = ON_FILE[0]


@dataclass
class FakeContext:
    userdata: CallState


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


@pytest.fixture
def context() -> FakeContext:
    return FakeContext(CallState())


def today() -> date:
    return datetime.now(BANK_TZ).date()


async def test_the_purchase_stays_hidden_until_the_birth_date_matches(agent, context):
    missed = json.loads(await agent.verify_birth_date(context, "1990-04-23"))
    assert missed == {"verified": False}
    assert context.userdata.cardholder is None

    verified = json.loads(await agent.verify_birth_date(context, JORDAN.date_of_birth))
    assert verified["verified"] is True
    assert verified["flagged_purchase"]["where"] == FLAGGED.place
    assert context.userdata.cardholder == JORDAN


async def test_nothing_is_recorded_before_verification(agent, context):
    with pytest.raises(ToolError):
        await agent.report_purchase(context, made_by_cardholder=False)


async def test_a_purchase_they_did_not_make_locks_the_card(agent, context, published):
    await agent.verify_birth_date(context, JORDAN.date_of_birth)
    result = json.loads(await agent.report_purchase(context, made_by_cardholder=False))
    assert result["card"] == "locked"
    assert result["loss_prevention_case"] == LOSS_PREVENTION_CASE
    assert published["alert"]["card"] == "locked"
    assert published["alert"]["caseId"] == LOSS_PREVENTION_CASE
    with pytest.raises(ToolError):
        await agent.set_travel_notice(
            context, "Italy", str(today() + timedelta(days=7))
        )


async def test_a_purchase_they_made_takes_a_travel_notice(agent, context, published):
    await agent.verify_birth_date(context, JORDAN.date_of_birth)
    await agent.report_purchase(context, made_by_cardholder=True)
    assert published["alert"]["card"] == "active"
    back = today() + timedelta(days=14)
    said = await agent.set_travel_notice(context, "Italy", back.isoformat())
    assert "Italy" in said
    assert published["alert"]["travel"] == {
        "destination": "Italy",
        "until": back.isoformat(),
    }


async def test_a_travel_notice_needs_a_return_date_that_makes_sense(agent, context):
    await agent.verify_birth_date(context, JORDAN.date_of_birth)
    await agent.report_purchase(context, made_by_cardholder=True)
    with pytest.raises(ToolError):
        await agent.set_travel_notice(context, "Italy", "2020-01-01")
    with pytest.raises(ToolError):
        await agent.set_travel_notice(
            context, "Italy", str(today() + timedelta(days=400))
        )


def test_the_greeting_names_the_cardholder_but_not_their_birth_date():
    greeting = outbound_greeting()
    assert JORDAN.name in greeting and JORDAN.last_four in greeting
    assert JORDAN.date_of_birth not in greeting


def test_the_models_are_never_told_the_birth_date_or_the_purchase(agent):
    assert JORDAN.date_of_birth not in agent.instructions
    assert FLAGGED.merchant not in agent.instructions
    assert FLAGGED.amount not in agent.instructions


async def test_script_follows_the_call_from_verification_to_the_travel_notice(
    agent, context, bank
):
    assert "date of birth" in outbound_caller_script(bank)
    await agent.verify_birth_date(context, JORDAN.date_of_birth)
    assert "that was them" in outbound_caller_script(bank)
    await agent.report_purchase(context, made_by_cardholder=True)
    assert "two weeks" in outbound_caller_script(bank)
    await agent.set_travel_notice(context, "Italy", str(today() + timedelta(days=14)))
    assert "goodbye" in outbound_caller_script(bank)
